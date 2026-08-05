import cupy as cp
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# Kernel translation units, in dependency order.  Single source of truth: tests that
# compile the kernels standalone import this rather than repeating the list.
# diva.cu sits right after stress.cu -- it depends only on common.cu and stress.cu, and
# residuals.cu / vanka.cu need diva_coeffs_cell<T> from it.
CUDA_FILES = ['common.cu', 'viscosity.cu', 'stress.cu', 'diva.cu', 'flux.cu',
              'residuals.cu', 'vanka.cu', 'grad.cu']

class ForwardOperators:
    def __init__(self,grid,
            use_fast_math=True):

        self.grid = grid

        cuda_dir = Path(__file__).parent / "cuda"

        # Concatenate ice kernel files in dependency order
        cuda_files = CUDA_FILES
        cuda_source = '\n'.join((cuda_dir / f).read_text() for f in cuda_files)
        
        if use_fast_math:
            options=("--use_fast_math",)
        else:
            options=()

        self.kernels = cp.RawModule(code=cuda_source, options=options)

        self.r_u = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.r_v = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.r_H = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)
        
        self.f_u = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.f_v = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.f_H = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)
        
        self.F_u = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.F_v = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.F_H = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)

        self.delta_u = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.delta_v = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.delta_H = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)

        self._var_u = None
        self._var_v = None
        self._var_H = None

        self._jvp_u = None
        self._jvp_v = None
        self._jvp_H = None

        # DIVA closure diagnostics, allocated on first use
        self._r_ub = None
        self._u_bar = None
        self._diva_caps = None

        # Gauss-Legendre quadrature rule for the vertical integrals, cached on n_sigma
        self._gl_n = None
        self._gl_zeta = None
        self._gl_w = None

        self.gamma = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)
        self.gamma.fill(grid.geometry.thklim.value)

        self.vanka_config = VankaConfig()



    def _quadrature(self):
        """Gauss-Legendre nodes and weights on [0,1] for DIVA's vertical integrals.

        The integrands are zeta/eta and zeta^2/eta, and eta ~ zeta^(1-n) where the shear
        dominates, so they behave like polynomials of degree n and n+1 -- which Gauss-Legendre
        integrates EXACTLY with ceil((n+2)/2) nodes.  Midpoint quadrature is only second order
        and needed 32+ nodes for what 4 Gauss nodes deliver; see notes/diva_numerics.md 5.10.

        Built on the host by numpy, so any n_sigma works with no table in the kernel, and cached
        because it changes only if n_sigma does.  Weights are scaled to sum to 1, so the eta
        accumulation is a depth AVERAGE with no further division.
        """
        n = int(self.grid.rheology.n_sigma.value)
        if self._gl_n != n:
            x, w = np.polynomial.legendre.leggauss(n)
            self._gl_zeta = cp.asarray(0.5*(x + 1.0), dtype=cp.float32)
            self._gl_w = cp.asarray(0.5*w, dtype=cp.float32)
            self._gl_n = n
        return self._gl_zeta, self._gl_w

    @property
    def _kernel_config(self):
        block_size = (16, 16)
        stride = 14
        halo = 1
        grid_size = (self.grid.nx // stride + 1, self.grid.ny // stride + 1)
        return grid_size, block_size, stride, halo

    def compute_residual(self, dt, 
            use_mask=True, 
            operator_only=False, 
            freeze_calving=False, 
            freeze_phi=False,
            return_norms=False):

        grid = self.grid
        state = grid.state
        geometry = grid.geometry
        rheology = grid.rheology
        sliding = grid.sliding
        calving = grid.calving
        forcing = grid.forcing

        diva = float(rheology.stress_balance.value) > 0.5
        kernel = self.kernels.get_function('compute_residual_diva' if diva
                                           else 'compute_residual')
        grid_size, block_size, stride, halo = self._kernel_config

        if freeze_calving:
            calving_rate = cp.float32(0.0)
        else:
            calving_rate = calving.calving_rate.value

        if not freeze_phi:
            self.compute_phi()

        if diva:
            # eta_bar/beta_eff are lagged fields, so they must be refreshed against the
            # current state before the residual is formed (Goldberg's iteration on
            # viscosity); SSA gets this for free by computing eta inline.
            self.compute_diva_coeffs()

        if operator_only:
            out_u = self.F_u
            out_v = self.F_v
            out_H = self.F_H
            use_forcing = False
        else:
            out_u = self.r_u
            out_v = self.r_v
            out_H = self.r_H
            use_forcing = True

        args = (out_u, out_v, out_H,
                state.u.data, state.v.data, state.H.data,
                state.phi.data, state.mask.data,
                self.f_u, self.f_v, self.f_H,
                geometry.bed.data,
                rheology.B.data,
                sliding.beta.data, sliding.u_c.data,
                self.gamma)
        if diva:
            args += (rheology.eta_bar.data, sliding.beta_eff.data)
        args += (use_forcing,use_mask,
                rheology.n.value, rheology.eps_reg.value,
                geometry.sigmoid_c.value,
                sliding.m.value, sliding.u_reg.value,
                sliding.water_drag.value, sliding.flotation_reg_sliding.value, sliding.sliding_law.value,
                calving_rate, calving.flotation_reg_calving.value,
                grid.dx, dt,
                grid.ny, grid.nx, stride, halo)

        kernel(grid_size, block_size, args)

        if return_norms:
            return cp.linalg.norm(out_u),cp.linalg.norm(out_v),cp.linalg.norm(out_H)

    def compute_jvp(self, dt, 
            use_mask=True, 
            freeze_calving=False, 
            freeze_phi=False,
            return_norms=False):

        diva = float(self.grid.rheology.stress_balance.value) > 0.5
        kernel = self.kernels.get_function('compute_jvp_diva' if diva
                                           else 'compute_jvp')
        grid_size, block_size, stride, halo = self._kernel_config
  
        grid = self.grid
        state = grid.state
        geometry = grid.geometry        
        rheology = grid.rheology
        sliding = grid.sliding
        calving = grid.calving
        forcing = grid.forcing

        if freeze_calving:
            calving_rate = cp.float32(0.0)
        else:
            calving_rate = calving.calving_rate.value

        if not freeze_phi:
            self.compute_phi()

        kernel(grid_size, block_size,
               (self.jvp_u, self.jvp_v, self.jvp_H,
                state.u.data, state.v.data, state.H.data, 
                self.var_u, self.var_v, self.var_H, 
                state.phi.data, state.mask.data,
                self.f_u, self.f_v, self.f_H,
                geometry.bed.data, 
                rheology.B.data, 
                sliding.beta.data, sliding.u_c.data,
                self.gamma,
                *((state.u_b.data,) if diva else ()),
                use_mask,
                rheology.n.value, rheology.eps_reg.value, 
                geometry.sigmoid_c.value,
                sliding.m.value, sliding.u_reg.value, 
                sliding.water_drag.value, sliding.flotation_reg_sliding.value, sliding.sliding_law.value,
                calving_rate, calving.flotation_reg_calving.value,
                grid.dx, dt,
                *((int(rheology.n_sigma.value), *self._quadrature()) if diva else ()),
                grid.ny, grid.nx, stride, halo)) 

    def compute_phi(self, relaxation=cp.float32(0.0)):
        kernel = self.kernels.get_function('compute_grounded')
        grid_size, block_size, stride, halo = self._kernel_config
            
        grid = self.grid
        kernel(grid_size, block_size,
                   (grid.state.phi.data,
                    grid.state.H.data, grid.geometry.depth.data, 
                    grid.geometry.sigmoid_c.value,
                    grid.geometry.sigmoid_k.value,
                    relaxation,
                    grid.ny, grid.nx, 
                    stride, halo))

    def compute_diva_coeffs(self):
        """Diagnose the DIVA coefficients (eta_bar, F2) and the basal speed u_b from
        the current state.  Only meaningful when rheology.stress_balance = 1; the SSA
        path never calls this."""
        kernel = self.kernels.get_function('compute_diva_coeffs')
        grid_size, block_size, stride, halo = self._kernel_config
        if self._diva_caps is None:
            self._diva_caps = cp.zeros((self.grid.ny, self.grid.nx), dtype=cp.float32)

        grid = self.grid
        state = grid.state
        rheology = grid.rheology
        sliding = grid.sliding

        kernel(grid_size, block_size,
                   (rheology.eta_bar.data, rheology.F2.data, state.u_b.data,
                    sliding.beta_eff.data, rheology.F1.data, self._diva_caps,
                    *self._quadrature(),
                    state.u.data, state.v.data, state.H.data, state.phi.data,
                    rheology.B.data, sliding.beta.data, sliding.u_c.data,
                    sliding.m.value, sliding.u_reg.value,
                    sliding.water_drag.value, sliding.sliding_law.value,
                    rheology.n.value, rheology.eps_reg.value, grid.dx,
                    int(rheology.n_sigma.value),
                    grid.ny, grid.nx,
                    stride, halo))

    def compute_diva_closure_residual(self):
        """DIVA's SECOND convergence criterion: does the closure still hold at the current
        velocity?

        eta_bar and beta_eff are frozen fields refreshed between sweeps, so a zero momentum
        residual only says the equations hold for THOSE coefficients.  This returns
        (|r_Ub|, |Ubar|) so the caller can report |r_Ub|/|Ubar| -- the coefficients' staleness.
        Zero right after a refresh, grows as the smoother moves the velocity, and vanishes only
        at convergence.

        Deliberately returned separately rather than folded into the momentum norm: r_Ub is a
        velocity residual and r_u is a momentum one (see notes/open_questions.md Q6).
        """
        if float(self.grid.rheology.stress_balance.value) < 0.5:
            return None, None
        grid = self.grid
        if self._r_ub is None:
            self._r_ub = cp.zeros((grid.ny, grid.nx), dtype=cp.float32)
            self._u_bar = cp.zeros((grid.ny, grid.nx), dtype=cp.float32)

        kernel = self.kernels.get_function('compute_diva_closure_residual')
        grid_size, block_size, stride, halo = self._kernel_config
        sliding = grid.sliding
        kernel(grid_size, block_size,
               (self._r_ub, self._u_bar,
                grid.state.u.data, grid.state.v.data, grid.state.phi.data,
                sliding.beta.data, sliding.u_c.data,
                grid.state.u_b.data, grid.rheology.F2.data,
                sliding.m.value, sliding.u_reg.value,
                sliding.water_drag.value, sliding.sliding_law.value,
                grid.ny, grid.nx, stride, halo))
        return cp.linalg.norm(self._r_ub), cp.linalg.norm(self._u_bar)

    def diva_cap_counts(self):
        """How many cells failed to reach tolerance in the local closure solves.

        Returns (closure_capped, eta_capped) as cell counts.  Both should be zero; a nonzero
        value means an adaptive loop hit its backstop and the coefficients in those cells are
        not converged.  Exists because an adaptive loop that silently caps is worse than a
        fixed count -- it looks converged, which is how the two earlier closure defects
        survived undetected."""
        if self._diva_caps is None:
            return 0, 0
        flags = self._diva_caps
        return int(cp.count_nonzero(cp.mod(flags, 2.0) >= 0.5)), \
               int(cp.count_nonzero(flags >= 2.0))

    def compute_diva_derivs(self):
        """Total derivatives of the cell-local DIVA closure, one dual seeding per input:
        eps_mem^2 and Ubar for the state transpose, then beta, u_c and m for the parameter
        gradients.  Needed only by the adjoint."""
        kernel = self.kernels.get_function('compute_diva_derivs')
        grid_size, block_size, stride, halo = self._kernel_config

        grid = self.grid
        state = grid.state
        rheology = grid.rheology
        sliding = grid.sliding

        kernel(grid_size, block_size,
                   (rheology.deta_deps.data, rheology.deta_dU.data,
                    rheology.dbe_deps.data, rheology.dbe_dU.data,
                    rheology.deta_dbeta.data, rheology.dbe_dbeta.data,
                    rheology.deta_duc.data, rheology.dbe_duc.data,
                    rheology.deta_dm.data, rheology.dbe_dm.data,
                    state.u.data, state.v.data, state.H.data, state.phi.data,
                    rheology.B.data, sliding.beta.data, sliding.u_c.data, state.u_b.data,
                    sliding.m.value, sliding.u_reg.value,
                    sliding.water_drag.value, sliding.sliding_law.value,
                    rheology.n.value, rheology.eps_reg.value, grid.dx,
                    int(rheology.n_sigma.value),
                    *self._quadrature(),
                    grid.ny, grid.nx,
                    stride, halo))

    def vanka_smooth(self, dt,
            freeze_calving=False,
            freeze_phi=False):

        grid = self.grid
        state = grid.state
        geometry = grid.geometry
        rheology = grid.rheology
        sliding = grid.sliding
        calving = grid.calving
        forcing = grid.forcing

        diva = float(rheology.stress_balance.value) > 0.5
        kernel = self.kernels.get_function('vanka_smooth_diva' if diva
                                           else 'vanka_smooth')
        grid_size, block_size, stride, halo = self._kernel_config

        if freeze_calving:
            calving_rate = cp.float32(0.0)
        else:
            calving_rate = calving.calving_rate.value

        if not freeze_phi:
            self.compute_phi(relaxation=self.vanka_config.relax_phi)

        if diva:
            # Refresh the lagged coefficients against the current state before the
            # block solve -- Goldberg's iteration on viscosity (his eqs 41-44).
            self.compute_diva_coeffs()

        self.delta_u.fill(0.0)
        self.delta_v.fill(0.0)
        self.delta_H.fill(0.0)
        args = (self.delta_u, self.delta_v, self.delta_H,
                state.mask.data,
                state.u.data, state.v.data, state.H.data,
                state.phi.data,
                self.f_u, self.f_v, self.f_H,
                geometry.bed.data, rheology.B.data, sliding.beta.data, sliding.u_c.data,
                self.gamma)
        if diva:
            args += (rheology.eta_bar.data, sliding.beta_eff.data,
                     state.u_b.data, rheology.F2.data)
        args += (rheology.n.value, rheology.eps_reg.value, 
                geometry.sigmoid_c.value,
                sliding.m.value, sliding.u_reg.value, 
                sliding.water_drag.value, 
                sliding.flotation_reg_sliding.value, sliding.sliding_law.value,
                calving_rate, 
                calving.flotation_reg_calving.value,
                grid.dx, dt,
                grid.ny, grid.nx, stride, halo,
                self.vanka_config.newton_config.steps,
                self.vanka_config.newton_config.relaxation,
                self.vanka_config.newton_config.ssa_damping,
                self.vanka_config.newton_config.mc_damping)

        kernel(grid_size, block_size, args)

    def vanka_sweep(self, dt, n_iter, 
            freeze_calving=False,
            freeze_phi=False):
        for i in range(n_iter):
            self.vanka_smooth(dt,freeze_calving=freeze_calving,freeze_phi=freeze_phi)
            self.grid.state.u.data[:] += self.vanka_config.omega * self.delta_u
            self.grid.state.v.data[:] += self.vanka_config.omega * self.delta_v
            self.grid.state.H.data[:] += self.vanka_config.omega * self.delta_H
            self.vanka_config.hook_func(i)

    def vanka_dump(self,dt):
        kernel = self.kernels.get_function('vanka_dump')
        grid_size, block_size, stride, halo = self._kernel_config
        grid = self.grid

        J = cp.zeros((self.grid.ny*self.grid.nx,25),dtype=cp.float32)
        r = cp.zeros((self.grid.ny*self.grid.nx,5),dtype=cp.float32)
        kernel(grid_size, block_size,
               (J,r,
                grid.state.u.data, grid.state.v.data, grid.state.H.data, grid.state.phi.data,
                self.f_u, self.f_v, self.f_H,
                grid.geometry.bed.data, grid.rheology.B.data, grid.sliding.beta.data, grid.sliding.u_c.data, self.gamma,
                grid.rheology.n.value, grid.rheology.eps_reg.value, grid.geometry.sigmoid_c.value,
                grid.sliding.m.value, grid.sliding.u_reg.value, grid.sliding.water_drag.value, grid.sliding.flotation_reg_sliding.value, grid.sliding.sliding_law.value,
                grid.calving.calving_rate.value, grid.calving.flotation_reg_calving.value,
                grid.dx, dt,
                grid.ny, grid.nx, stride, halo,
                )
        )

        return J,r
            
    def set_rhs(self,dt):
        self.f_H[:,:] = self.grid.state.H_prev.data/dt + self.grid.forcing.smb.data

    @property
    def var_u(self):
        if self._var_u is None:
            self._var_u = cp.zeros((self.grid.ny,self.grid.nx+1),dtype=cp.float32)
        return self._var_u
    
    @property
    def var_v(self):
        if self._var_v is None:
            self._var_v = cp.zeros((self.grid.ny+1,self.grid.nx),dtype=cp.float32)
        return self._var_v
    
    @property
    def var_H(self):
        if self._var_H is None:
            self._var_H = cp.zeros((self.grid.ny,self.grid.nx),dtype=cp.float32)
        return self._var_H

    @property
    def jvp_u(self):
        if self._jvp_u is None:
            self._jvp_u = cp.zeros((self.grid.ny,self.grid.nx+1),dtype=cp.float32)
        return self._jvp_u
    
    @property
    def jvp_v(self):
        if self._jvp_v is None:
            self._jvp_v = cp.zeros((self.grid.ny+1,self.grid.nx),dtype=cp.float32)
        return self._jvp_v
    
    @property
    def jvp_H(self):
        if self._jvp_H is None:
            self._jvp_H = cp.zeros((self.grid.ny,self.grid.nx),dtype=cp.float32)
        return self._jvp_H

@dataclass
class NewtonConfig:
    steps: int = 30
    relaxation: cp.float32 = cp.float32(0.5)
    ssa_damping: cp.float32 = cp.float32(0.01)
    mc_damping: cp.float32 = cp.float32(1.0)

@dataclass
class VankaConfig:
    omega: cp.float32 = cp.float32(0.5)
    newton_config: NewtonConfig = field(default_factory = lambda: NewtonConfig())
    relax_phi: cp.float32 = cp.float32(0.0)
    hook_interval: int = 1
    hook_func: Callable[[int],None] = field(default_factory = lambda: lambda i:None)


class AdjointOperators:
    def __init__(self,grid,
            use_fast_math=True):

        self.grid = grid

        cuda_dir = Path(__file__).parent / "cuda"

        # Concatenate ice kernel files in dependency order
        cuda_files = CUDA_FILES
        cuda_source = '\n'.join((cuda_dir / f).read_text() for f in cuda_files)
        
        if use_fast_math:
            options=("--use_fast_math",)
        else:
            options=()

        self.kernels = cp.RawModule(code=cuda_source, options=options)

        self.r_u = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.r_v = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.r_H = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)
        
        self.f_u = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.f_v = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.f_H = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)
        
        self.vjp_u = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.vjp_v = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.vjp_H = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)
        
        self.delta_lambda_u = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.delta_lambda_v = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.delta_lambda_H = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)

        # DIVA: per-cell adjoints of the closure coefficients -- the intermediate the
        # transpose is split at (see _apply_diva_coeff_adjoints).
        #
        # On by default: the adjoint residual is then the exact transpose (dot-product
        # identity 9.7e-2 -> 5e-7, i.e. round-off).  The smoother still assembles the
        # frozen block, which is fine -- it is only a preconditioner, exactly as in SSA,
        # where the VJP carries d(eta)/du and the smoother does not.
        self.diva_exact_coeff_adjoint = True
        self.W_eta = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)
        self.W_be = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)

        self.gamma = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)
        self.gamma.fill(grid.geometry.thklim.value)

        self.vanka_config = VankaConfig()

    @property
    def _kernel_config(self):
        block_size = (16, 16)
        stride = 14
        halo = 1
        grid_size = (self.grid.nx // stride + 1, self.grid.ny // stride + 1)
        return grid_size, block_size, stride, halo

    def compute_residual(self, dt, 
            use_mask=True, 
            freeze_calving=False, 
            return_norms=False):

        diva = float(self.grid.rheology.stress_balance.value) > 0.5
        kernel = self.kernels.get_function('compute_vjp_diva' if diva
                                           else 'compute_vjp')
        grid_size, block_size, stride, halo = self._kernel_config
  
        grid = self.grid
        state = grid.state
        adjoint = grid.adjoint
        geometry = grid.geometry        
        rheology = grid.rheology
        sliding = grid.sliding
        calving = grid.calving
        forcing = grid.forcing

        if freeze_calving:
            calving_rate = cp.float32(0.0)
        else:
            calving_rate = calving.calving_rate.value

        self.r_u.fill(0)
        self.r_v.fill(0)
        self.r_H.fill(0)
        use_forcing=True
        diva_exact = diva and self.diva_exact_coeff_adjoint
        self._last_dt = dt
        if diva:
            self.W_eta.fill(0.0)
            self.W_be.fill(0.0)

        kernel(grid_size, block_size,
               (self.r_u, self.r_v, self.r_H,
                state.u.data, state.v.data, state.H.data, 
                adjoint.lambda_u.data, adjoint.lambda_v.data, adjoint.lambda_H.data, 
                state.phi.data, state.mask.data,
                self.f_u, self.f_v, self.f_H,
                geometry.bed.data, 
                rheology.B.data, 
                sliding.beta.data, sliding.u_c.data,
                self.gamma,
                *((rheology.eta_bar.data, sliding.beta_eff.data,
                   self.W_eta, self.W_be) if diva else ()),
                use_forcing, use_mask,
                rheology.n.value, rheology.eps_reg.value, 
                geometry.sigmoid_c.value,
                sliding.m.value, sliding.u_reg.value, 
                sliding.water_drag.value, sliding.flotation_reg_sliding.value, sliding.sliding_law.value,
                calving_rate, calving.flotation_reg_calving.value,
                grid.dx, dt,
                grid.ny, grid.nx, stride, halo)) 

        if diva_exact:
            self._apply_diva_coeff_adjoints(self.r_u, self.r_v)

        if return_norms:
            return cp.linalg.norm(self.r_u),cp.linalg.norm(self.r_v),cp.linalg.norm(self.r_H)



    def _apply_diva_coeff_adjoints(self, out_u, out_v):
        """Second half of the DIVA transpose.  vjp_body pushes
        lambda_row * d(r_row)/d(coefficient) onto the owning cell, filling W_eta and
        W_be; this converts those per-cell coefficient adjoints into velocity
        sensitivities and adds them into out_u/out_v.

        No symmetry is assumed: this is the honest transpose of the four closure paths
        the frozen adjoint omitted.  Splitting the transpose at the cell is what keeps
        it feasible -- the composite row->facet dependence reaches +/-2, but row->cell
        and cell->facet are each +/-1, so neither half needs a wider halo."""
        kernel = self.kernels.get_function('compute_diva_vjp_coeffs')
        grid_size, block_size, stride, halo = self._kernel_config

        grid = self.grid
        rheology = grid.rheology
        grid.forward_operators.compute_diva_derivs()

        kernel(grid_size, block_size,
               (out_u, out_v,
                grid.state.u.data, grid.state.v.data,
                self.W_eta, self.W_be,
                rheology.deta_deps.data, rheology.deta_dU.data,
                rheology.dbe_deps.data, rheology.dbe_dU.data,
                grid.dx,
                grid.ny, grid.nx, stride, halo))

    def compute_vjp(self, dt,
            use_mask=True,
            use_forcing=False,
            freeze_calving=False):

        diva = float(self.grid.rheology.stress_balance.value) > 0.5
        kernel = self.kernels.get_function('compute_vjp_diva' if diva
                                           else 'compute_vjp')
        grid_size, block_size, stride, halo = self._kernel_config
  
        grid = self.grid
        state = grid.state
        adjoint = grid.adjoint
        geometry = grid.geometry        
        rheology = grid.rheology
        sliding = grid.sliding
        calving = grid.calving
        forcing = grid.forcing

        if freeze_calving:
            calving_rate = cp.float32(0.0)
        else:
            calving_rate = calving.calving_rate.value

        use_forcing=False
        self.vjp_u.fill(0)
        self.vjp_v.fill(0)
        self.vjp_H.fill(0)
        diva_exact = diva and self.diva_exact_coeff_adjoint
        self._last_dt = dt
        if diva:
            self.W_eta.fill(0.0)
            self.W_be.fill(0.0)

        kernel(grid_size, block_size,
               (self.vjp_u, self.vjp_v, self.vjp_H,
                state.u.data, state.v.data, state.H.data, 
                adjoint.lambda_u.data, adjoint.lambda_v.data, adjoint.lambda_H.data, 
                state.phi.data, state.mask.data,
                self.f_u, self.f_v, self.f_H,
                geometry.bed.data, 
                rheology.B.data, 
                sliding.beta.data, sliding.u_c.data,
                self.gamma,
                *((rheology.eta_bar.data, sliding.beta_eff.data,
                   self.W_eta, self.W_be) if diva else ()),
                use_forcing, use_mask,
                rheology.n.value, rheology.eps_reg.value, 
                geometry.sigmoid_c.value,
                sliding.m.value, sliding.u_reg.value, 
                sliding.water_drag.value, sliding.flotation_reg_sliding.value, sliding.sliding_law.value,
                calving_rate, calving.flotation_reg_calving.value,
                grid.dx, dt,
                grid.ny, grid.nx, stride, halo)) 

        if diva_exact:
            self._apply_diva_coeff_adjoints(self.vjp_u, self.vjp_v)


    def vanka_smooth(self, dt,
            freeze_calving=False):

        diva = float(self.grid.rheology.stress_balance.value) > 0.5
        kernel = self.kernels.get_function('vanka_smooth_adjoint_diva' if diva
                                           else 'vanka_smooth_adjoint')
        grid_size, block_size, stride, halo = self._kernel_config

        grid = self.grid
        state = grid.state
        geometry = grid.geometry        
        rheology = grid.rheology
        sliding = grid.sliding
        calving = grid.calving
        forcing = grid.forcing

        if freeze_calving:
            calving_rate = cp.float32(0.0)
        else:
            calving_rate = calving.calving_rate.value

        self.delta_lambda_u.fill(0.0)
        self.delta_lambda_v.fill(0.0)
        self.delta_lambda_H.fill(0.0)
        kernel(grid_size, block_size,
               (self.delta_lambda_u, self.delta_lambda_v, self.delta_lambda_H, 
                state.u.data, state.v.data, state.H.data, 
                state.phi.data, state.mask.data,
                self.r_u, self.r_v, self.r_H,
                geometry.bed.data, rheology.B.data, sliding.beta.data, sliding.u_c.data, 
                self.gamma,
                *((rheology.eta_bar.data, sliding.beta_eff.data) if diva else ()),
                rheology.n.value, rheology.eps_reg.value, 
                geometry.sigmoid_c.value,
                sliding.m.value, sliding.u_reg.value, 
                sliding.water_drag.value, 
                sliding.flotation_reg_sliding.value, sliding.sliding_law.value,
                calving_rate, 
                calving.flotation_reg_calving.value,
                grid.dx, dt,
                grid.ny, grid.nx, stride, halo,
                self.vanka_config.newton_config.ssa_damping,
                self.vanka_config.newton_config.mc_damping)
        )

    def vanka_sweep(self, dt, n_iter, 
            freeze_calving=False):
        for i in range(n_iter):
            self.compute_residual(dt,freeze_calving=freeze_calving)
            self.vanka_smooth(dt,freeze_calving=freeze_calving)
            self.grid.adjoint.lambda_u.data[:] -= self.vanka_config.omega * self.delta_lambda_u
            self.grid.adjoint.lambda_v.data[:] -= self.vanka_config.omega * self.delta_lambda_v
            self.grid.adjoint.lambda_H.data[:] -= self.vanka_config.omega * self.delta_lambda_H

    def compute_gradient_beta(self):
        if float(self.grid.rheology.stress_balance.value) > 0.5:
            rheology = self.grid.rheology
            return self._diva_param_gradient(self.grid.sliding.beta.grad,
                                             rheology.deta_dbeta, rheology.dbe_dbeta)
        kernel = self.kernels.get_function('compute_gradient_beta')
        grid_size, block_size, stride, halo = self._kernel_config

        grid = self.grid
        state = grid.state
        adjoint = grid.adjoint
        geometry = grid.geometry        
        rheology = grid.rheology
        sliding = grid.sliding
        calving = grid.calving
        forcing = grid.forcing

        sliding.beta.grad.fill(0)
        kernel(grid_size, block_size,
               (sliding.beta.grad,
                state.u.data, state.v.data, state.H.data, 
                adjoint.lambda_u.data, adjoint.lambda_v.data, adjoint.lambda_H.data, 
                state.phi.data, state.mask.data,
                geometry.bed.data, 
                rheology.B.data, 
                sliding.beta.data, sliding.u_c.data,
                self.gamma,
                rheology.n.value, rheology.eps_reg.value, 
                geometry.sigmoid_c.value,
                sliding.m.value, sliding.u_reg.value, 
                sliding.water_drag.value, sliding.flotation_reg_sliding.value, sliding.sliding_law.value,
                calving.calving_rate.value, calving.flotation_reg_calving.value,
                grid.dx, cp.float32(0.0),
                grid.ny, grid.nx, stride, halo))

    def _diva_param_gradient(self, out, deta_dp, dbe_dp, reduce=False):
        """Any DIVA sliding-parameter gradient, as a cell-local product.

        W_eta and W_be (filled by the VJP) already are lambda^T d(r)/d(coefficient)
        summed over every row touching the cell, so the parameter gradient is just the
        chain rule per cell -- no facet loop.  Nothing here knows which parameter it is:
        beta, u_c and m differ only in which pair of derivative fields is passed, and m
        additionally reduces to a scalar.  Contrast the SSA path, which needs a separate
        facet-walking kernel per parameter because there the parameters enter the momentum
        stencils directly rather than through a state-dependent coefficient.

        The VJP is re-evaluated first so W corresponds to the current (converged) lambda
        rather than to whatever the last smoother sweep left behind.  That costs one
        residual plus one derivative kernel per call -- negligible next to the multigrid
        adjoint solve that precedes it, and it keeps the result independent of call order.
        """
        grid = self.grid

        self.compute_residual(getattr(self, '_last_dt', cp.float32(1.0)), use_mask=False)
        grid.forward_operators.compute_diva_derivs()

        name = 'compute_gradient_param_sum_diva' if reduce else 'compute_gradient_param_diva'
        kernel = self.kernels.get_function(name)
        grid_size, block_size, stride, halo = self._kernel_config
        # Required for the reducing variant (it accumulates); belt-and-braces for the
        # per-cell one, whose tiling writes every cell exactly once.
        out.fill(0)
        kernel(grid_size, block_size,
               (out, self.W_eta, self.W_be, deta_dp.data, dbe_dp.data,
                grid.ny, grid.nx, stride, halo))

    def compute_gradient_m(self):
        """Gradient w.r.t. the global Weertman exponent m (a single scalar).
        Contracts the adjoint state with d(tau_b)/dm over all momentum facets and
        stores the reduced value on sliding.m.grad."""
        if float(self.grid.rheology.stress_balance.value) > 0.5:
            rheology = self.grid.rheology
            grad_m = cp.zeros(1, dtype=cp.float32)
            self._diva_param_gradient(grad_m, rheology.deta_dm, rheology.dbe_dm,
                                      reduce=True)
            self.grid.sliding.m.grad = float(grad_m[0])
            return
        kernel = self.kernels.get_function('compute_gradient_m')
        grid_size, block_size, stride, halo = self._kernel_config

        grid = self.grid
        state = grid.state
        adjoint = grid.adjoint
        geometry = grid.geometry
        rheology = grid.rheology
        sliding = grid.sliding
        calving = grid.calving

        grad_m = cp.zeros(1, dtype=cp.float32)
        kernel(grid_size, block_size,
               (grad_m,
                state.u.data, state.v.data, state.H.data,
                adjoint.lambda_u.data, adjoint.lambda_v.data, adjoint.lambda_H.data,
                state.phi.data, state.mask.data,
                geometry.bed.data,
                rheology.B.data,
                sliding.beta.data, sliding.u_c.data,
                self.gamma,
                rheology.n.value, rheology.eps_reg.value,
                geometry.sigmoid_c.value,
                sliding.m.value, sliding.u_reg.value,
                sliding.water_drag.value, sliding.flotation_reg_sliding.value, sliding.sliding_law.value,
                calving.calving_rate.value, calving.flotation_reg_calving.value,
                grid.dx, cp.float32(0.0),
                grid.ny, grid.nx, stride, halo))
        sliding.m.grad = float(grad_m[0])

    def compute_gradient_u_c(self):
        if float(self.grid.rheology.stress_balance.value) > 0.5:
            rheology = self.grid.rheology
            return self._diva_param_gradient(self.grid.sliding.u_c.grad,
                                             rheology.deta_duc, rheology.dbe_duc)
        kernel = self.kernels.get_function('compute_gradient_u_c')
        grid_size, block_size, stride, halo = self._kernel_config

        grid = self.grid
        state = grid.state
        adjoint = grid.adjoint
        geometry = grid.geometry        
        rheology = grid.rheology
        sliding = grid.sliding
        calving = grid.calving
        forcing = grid.forcing

        sliding.u_c.grad.fill(0)
        kernel(grid_size, block_size,
               (sliding.u_c.grad,
                state.u.data, state.v.data, state.H.data, 
                adjoint.lambda_u.data, adjoint.lambda_v.data, adjoint.lambda_H.data, 
                state.phi.data, state.mask.data,
                geometry.bed.data, 
                rheology.B.data, 
                sliding.beta.data, sliding.u_c.data,
                self.gamma,
                rheology.n.value, rheology.eps_reg.value, 
                geometry.sigmoid_c.value,
                sliding.m.value, sliding.u_reg.value, 
                sliding.water_drag.value, sliding.flotation_reg_sliding.value, sliding.sliding_law.value,
                calving.calving_rate.value, calving.flotation_reg_calving.value,
                grid.dx, cp.float32(0.0),
                grid.ny, grid.nx, stride, halo))

    def compute_gradient_bed(self):
        kernel = self.kernels.get_function('compute_gradient_bed')
        grid_size, block_size, stride, halo = self._kernel_config

        grid = self.grid
        state = grid.state
        adjoint = grid.adjoint
        geometry = grid.geometry        
        rheology = grid.rheology
        sliding = grid.sliding
        calving = grid.calving
        forcing = grid.forcing

        geometry.bed.grad.fill(0)
        kernel(grid_size, block_size,
               (geometry.bed.grad,
                state.u.data, state.v.data, state.H.data, 
                adjoint.lambda_u.data, adjoint.lambda_v.data, adjoint.lambda_H.data, 
                state.phi.data, state.mask.data,
                geometry.bed.data, 
                rheology.B.data, 
                sliding.beta.data, sliding.u_c.data,
                self.gamma,
                rheology.n.value, rheology.eps_reg.value, 
                geometry.sigmoid_c.value,
                sliding.m.value, sliding.u_reg.value, 
                sliding.water_drag.value, sliding.flotation_reg_sliding.value, sliding.sliding_law.value,
                calving.calving_rate.value, calving.flotation_reg_calving.value,
                grid.dx, cp.float32(0.0),
                grid.ny, grid.nx, stride, halo)) 


    def compute_gradient_H_prev(self, dt):
        self.grid.state.H_prev.grad[:,:] = -self.grid.adjoint.lambda_H.data[:,:]/dt

    def compute_gradient_smb(self):
        self.grid.forcing.smb.grad[:,:] = -self.grid.adjoint.lambda_H.data[:,:] 
  
    



