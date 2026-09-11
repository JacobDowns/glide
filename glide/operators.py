import cupy as cp
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

class ForwardOperators:
    def __init__(self,grid,
            use_fast_math=True):

        self.grid = grid

        cuda_dir = Path(__file__).parent / "cuda"

        # Concatenate ice kernel files in dependency order
        cuda_files = ['common.cu', 'viscosity.cu', 'stress.cu', 'diva.cu',
                          'flux.cu', 'residuals.cu', 'vanka.cu', 'grad.cu']
        cuda_source = '\n'.join((cuda_dir / f).read_text() for f in cuda_files)
        
        if use_fast_math:
            options=("--use_fast_math",)
        else:
            options=()
        # Compile-time stress scheme specialization: the SSA build removes
        # all deformational physics and solves 5x5 patches (see common.cu)
        options += ("-DGLIDE_MOLHO=%d" % (0 if grid.ssa else 1),)
        # DIVA rides the SSA-shaped build (GLIDE_MOLHO=0) and toggles the
        # closure-coefficient reads; only ever 1 when grid.ssa is also True.
        options += ("-DGLIDE_DIVA=%d" % (1 if grid.diva else 0),)

        self.kernels = cp.RawModule(code=cuda_source, options=options)

        self.r_u = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.r_v = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.r_ud = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.r_vd = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.r_H = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)
        
        self.f_u = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.f_v = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.f_ud = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.f_vd = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.f_H = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)
        
        self.F_u = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.F_v = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.F_ud = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.F_vd = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.F_H = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)

        self.delta_u = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.delta_v = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.delta_ud = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.delta_vd = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.delta_H = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)

        self._var_u = None
        self._var_v = None
        self._var_ud = None
        self._var_vd = None
        self._var_H = None

        self._jvp_u = None
        self._jvp_v = None
        self._jvp_ud = None
        self._jvp_vd = None
        self._jvp_H = None

        self.gamma = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)
        self.gamma.fill(grid.geometry.thklim.value)

        self.vanka_config = VankaConfig()

        # DIVA: cached Gauss-Legendre quadrature (rebuilt only when n_sigma changes) and the
        # per-cell closure-cap counter.  Unused (and unallocated) under SSA/MOLHO.
        self._gl_n = None
        self._gl_zeta = None
        self._gl_w = None
        self._diva_caps = None

    def _quadrature(self):
        """Gauss-Legendre nodes and weights on [0,1] for DIVA's vertical integrals.

        The integrands are zeta/eta and zeta^2/eta, and eta ~ zeta^(1-n) where the shear
        dominates, so Gauss-Legendre integrates them exactly with a handful of nodes (4 Gauss
        nodes match 32+ midpoint nodes).  Built on the host, cached
        on n_sigma, weights scaled to sum to 1 so the eta accumulation is a depth AVERAGE.
        """
        n = int(self.grid.rheology.n_sigma.value)
        if self._gl_n != n:
            x, w = np.polynomial.legendre.leggauss(n)
            self._gl_zeta = cp.asarray(0.5*(x + 1.0), dtype=cp.float32)
            self._gl_w = cp.asarray(0.5*w, dtype=cp.float32)
            self._gl_n = n
        return self._gl_zeta, self._gl_w

    def compute_diva_coeffs(self):
        """Diagnose the DIVA coefficients (eta_bar, F1, F2, beta_eff) and the basal/surface
        speeds (u_b, u_s) from the current state, via the per-cell column closure.  Only called
        on DIVA grids; the residual/JVP/Vanka read these as frozen coefficients."""
        kernel = self.kernels.get_function('compute_diva_coeffs')
        grid_size, block_size, stride, halo = self._kernel_config
        if self._diva_caps is None:
            self._diva_caps = cp.zeros((self.grid.ny, self.grid.nx), dtype=cp.float32)

        grid = self.grid
        state, rheology, sliding = grid.state, grid.rheology, grid.sliding

        kernel(grid_size, block_size,
                   (rheology.eta_bar.data, rheology.F2.data, state.u_b.data,
                    sliding.beta_eff.data, rheology.F1.data, state.u_s.data, self._diva_caps,
                    *self._quadrature(),
                    state.u.data, state.v.data, state.H.data, state.xi.data,
                    rheology.B.data, sliding.beta.data, sliding.u_c.data,
                    sliding.m.value, sliding.u_reg.value,
                    sliding.water_drag.value, sliding.sliding_law.value,
                    rheology.n.value, rheology.eps_reg.value,
                    rheology.eps_reg_shear.value, grid.dx,
                    int(rheology.n_sigma.value),
                    grid.ny, grid.nx,
                    stride, halo))

    def diva_cap_counts(self):
        """Cells whose closure Newton (bit 0) or per-level eta solve (bit 1) hit its iteration
        cap in the last compute_diva_coeffs.  An adaptive loop that silently caps looks converged,
        so this lets a test assert it did not.  Returns (n_closure_capped, n_eta_capped)."""
        if self._diva_caps is None:
            return 0, 0
        flags = self._diva_caps.astype(cp.int32)
        return int(cp.count_nonzero(flags & 1)), int(cp.count_nonzero(flags & 2))

    def compute_diva_derivs(self):
        """Total derivatives of the DIVA cell closure (one dual seeding per input): eta_bar,
        beta_eff and u_s w.r.t. eps_mem^2, Ubar and H (state legs) and beta, u_c, m (parameters).
        These are what the exact adjoint and the parameter gradients consume."""
        kernel = self.kernels.get_function('compute_diva_derivs')
        grid_size, block_size, stride, halo = self._kernel_config
        grid = self.grid
        state, rheology, sliding = grid.state, grid.rheology, grid.sliding
        kernel(grid_size, block_size,
               (rheology.deta_deps.data, rheology.deta_dU.data, rheology.dbe_deps.data, rheology.dbe_dU.data,
                rheology.deta_dbeta.data, rheology.dbe_dbeta.data, rheology.deta_duc.data, rheology.dbe_duc.data,
                rheology.deta_dm.data, rheology.dbe_dm.data,
                rheology.dus_deps.data, rheology.dus_dU.data, rheology.dus_dbeta.data,
                rheology.dus_duc.data, rheology.dus_dm.data,
                rheology.deta_dH.data, rheology.dbe_dH.data, rheology.dus_dH.data,
                state.u.data, state.v.data, state.H.data, state.xi.data,
                rheology.B.data, sliding.beta.data, sliding.u_c.data, state.u_b.data,
                sliding.m.value, sliding.u_reg.value, sliding.water_drag.value, sliding.sliding_law.value,
                rheology.n.value, rheology.eps_reg.value, rheology.eps_reg_shear.value, grid.dx,
                int(rheology.n_sigma.value), *self._quadrature(),
                grid.ny, grid.nx, stride, halo))

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

        kernel = self.kernels.get_function('compute_residual')
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
            self.compute_xi()

        if operator_only:
            out_u = self.F_u
            out_v = self.F_v
            out_ud = self.F_ud
            out_vd = self.F_vd
            out_H = self.F_H
            use_forcing = False
        else:
            out_u = self.r_u
            out_v = self.r_v
            out_ud = self.r_ud
            out_vd = self.r_vd
            out_H = self.r_H
            use_forcing = True

        if grid.diva:
            self.compute_diva_coeffs()      # refresh eta_bar/beta_eff consistent with the state

        args = (out_u, out_v, out_ud, out_vd, out_H,
                state.u.data, state.v.data, state.ud.data, state.vd.data, state.H.data,
                state.phi.data, state.xi.data, state.mask.data,
                self.f_u, self.f_v, self.f_ud, self.f_vd, self.f_H,
                geometry.bed.data,
                rheology.B.data,
                sliding.beta.data,
                self.gamma,
                use_forcing,use_mask,bool(grid.ssa),
                rheology.n.value, rheology.eps_reg.value, rheology.H_reg.value,
                geometry.sigmoid_c.value,
                sliding.m.value, sliding.u_reg.value,
                sliding.water_drag.value, sliding.flotation_reg_sliding.value,
                calving_rate, calving.flotation_reg_calving.value,
                grid.dx, dt,
                grid.ny, grid.nx, stride, halo)
        if grid.diva:
            args = args + (rheology.eta_bar.data, sliding.beta_eff.data)
        kernel(grid_size, block_size, args)

        if return_norms:
            return cp.linalg.norm(out_u),cp.linalg.norm(out_v),cp.linalg.norm(out_ud),cp.linalg.norm(out_vd),cp.linalg.norm(out_H)

    def compute_jvp(self, dt, 
            use_mask=True, 
            freeze_calving=False, 
            freeze_phi=False,
            return_norms=False):

        kernel = self.kernels.get_function('compute_jvp')
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
            self.compute_xi()

        if grid.diva:
            self.compute_diva_coeffs()      # refresh u_b warm-start / current state

        args = (self.jvp_u, self.jvp_v, self.jvp_ud, self.jvp_vd, self.jvp_H,
                state.u.data, state.v.data, state.ud.data, state.vd.data, state.H.data,
                self.var_u, self.var_v, self.var_ud, self.var_vd, self.var_H,
                state.phi.data, state.xi.data, state.mask.data,
                self.f_u, self.f_v, self.f_ud, self.f_vd, self.f_H,
                geometry.bed.data,
                rheology.B.data,
                sliding.beta.data,
                self.gamma,
                use_mask, bool(grid.ssa),
                rheology.n.value, rheology.eps_reg.value, rheology.H_reg.value,
                geometry.sigmoid_c.value,
                sliding.m.value, sliding.u_reg.value,
                sliding.water_drag.value, sliding.flotation_reg_sliding.value,
                calving_rate, calving.flotation_reg_calving.value,
                grid.dx, dt,
                grid.ny, grid.nx, stride, halo)
        if grid.diva:
            args = args + (sliding.u_c.data, state.u_b.data,
                           sliding.sliding_law.value, rheology.eps_reg_shear.value,
                           int(rheology.n_sigma.value), *self._quadrature())
        kernel(grid_size, block_size, args)

        if return_norms:
            return (cp.linalg.norm(self.jvp_u), cp.linalg.norm(self.jvp_v),
                    cp.linalg.norm(self.jvp_ud), cp.linalg.norm(self.jvp_vd),
                    cp.linalg.norm(self.jvp_H))

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

    def compute_xi(self, relaxation=cp.float32(0.0)):
        kernel = self.kernels.get_function('compute_flotation_fraction')
        grid_size, block_size, stride, halo = self._kernel_config
            
        grid = self.grid
        kernel(grid_size, block_size,
                   (grid.state.xi.data,
                    grid.state.H.data, grid.geometry.depth.data, 
                    grid.geometry.sigmoid_c.value,
                    grid.geometry.sigmoid_k.value,
                    relaxation,
                    grid.ny, grid.nx, 
                    stride, halo))

    def vanka_smooth(self, dt,
            freeze_calving=False,
            freeze_phi=False):

        kernel = self.kernels.get_function('vanka_smooth')
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
            self.compute_phi(relaxation=self.vanka_config.relax_phi)
            self.compute_xi(relaxation=self.vanka_config.relax_phi)

        self.delta_u.fill(0.0)
        self.delta_v.fill(0.0)
        self.delta_H.fill(0.0)
        if not grid.ssa:
            # the SSA kernel build never scatters deformational deltas
            self.delta_ud.fill(0.0)
            self.delta_vd.fill(0.0)
        if grid.diva:
            self.compute_diva_coeffs()      # frozen coefficients consistent with the current state

        args = (self.delta_u, self.delta_v, self.delta_ud, self.delta_vd, self.delta_H,
                state.mask.data,
                state.u.data, state.v.data, state.ud.data, state.vd.data, state.H.data,
                state.phi.data, state.xi.data,
                self.f_u, self.f_v, self.f_ud, self.f_vd, self.f_H,
                geometry.bed.data, rheology.B.data, sliding.beta.data,
                self.gamma,
                rheology.n.value, rheology.eps_reg.value, rheology.H_reg.value,
                geometry.sigmoid_c.value,
                sliding.m.value, sliding.u_reg.value,
                sliding.water_drag.value,
                sliding.flotation_reg_sliding.value,
                calving_rate,
                calving.flotation_reg_calving.value,
                grid.dx, dt,
                grid.ny, grid.nx, stride, halo,
                cp.int32(self.vanka_config.newton_config.steps),
                cp.float32(self.vanka_config.newton_config.relaxation),
                cp.float32(self.vanka_config.newton_config.step_tolerance),
                cp.float32(self.vanka_config.newton_config.momentum_damping),
                cp.float32(self.vanka_config.newton_config.mc_damping),
                bool(grid.ssa))
        if grid.diva:
            args = args + (rheology.eta_bar.data, sliding.beta_eff.data)
        kernel(grid_size, block_size, args)

    def vanka_sweep(self, dt, n_iter, 
            freeze_calving=False,
            freeze_phi=False):
        for i in range(n_iter):
            self.vanka_smooth(dt,freeze_calving=freeze_calving,freeze_phi=freeze_phi)
            self.grid.state.u.data[:] += self.vanka_config.omega * self.delta_u
            self.grid.state.v.data[:] += self.vanka_config.omega * self.delta_v
            if not self.grid.ssa:
                self.grid.state.ud.data[:] += self.vanka_config.omega * self.delta_ud
                self.grid.state.vd.data[:] += self.vanka_config.omega * self.delta_vd
            self.grid.state.H.data[:] += self.vanka_config.omega * self.delta_H
            self.vanka_config.hook_func(i)

    def vanka_dump(self,dt):
        kernel = self.kernels.get_function('vanka_dump')
        grid_size, block_size, stride, halo = self._kernel_config
        grid = self.grid

        J = cp.zeros((self.grid.ny*self.grid.nx,81),dtype=cp.float32)
        r = cp.zeros((self.grid.ny*self.grid.nx,9),dtype=cp.float32)
        kernel(grid_size, block_size,
               (J,r,
                grid.state.u.data, grid.state.v.data, grid.state.ud.data, grid.state.vd.data, grid.state.H.data, grid.state.phi.data, grid.state.xi.data,
                self.f_u, self.f_v, self.f_ud, self.f_vd, self.f_H,
                grid.geometry.bed.data, grid.rheology.B.data, grid.sliding.beta.data, self.gamma,
                grid.rheology.n.value, grid.rheology.eps_reg.value, grid.rheology.H_reg.value, grid.geometry.sigmoid_c.value,
                grid.sliding.m.value, grid.sliding.u_reg.value, grid.sliding.water_drag.value, grid.sliding.flotation_reg_sliding.value,
                grid.calving.calving_rate.value, grid.calving.flotation_reg_calving.value,
                grid.dx, dt,
                grid.ny, grid.nx, stride, halo,
                bool(grid.ssa))
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
    def var_ud(self):
        if self._var_ud is None:
            self._var_ud = cp.zeros((self.grid.ny,self.grid.nx+1),dtype=cp.float32)
        return self._var_ud
    
    @property
    def var_vd(self):
        if self._var_vd is None:
            self._var_vd = cp.zeros((self.grid.ny+1,self.grid.nx),dtype=cp.float32)
        return self._var_vd

    
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
    def jvp_ud(self):
        if self._jvp_ud is None:
            self._jvp_ud = cp.zeros((self.grid.ny,self.grid.nx+1),dtype=cp.float32)
        return self._jvp_ud
    
    @property
    def jvp_vd(self):
        if self._jvp_vd is None:
            self._jvp_vd = cp.zeros((self.grid.ny+1,self.grid.nx),dtype=cp.float32)
        return self._jvp_vd
    
    @property
    def jvp_H(self):
        if self._jvp_H is None:
            self._jvp_H = cp.zeros((self.grid.ny,self.grid.nx),dtype=cp.float32)
        return self._jvp_H

@dataclass
class NewtonConfig:
    steps: int = 30
    relaxation: cp.float32 = cp.float32(0.5)
    # Dimensionless patch Newton exit: iterate until the remaining
    # correction falls below this fraction of the patch state (measured in
    # the equilibrated metric). Converged patches exit on their first
    # iteration; loosening speeds sweeps at the cost of smoothing depth
    # (1e-5 works but converges noticeably less smoothly than 1e-6).
    step_tolerance: cp.float32 = cp.float32(1e-6)
    # Both dampings are ABSOLUTE diagonal shifts acting as pseudo-transient
    # continuation: momentum_damping on the velocity rows (in the model's head
    # units), mc_damping on the transport row (units of 1/time; an
    # effective pseudo-timestep of 1/mc_damping that dominates when dt is
    # large). A relative/multiplicative velocity damping was tried and
    # abandoned: the patches that need protection are those whose diagonals
    # are small relative to their off-diagonal coupling, and damping
    # proportional to the diagonal gives exactly those patches the least
    # absolute protection (destabilizes Greenland-scale problems).
    # The stable operating point couples momentum_damping to the Schwarz
    # relaxation omega: at omega = 0.5, momentum_damping < 1 risks patch
    # Newton divergence (NaNs) at hard-to-predict moments in difficult
    # configurations (thin ice, low water drag, large dt). Halving omega to
    # 0.25 admits a 10x smaller damping, which converges faster overall and
    # removes the slow-shelf transient (over-damped floating ice taking
    # several timesteps to spin up).
    momentum_damping: cp.float32 = cp.float32(0.1)
    mc_damping: cp.float32 = cp.float32(1.0)
    # Optional separate damping for the deformational (ud/vd) rows of the
    # ADJOINT patch: the shear block's H^(n+1)-scaled local response is what
    # stiffens at steep margins, so it can carry stronger control without
    # slowing the depth-averaged modes. None (default): follow
    # momentum_damping (including the divergence-recovery escalation).
    # Currently consumed by the adjoint smoother only.
    shear_damping: cp.float32 = None

@dataclass
class VankaConfig:
    omega: cp.float32 = cp.float32(0.25)
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
        cuda_files = ['common.cu', 'viscosity.cu', 'stress.cu', 'diva.cu',
                          'flux.cu', 'residuals.cu', 'vanka.cu', 'grad.cu']
        cuda_source = '\n'.join((cuda_dir / f).read_text() for f in cuda_files)
        
        if use_fast_math:
            options=("--use_fast_math",)
        else:
            options=()
        # Compile-time stress scheme specialization: the SSA build removes
        # all deformational physics and solves 5x5 patches (see common.cu)
        options += ("-DGLIDE_MOLHO=%d" % (0 if grid.ssa else 1),)
        # DIVA rides the SSA-shaped build (GLIDE_MOLHO=0) and toggles the
        # closure-coefficient reads; only ever 1 when grid.ssa is also True.
        options += ("-DGLIDE_DIVA=%d" % (1 if grid.diva else 0),)

        self.kernels = cp.RawModule(code=cuda_source, options=options)

        self.r_u = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.r_v = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.r_ud = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.r_vd = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.r_H = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)
        
        self.f_u = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.f_v = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.f_ud = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.f_vd = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.f_H = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)
        
        self.vjp_u = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.vjp_v = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.vjp_ud = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.vjp_vd = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.vjp_H = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)
        
        self.delta_lambda_u = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.delta_lambda_v = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.delta_lambda_ud = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.delta_lambda_vd = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.delta_lambda_H = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)

        # Row-projected multipliers for the vjp launch (see the constraint
        # convention in cuda/common.cu)
        self.lam_free_u = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.lam_free_v = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.lam_free_ud = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.lam_free_vd = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.lam_free_H = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)

        self.gamma = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)
        self.gamma.fill(grid.geometry.thklim.value)

        # DIVA exact adjoint: per-cell coefficient adjoints (eta_bar, beta_eff) accumulated by
        # the transpose kernel, then pushed back to velocity/thickness by compute_diva_vjp_coeffs.
        # Because stock DIVA's velocity block is self-adjoint, this exact 2-pass transpose is the
        # forward transpose (no growing-mode smoother guard needed).
        self.diva_exact_coeff_adjoint = True
        self.W_eta = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)
        self.W_be  = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)

        # The adjoint solve is LINEAR: it does not need the heavy damping
        # that protects the forward patch Newton iteration from divergence,
        # and momentum_damping ~ 1 on the depth-averaged rows severely
        # degrades its convergence rate (0.01 is field-tested on Greenland
        # and Antarctica). The deformational (shear) rows are the opposite
        # case: their H^(n+1)-scaled local response destabilizes the
        # transposed smoother at steep margins at light damping, while
        # strong damping costs nothing measurable on healthy problems
        # (identical 8-cycle reduction on the Greenland benchmark for
        # shear_damping anywhere in [0.01, 3]) - so they default to the
        # same strong PTC as the transport row.
        self.vanka_config = VankaConfig(
                newton_config=NewtonConfig(momentum_damping=cp.float32(0.01),
                                           shear_damping=cp.float32(1.0)))

    @property
    def _kernel_config(self):
        block_size = (16, 16)
        stride = 14
        halo = 1
        grid_size = (self.grid.nx // stride + 1, self.grid.ny // stride + 1)
        return grid_size, block_size, stride, halo

    def _launch_vjp(self, out_u, out_v, out_ud, out_vd, out_H, dt,
            use_mask=True,
            use_forcing=False,
            freeze_calving=False):
        """Apply the exact transpose J^T lambda (minus f when use_forcing).

        The kernel computes the pure physics transpose. The constrained-row
        structure of the Jacobian - identity rows at Dirichlet facets and on
        the active set (see the constraint convention in cuda/common.cu) - is
        applied here, once: multipliers are projected off constrained rows
        before launch (their rows have no off-diagonal entries), and the
        identity parts lambda_c are added to the outputs afterwards.
        Constrained COLUMNS are genuine and handled by the kernel.
        """
        kernel = self.kernels.get_function('compute_vjp')
        grid_size, block_size, stride, halo = self._kernel_config
        self._last_dt = dt

        grid = self.grid
        state = grid.state
        adjoint = grid.adjoint
        geometry = grid.geometry
        rheology = grid.rheology
        sliding = grid.sliding
        calving = grid.calving

        if freeze_calving:
            calving_rate = cp.float32(0.0)
        else:
            calving_rate = calving.calving_rate.value

        # Row projection
        self.lam_free_u[:,:] = adjoint.lambda_u.data
        self.lam_free_u[:,0] = 0.0
        self.lam_free_u[:,-1] = 0.0
        self.lam_free_v[:,:] = adjoint.lambda_v.data
        self.lam_free_v[0,:] = 0.0
        self.lam_free_v[-1,:] = 0.0
        if grid.ssa:
            # SSA mode: every ud/vd row is an identity row, so all of their
            # multipliers are projected off the physics transpose
            self.lam_free_ud.fill(0.0)
            self.lam_free_vd.fill(0.0)
        else:
            self.lam_free_ud[:,:] = adjoint.lambda_ud.data
            self.lam_free_ud[:,0] = 0.0
            self.lam_free_ud[:,-1] = 0.0
            self.lam_free_vd[:,:] = adjoint.lambda_vd.data
            self.lam_free_vd[0,:] = 0.0
            self.lam_free_vd[-1,:] = 0.0
        if use_mask:
            self.lam_free_H[:,:] = (1.0 - state.mask.data)*adjoint.lambda_H.data
        else:
            self.lam_free_H[:,:] = adjoint.lambda_H.data

        out_u.fill(0)
        out_v.fill(0)
        out_ud.fill(0)
        out_vd.fill(0)
        out_H.fill(0)
        if grid.diva:
            grid.forward_operators.compute_diva_coeffs()   # frozen coeffs at the fixed forward state
            self.W_eta.fill(0.0)
            self.W_be.fill(0.0)
        args = (out_u, out_v, out_ud, out_vd, out_H,
                state.u.data, state.v.data, state.ud.data, state.vd.data, state.H.data,
                self.lam_free_u, self.lam_free_v,
                self.lam_free_ud, self.lam_free_vd, self.lam_free_H,
                state.phi.data, state.xi.data, state.mask.data,
                self.f_u, self.f_v, self.f_ud, self.f_vd, self.f_H,
                geometry.bed.data,
                rheology.B.data,
                sliding.beta.data,
                self.gamma,
                use_forcing, use_mask,
                rheology.n.value, rheology.eps_reg.value, rheology.H_reg.value,
                geometry.sigmoid_c.value,
                sliding.m.value, sliding.u_reg.value,
                sliding.water_drag.value, sliding.flotation_reg_sliding.value,
                calving_rate, calving.flotation_reg_calving.value,
                grid.dx, dt,
                grid.ny, grid.nx, stride, halo)
        if grid.diva:
            args = args + (rheology.eta_bar.data, sliding.beta_eff.data,
                           self.W_eta, self.W_be, cp.int32(0))   # freeze_coeffs=0: stock exact transpose
        kernel(grid_size, block_size, args)

        # Identity rows: (J^T lambda)_c = lambda_c + interior contributions
        # (the latter already accumulated by the kernel)
        out_u[:,0] += adjoint.lambda_u.data[:,0]
        out_u[:,-1] += adjoint.lambda_u.data[:,-1]
        out_v[0,:] += adjoint.lambda_v.data[0,:]
        out_v[-1,:] += adjoint.lambda_v.data[-1,:]
        if grid.ssa:
            out_ud += adjoint.lambda_ud.data
            out_vd += adjoint.lambda_vd.data
        else:
            out_ud[:,0] += adjoint.lambda_ud.data[:,0]
            out_ud[:,-1] += adjoint.lambda_ud.data[:,-1]
            out_vd[0,:] += adjoint.lambda_vd.data[0,:]
            out_vd[-1,:] += adjoint.lambda_vd.data[-1,:]
        if use_mask:
            out_H += state.mask.data*adjoint.lambda_H.data

        if grid.diva and self.diva_exact_coeff_adjoint:
            self._apply_diva_coeff_adjoints(out_u, out_v, out_H)

    def _apply_diva_coeff_adjoints(self, out_u, out_v, out_H):
        """Second pass of the exact DIVA transpose: push the per-cell coefficient adjoints
        W_eta/W_be back to the velocity facets and thickness through the closure derivatives
        (compute_diva_derivs then compute_diva_vjp_coeffs)."""
        grid = self.grid
        rheology = grid.rheology
        grid.forward_operators.compute_diva_derivs()
        kernel = self.kernels.get_function('compute_diva_vjp_coeffs')
        grid_size, block_size, stride, halo = self._kernel_config
        kernel(grid_size, block_size,
               (out_u, out_v, out_H,
                grid.state.u.data, grid.state.v.data,
                self.W_eta, self.W_be,
                rheology.deta_deps.data, rheology.deta_dU.data,
                rheology.dbe_deps.data, rheology.dbe_dU.data,
                rheology.deta_dH.data, rheology.dbe_dH.data,
                grid.dx, grid.ny, grid.nx, stride, halo))

    def compute_residual(self, dt,
            use_mask=True,
            freeze_calving=False,
            return_norms=False):

        self._launch_vjp(self.r_u, self.r_v, self.r_ud, self.r_vd, self.r_H,
                dt, use_mask=use_mask, use_forcing=True,
                freeze_calving=freeze_calving)

        if return_norms:
            return (cp.linalg.norm(self.r_u),cp.linalg.norm(self.r_v),
                    cp.linalg.norm(self.r_ud),cp.linalg.norm(self.r_vd),
                    cp.linalg.norm(self.r_H))

    def compute_vjp(self, dt,
            use_mask=True,
            freeze_calving=False):

        self._launch_vjp(self.vjp_u, self.vjp_v, self.vjp_ud, self.vjp_vd,
                self.vjp_H, dt, use_mask=use_mask, use_forcing=False,
                freeze_calving=freeze_calving)


    def vanka_smooth(self, dt,
            freeze_calving=False):

        kernel = self.kernels.get_function('vanka_smooth_adjoint')
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
        self.delta_lambda_ud.fill(0.0)
        self.delta_lambda_vd.fill(0.0)
        self.delta_lambda_H.fill(0.0)
        # DIVA: eta_bar/beta_eff are the forward-state closure coefficients (already current for a
        # fixed adjoint state); the smoother reads them as frozen, exactly like the forward.
        args = (self.delta_lambda_u, self.delta_lambda_v,
                self.delta_lambda_ud, self.delta_lambda_vd, self.delta_lambda_H,
                state.u.data, state.v.data, state.ud.data, state.vd.data, state.H.data,
                state.phi.data, state.xi.data, state.mask.data,
                self.r_u, self.r_v, self.r_ud, self.r_vd, self.r_H,
                geometry.bed.data, rheology.B.data, sliding.beta.data,
                self.gamma,
                rheology.n.value, rheology.eps_reg.value, rheology.H_reg.value,
                geometry.sigmoid_c.value,
                sliding.m.value, sliding.u_reg.value,
                sliding.water_drag.value,
                sliding.flotation_reg_sliding.value,
                calving_rate,
                calving.flotation_reg_calving.value,
                grid.dx, dt,
                grid.ny, grid.nx, stride, halo,
                cp.float32(self.vanka_config.newton_config.momentum_damping),
                cp.float32(self.vanka_config.newton_config.momentum_damping
                           if self.vanka_config.newton_config.shear_damping is None
                           else self.vanka_config.newton_config.shear_damping),
                cp.float32(self.vanka_config.newton_config.mc_damping),
                bool(grid.ssa))
        if grid.diva:
            args = args + (rheology.eta_bar.data, sliding.beta_eff.data)
        kernel(grid_size, block_size, args)

    def vanka_sweep(self, dt, n_iter,
            freeze_calving=False):
        for i in range(n_iter):
            self.compute_residual(dt,freeze_calving=freeze_calving)
            self.vanka_smooth(dt,freeze_calving=freeze_calving)
            self.grid.adjoint.lambda_u.data[:] -= self.vanka_config.omega * self.delta_lambda_u
            self.grid.adjoint.lambda_v.data[:] -= self.vanka_config.omega * self.delta_lambda_v
            self.grid.adjoint.lambda_ud.data[:] -= self.vanka_config.omega * self.delta_lambda_ud
            self.grid.adjoint.lambda_vd.data[:] -= self.vanka_config.omega * self.delta_lambda_vd
            self.grid.adjoint.lambda_H.data[:] -= self.vanka_config.omega * self.delta_lambda_H

    def _diva_param_gradient(self, out, deta_dp, dbe_dp, reduce=False):
        """DIVA parameter gradient dJ/dp = sum_cells (W_eta*deta_dp + W_be*dbe_dp), with W_eta/W_be
        the coefficient adjoints at the converged multiplier lambda.  A VJP pass re-fills them, and
        compute_diva_derivs supplies the closure's parameter sensitivities deta_dp/dbe_dp."""
        grid = self.grid
        self.compute_vjp(self._last_dt)                 # refill W_eta/W_be at the converged lambda
        grid.forward_operators.compute_diva_derivs()
        name = 'compute_gradient_param_sum_diva' if reduce else 'compute_gradient_param_diva'
        kernel = self.kernels.get_function(name)
        grid_size, block_size, stride, halo = self._kernel_config
        out.fill(0)
        kernel(grid_size, block_size,
               (out, self.W_eta, self.W_be, deta_dp.data, dbe_dp.data,
                grid.ny, grid.nx, stride, halo))

    def diva_surface_misfit_rhs(self, cot, out_u=None, out_v=None, out_H=None):
        """Adjoint right-hand side for an objective built on DIVA SURFACE speed.

        Given the per-cell cotangent ``cot = dJ/d(u_s)``, fills ``(f_u, f_v, f_H) = -dJ/d(u,v,H)``:
        the surface cotangent scattered to the velocity facets -- and to thickness, since u_s depends
        on H through the shear moments -- via the closure's stored dus_deps/dus_dU/dus_dH.

        Observations are of SURFACE velocity, and under DIVA that differs from the depth average by
        the vertical shear (the whole point of the scheme), so an inversion against surface data
        should route the misfit through here rather than through dJdu/dJdv.  Under SSA there is no
        shear (u_s == |ubar|), so this is a no-op and returns (None, None); build the RHS from the
        depth-averaged misfit directly."""
        if not self.grid.diva:
            return None, None
        grid = self.grid
        if out_u is None:
            out_u, out_v = self.f_u, self.f_v
            if out_H is None:
                out_H = self.f_H
        out_u.fill(0); out_v.fill(0)
        if out_H is not None:
            out_H.fill(0)
        # dus_* must match the current state; they are written by the forward derivative kernel.
        grid.forward_operators.compute_diva_derivs()
        kernel = self.kernels.get_function('compute_diva_us_vjp')
        grid_size, block_size, stride, halo = self._kernel_config
        kernel(grid_size, block_size,
               (out_u, out_v, out_H,
                grid.state.u.data, grid.state.v.data,
                cot, grid.rheology.dus_deps.data, grid.rheology.dus_dU.data,
                grid.rheology.dus_dH.data,
                grid.dx, grid.ny, grid.nx, stride, halo))
        out_u *= -1.0; out_v *= -1.0            # f = -dJ/dx
        if out_H is not None:
            out_H *= -1.0
        return out_u, out_v

    def diva_surface_param_gradient(self, cot, param):
        """The EXPLICIT dJ/d(param) term a surface objective adds on top of lambda^T dr/dp.

        u_s depends on beta, u_c and m DIRECTLY through the closure, so the usual adjoint gradient
        (which assumes the objective sees the parameters only through the state) is incomplete:

            dJ/dp = sum_c cot_c * d(u_s)_c/dp  +  lambda^T dr/dp

        This returns the first, cell-local term.  Field-shaped for beta/u_c, scalar for the global m;
        None under SSA (u_s == ubar, no explicit dependence)."""
        if not self.grid.diva:
            return None
        rheology = self.grid.rheology
        field = {'beta': rheology.dus_dbeta, 'u_c': rheology.dus_duc, 'm': rheology.dus_dm}[param]
        term = cot * field.data
        return float(cp.sum(term)) if param == 'm' else term

    def compute_gradient_beta(self):
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

        if grid.diva:
            # DIVA: dR/dbeta flows through the closure (beta -> beta_eff, eta_bar), so the gradient
            # is the coefficient-adjoint contraction, not the direct SSA sliding-law transpose.
            self._diva_param_gradient(sliding.beta.grad, rheology.deta_dbeta, rheology.dbe_dbeta)
            return

        if grid.ssa:
            # SSA mode: all ud/vd rows are identity rows with no beta
            # dependence; project their multipliers out of the drag
            # transpose by passing zeroed arrays
            self.lam_free_ud.fill(0.0)
            self.lam_free_vd.fill(0.0)
            lambda_ud = self.lam_free_ud
            lambda_vd = self.lam_free_vd
        else:
            lambda_ud = adjoint.lambda_ud.data
            lambda_vd = adjoint.lambda_vd.data

        sliding.beta.grad.fill(0)
        kernel(grid_size, block_size,
               (sliding.beta.grad,
                state.u.data, state.v.data, state.ud.data, state.vd.data, state.H.data,
                adjoint.lambda_u.data, adjoint.lambda_v.data,
                lambda_ud, lambda_vd, adjoint.lambda_H.data,
                state.phi.data, state.xi.data, state.mask.data,
                geometry.bed.data, 
                rheology.B.data, 
                sliding.beta.data,
                self.gamma,
                rheology.n.value, rheology.eps_reg.value, 
                geometry.sigmoid_c.value,
                sliding.m.value, sliding.u_reg.value, 
                sliding.water_drag.value, sliding.flotation_reg_sliding.value,
                calving.calving_rate.value, calving.flotation_reg_calving.value,
                grid.dx, cp.float32(0.0),
                grid.ny, grid.nx, stride, halo))

    def compute_gradient_u_c(self):
        """Gradient w.r.t. the per-cell regularized-Coulomb threshold u_c.

        DIVA only: u_c enters solely through the cell closure (beta_eff/eta_bar), so the whole
        gradient is the coefficient-adjoint contraction dJ/du_c = W_eta*deta_duc + W_be*dbe_duc.
        SSA/MOLHO are Weertman-only in this tree (no u_c field), so there is nothing to
        differentiate; invert beta instead, or make u_c a shared field first."""
        grid = self.grid
        if not grid.diva:
            raise NotImplementedError(
                "compute_gradient_u_c requires DIVA: the Coulomb threshold u_c exists only under "
                "stress_scheme='diva' (SSA/MOLHO are Weertman-only here).")
        self._diva_param_gradient(grid.sliding.u_c.grad,
                                  grid.rheology.deta_duc, grid.rheology.dbe_duc)

    def compute_gradient_m(self):
        """Gradient w.r.t. the global Weertman exponent m (a single scalar), stored on
        sliding.m.grad.

        DIVA only: under DIVA m enters solely through the closure, so the gradient is the
        coefficient-adjoint contraction reduced to a scalar.  The SSA/MOLHO Weertman-exponent
        gradient kernel is not ported (m is rarely inverted; add a grad.cu kernel if needed)."""
        grid = self.grid
        if not grid.diva:
            raise NotImplementedError(
                "compute_gradient_m is wired for DIVA only; the SSA/MOLHO Weertman-exponent "
                "gradient kernel is not ported.")
        grad_m = cp.zeros(1, dtype=cp.float32)
        self._diva_param_gradient(grad_m, grid.rheology.deta_dm, grid.rheology.dbe_dm, reduce=True)
        grid.sliding.m.grad = float(grad_m[0])

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
                state.phi.data, state.xi.data, state.mask.data,
                geometry.bed.data, 
                rheology.B.data, 
                sliding.beta.data,
                self.gamma,
                rheology.n.value, rheology.eps_reg.value, 
                geometry.sigmoid_c.value,
                sliding.m.value, sliding.u_reg.value, 
                sliding.water_drag.value, sliding.flotation_reg_sliding.value,
                calving.calving_rate.value, calving.flotation_reg_calving.value,
                grid.dx, cp.float32(0.0),
                grid.ny, grid.nx, stride, halo)) 


    def compute_gradient_H_prev(self, dt):
        # Active-set rows are identity rows with no H_prev dependence; project
        # out their multipliers (see the constraint convention in cuda/common.cu)
        free = 1.0 - self.grid.state.mask.data
        self.grid.state.H_prev.grad[:,:] = -free*self.grid.adjoint.lambda_H.data[:,:]/dt

    def compute_gradient_smb(self):
        free = 1.0 - self.grid.state.mask.data
        self.grid.forcing.smb.grad[:,:] = -free*self.grid.adjoint.lambda_H.data[:,:]
  
    



