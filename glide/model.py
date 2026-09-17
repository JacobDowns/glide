"""
Core ice physics API.

Provides the IcePhysics class that wraps the forward model and adjoint
computations into a clean interface.
"""

import cupy as cp
from .grid import Grid
from .multigrid import Multigrid, FASCDSolver,FASAdjointSolver
from .enthalpy import EnthalpyOperators, T_MELT

class IceDynamics:
    def __init__(self,mg=None,
            n_levels=None,grid=None,
            ny=None,nx=None,dx=None,
            x0=cp.float32(0.0),y0=cp.float32(0.0),crs=None,
            stress_scheme='molho'):
        # stress_scheme: 'molho' (two-field, shear-resolving) or 'ssa'
        # (deformational components pinned to zero). When an existing mg or
        # grid is supplied, its stress_scheme governs.
        if mg is not None:
            self.mg = mg
        elif grid is not None and n_levels is not None:
            self.mg = Multigrid(n_levels,finest_grid=grid)
        elif ny and nx and dx and n_levels:
            self.mg = Multigrid(n_levels,ny=ny,nx=nx,dx=dx,
                   x0=x0,y0=y0,crs=crs,stress_scheme=stress_scheme)
        else:
            raise ValueError('Must supply either (a) a multigrid object \
                              (b) a grid and number of levels \
                              (c) ny/nx/dx and number of levels')

        self._forward_solver = None
        self._adjoint_solver = None
        self.top_level = 0

        self._post_forward_hooks = []

    @property
    def forward_solver(self):
        if self._forward_solver is None:
            self._forward_solver = FASCDSolver(self.mg)
        return self._forward_solver
    
    @property
    def adjoint_solver(self):
        if self._adjoint_solver is None:
            self._adjoint_solver = FASAdjointSolver(self.mg)
        return self._adjoint_solver

    def set_top_level(self,level):
        self.top_level = level

    def register_post_forward_hook(self,hook):
        self._post_forward_hooks.append(hook)

    def forward(self,t,dt,update_geometry=True):
        self.forward_solver.solve(dt,start_level=self.top_level)
        if update_geometry:
            self.mg.levels[self.top_level].state.H_prev.data[:,:] = (
                self.mg.levels[self.top_level].state.H.data[:,:]
            )
        for f in self._post_forward_hooks:
            f(t+dt)

    def backward(self,t,dt,dJdu=None,dJdv=None,dJdud=None,dJdvd=None,dJdH=None,dJdu_s=None,
            compute_beta_grad=True,compute_bed_grad=True,
            compute_H_prev_grad=True,compute_smb_grad=True,
            compute_uc_grad=False,compute_m_grad=False):
        """Solve the adjoint and reduce it onto the parameter gradients.

        ``dJdu``/``dJdv``/``dJdud``/``dJdvd``/``dJdH`` are cotangents on the solved state.

        ``dJdu_s`` is the cotangent on the DIVA SURFACE speed (per cell), and enters differently:
        u_s is not a state variable, so its cotangent is scattered to the velocity/thickness rows
        through the closure (diva_surface_misfit_rhs), and u_s depends on the sliding parameters
        DIRECTLY, so an explicit sum_c cot_c d(u_s)_c/dp term is added after the reduction.  DIVA
        only."""
        ao = self.mg.levels[self.top_level].adjoint_operators

        # Surface term first: diva_surface_misfit_rhs FILLS f_u/f_v/f_H with -dJ/d(u,v,H), so the
        # depth-averaged cotangents accumulate on top of it rather than the reverse.
        if dJdu_s is not None:
            if ao.diva_surface_misfit_rhs(dJdu_s)[0] is None:
                raise ValueError(
                    "dJdu_s requires DIVA (stress_scheme='diva'): under SSA/MOLHO the surface and "
                    "depth-averaged velocities coincide, so pass the misfit through dJdu/dJdv.")
        else:
            ao.f_u.fill(0.0); ao.f_v.fill(0.0); ao.f_H.fill(0.0)
        if dJdu is not None: ao.f_u -= dJdu
        if dJdv is not None: ao.f_v -= dJdv
        if dJdH is not None: ao.f_H -= dJdH

        # Deformational rows (MOLHO only); u_s has no deformational analogue.
        if dJdud is not None: ao.f_ud[:,:] = -dJdud
        else: ao.f_ud.fill(0.0)
        if dJdvd is not None: ao.f_vd[:,:] = -dJdvd
        else: ao.f_vd.fill(0.0)

        converged = self.adjoint_solver.solve(dt,start_level=self.top_level)
        ao.compute_gradient_beta()
        ao.compute_gradient_bed()
        ao.compute_gradient_H_prev(dt)
        ao.compute_gradient_smb()
        # Coulomb threshold u_c and Weertman exponent m are DIVA-only sliding-law parameters
        # (see AdjointOperators.compute_gradient_{u_c,m}); off by default.
        if compute_uc_grad: ao.compute_gradient_u_c()
        if compute_m_grad: ao.compute_gradient_m()

        # The explicit parameter dependence of u_s, which only a surface objective has.
        if dJdu_s is not None:
            sliding = self.mg.levels[self.top_level].sliding
            sliding.beta.grad[:] += ao.diva_surface_param_gradient(dJdu_s,'beta')
            if compute_uc_grad:
                sliding.u_c.grad[:] += ao.diva_surface_param_gradient(dJdu_s,'u_c')
            if compute_m_grad:
                sliding.m.grad = float(sliding.m.grad) + ao.diva_surface_param_gradient(dJdu_s,'m')

        return converged


class ThermalModel:
    """
    Enthalpy solver for coupled momentum/thermal simulations.

    Wraps EnthalpyOperators into a step(dt) interface that:
    1. Syncs 3D velocity from the SSA solution on the Grid
    2. Solves the enthalpy advection-diffusion equation
    3. Feeds the updated Arrhenius factor B back to Grid.rheology

    Parameters
    ----------
    grid : Grid
        The finest-level grid (shared with IceDynamics).
    nz : int
        Number of sigma levels.
    n_smooth : int
        Maximum number of column smoothing sweeps per time step.
    update_rheology : bool
        If True, update grid.rheology.B after each step from the Arrhenius factor.
    frictional_heating : bool
        If True, compute Q_fh = beta * |u_b|^(m+1) each step.
    rho_i : float
        Ice density for the B unit conversion (default 917 kg/m^3).
    """

    SEC_PER_YR = 365.25 * 86400.0

    def __init__(self, grid, nz=21, n_smooth=10,
                 update_rheology=True, frictional_heating=True,
                 rho_i=917.0):
        self.ops = EnthalpyOperators(grid, nz=nz)
        self.n_smooth = n_smooth
        self.update_rheology = update_rheology
        self.frictional_heating = frictional_heating
        self.rho_i = rho_i
        self.g = 9.81

    def initialize(self, T_surface, T_field=None, Q_geo=None):
        """
        Set initial conditions and boundary data.

        Parameters
        ----------
        T_surface : array-like, shape (ny, nx) or scalar
            Surface temperature in Kelvin (Dirichlet BC).
        T_field : array-like, shape (ny, nx, nz) or scalar, optional
            Initial 3D temperature. Defaults to T_surface everywhere.
        Q_geo : array-like, shape (ny, nx) or scalar, optional
            Geothermal heat flux in W/m^2. Defaults to 0.
        """
        if T_field is None:
            T_field = T_surface
        self.ops.initialize_from_temperature(T_field)
        self.ops.set_surface_enthalpy_from_temperature(
            cp.asarray(T_surface, dtype=cp.float32))
        if Q_geo is not None:
            self.ops.enthalpy_forcing.Q_geo[:] = cp.asarray(
                Q_geo, dtype=cp.float32)

    def pre_momentum(self):
        """Snapshot E and H before the momentum step.

        Must be called BEFORE model.forward() so that the conservative
        time derivative rho_i*(H*E - H_prev*E_prev)/dt uses the
        correct pre-step thickness.
        """
        ops = self.ops
        ops.enthalpy_state.E_prev[:] = ops.enthalpy_state.E
        ops.H_prev[:] = ops.grid.state.H.data

    def step(self, dt):
        """
        Advance enthalpy by one time step.

        Call pre_momentum() before the momentum step, then step()
        after. This ensures H_prev captures the pre-momentum thickness.

        Parameters
        ----------
        dt : float
            Time step in seconds.
        """
        ops = self.ops

        # Sync velocity from the momentum solution.
        # SSA velocities are in m/yr; enthalpy needs m/s.
        ops.broadcast_velocity()

        # Compute omega BEFORE the m/yr → m/s velocity conversion.
        # The omega kernel evaluates div(Hu) in m/yr to match the
        # momentum solver's float32 arithmetic exactly — this avoids
        # catastrophic cancellation from scale-dependent rounding
        # differences between the m/yr and m/s evaluations.
        self._compute_omega(dt)

        sec_per_yr = cp.float32(self.SEC_PER_YR)
        ops.enthalpy_velocity.u3d /= sec_per_yr
        ops.enthalpy_velocity.v3d /= sec_per_yr

        if self.frictional_heating:
            self._compute_frictional_heating()

        # Precompute forcing array (E_prev and H_prev set by pre_momentum)
        ops.set_rhs(snapshot=False)

        # Enthalpy solve
        ops.column_sweep(dt, self.n_smooth)

        # Feed back to momentum solver.
        # get_arrhenius_factor() returns B in SI (Pa s^{1/n}).
        # GLIDE's SSA works in year-based head units:
        #   B_glide = B_SI / (rho_i * g * sec_per_yr^{1/n})
        if self.update_rheology:
            n = float(ops.grid.rheology.n.value)
            scale = cp.float32(
                self.rho_i * self.g * self.SEC_PER_YR ** (1.0 / n))
            ops.grid.rheology.B.data[:] = ops.get_arrhenius_factor() / scale

    def _compute_frictional_heating(self):
        """Compute the basal frictional heat flux Q_fh = tau_b . u_b  (W/m^2).

        In GLIDE's head units the drag coefficient beta gives the basal drag
        as tau_b/(rho*g) = beta*|u|^m, with |u| in m/yr (the momentum solver's
        native units).  So the physical basal shear stress is rho*g*beta*|u_yr|^m
        [Pa] and the sliding speed is |u_yr|/SEC_PER_YR [m/s], giving

            Q_fh = tau_b * u_b = (rho*g / SEC_PER_YR) * beta * |u_yr|^(m+1).

        Two unit conversions matter and were both missing before:
          - u3d has already been divided to m/s at this point in step(), so we
            multiply back to m/yr; the Weertman regularization u_reg is in m/yr
            and MUST be added at the same scale (adding it to m/s speeds floors
            the interior speed at ~1 m/s and dumps ~beta W/m^2 over the whole
            bed -- a spurious heat source hundreds of times the geothermal flux).
          - the head->Pa factor rho*g (matching the momentum nondimensionalization,
            same rho*g used in B_scale).

        The effective basal drag the momentum solver uses is beta_eff = beta*xi,
        where xi is the grounding factor (~1 grounded, ~0 floating; stress.cu).
        We gate the heating by the SAME xi so floating/near-floating margins (which
        have no bed contact) get no spurious basal friction -- without it, thin fast
        marine cells at the calving front receive ~200 W/m^2 and their enthalpy runs
        away.
        """
        grid = self.ops.grid
        sliding = grid.sliding
        vel = self.ops.enthalpy_velocity
        spy = cp.float32(self.SEC_PER_YR)
        u_bed = vel.u3d[0, :, :] * spy  # m/s -> m/yr, shape (ny, nx+1)
        v_bed = vel.v3d[0, :, :] * spy  # m/s -> m/yr, shape (ny+1, nx)

        u_cell = 0.5 * (u_bed[:, 1:] + u_bed[:, :-1])
        v_cell = 0.5 * (v_bed[1:, :] + v_bed[:-1, :])
        speed_yr = cp.sqrt(u_cell**2 + v_cell**2 + sliding.u_reg.value**2)  # all m/yr

        m = sliding.m.value
        scale = cp.float32(self.rho_i * self.g / self.SEC_PER_YR)  # head->Pa, /yr->/s
        beta_eff = sliding.beta.data * grid.state.xi.data          # grounding-gated drag
        self.ops.enthalpy_forcing.Q_fh[:] = (
            scale * beta_eff * speed_yr ** (m + 1.0))

    def _compute_omega(self, dt):
        """Compute omega from the actual thickness change.

        Uses (H_new - H_prev) / dt as the column dH/dt, ensuring exact
        consistency between the conservative enthalpy equation and the
        realized mass balance from the momentum step.

        The computation is performed in m/yr units (the momentum solver's
        native scale) so that the LF mass flux evaluation matches the
        momentum solver's float32 arithmetic. The resulting omega is
        converted to m/s for the enthalpy equation.

        Parameters
        ----------
        dt : float
            Time step in seconds.
        """
        ops = self.ops
        sec_per_yr = cp.float32(self.SEC_PER_YR)
        # dH/dt in m/yr (velocity is still in m/yr at this point)
        dt_yr = cp.float32(dt / self.SEC_PER_YR)
        dh_dt_yr = (ops.grid.state.H.data - ops.H_prev) / dt_yr
        ops.compute_omega(dh_dt_yr)
        # Convert omega from m/yr to m/s
        ops.enthalpy_velocity.omega /= sec_per_yr

    @property
    def B_scale(self):
        """Conversion factor: B_glide = B_SI / B_scale."""
        n = float(self.ops.grid.rheology.n.value)
        return self.rho_i * self.g * self.SEC_PER_YR ** (1.0 / n)

    @property
    def temperature(self):
        """3D temperature field, shape (ny, nx, nz)."""
        return self.ops.get_temperature()

    @property
    def water_content(self):
        """3D water content field, shape (ny, nx, nz)."""
        return self.ops.get_water_content()

    @property
    def enthalpy(self):
        """3D enthalpy field, shape (ny, nx, nz)."""
        return self.ops.enthalpy_state.E

    @property
    def sigma(self):
        """Sigma level positions, shape (nz,)."""
        return self.ops.sigma
