"""
Core ice physics API.

Provides:
- IceDynamics: momentum + mass conservation (SSA multigrid solver)
- ThermalModel: enthalpy advection-diffusion (column solver)

Coupled usage:
    model = IceDynamics(...)
    thermal = ThermalModel(grid, ...)
    thermal.initialize(T_surface, T_field)

    while t < t_end:
        model.forward(t, dt)
        thermal.step(dt)
        t += dt
"""

import cupy as cp
from .grid import Grid
from .enthalpy import EnthalpyOperators, T_MELT
from .multigrid import Multigrid, FASCDSolver, FASAdjointSolver

class IceDynamics:
    def __init__(self,mg=None,
            n_levels=None,grid=None,
            ny=None,nx=None,dx=None,
            x0=cp.float32(0.0),y0=cp.float32(0.0),crs=None):
        if mg is not None:
            self.mg = mg
        elif grid is not None and n_levels is not None:
            self.mg = Multigrid(n_levels,finest_grid=grid)
        elif ny and nx and dx and n_levels:
            self.mg = Multigrid(n_levels,ny=ny,nx=nx,dx=dx,
                   x0=x0,y0=y0,crs=crs)
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

    def backward(self,t,dt,dJdu=None,dJdv=None,dJdH=None,
            compute_beta_grad=True,compute_bed_grad=True,
            compute_H_prev_grad=True,compute_smb_grad=True):
        if dJdu is not None:
            self.mg.levels[self.top_level].adjoint_operators.f_u[:,:] = -dJdu
        else:
            self.mg.levels[self.top_level].adjoint_operators.f_u.fill(0.0)            
        if dJdv is not None:
            self.mg.levels[self.top_level].adjoint_operators.f_v[:,:] = -dJdv
        else:
            self.mg.levels[self.top_level].adjoint_operators.f_v.fill(0.0)
        if dJdH is not None:
            self.mg.levels[self.top_level].adjoint_operators.f_H[:,:] = -dJdH
        else:    
            self.mg.levels[self.top_level].adjoint_operators.f_H.fill(0.0)

        self.adjoint_solver.solve(dt,start_level=self.top_level)
        self.mg.levels[self.top_level].adjoint_operators.compute_gradient_beta()
        self.mg.levels[self.top_level].adjoint_operators.compute_gradient_bed()
        self.mg.levels[self.top_level].adjoint_operators.compute_gradient_H_prev(dt)
        self.mg.levels[self.top_level].adjoint_operators.compute_gradient_smb()


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
    sigma_q : float
        Bunching exponent for sigma levels (q=1 uniform, q>1 bunches toward bed).
    n_smooth : int
        Maximum number of column smoothing sweeps per time step.
    update_rheology : bool
        If True, update grid.rheology.B after each step from the Arrhenius factor.
    rho_i : float
        Ice density for the B unit conversion (default 917 kg/m^3).
    """

    SEC_PER_YR = 365.25 * 86400.0

    def __init__(self, grid, nz=21, sigma_q=2.0, n_smooth=10,
                 update_rheology=True, rho_i=917.0):
        self.ops = EnthalpyOperators(grid, nz=nz, sigma_q=sigma_q)
        self.n_smooth = n_smooth
        self.update_rheology = update_rheology
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

    def step(self, dt):
        """
        Advance enthalpy by one time step.

        Reads the current velocity (u, v) and thickness (H) from the
        Grid, solves the column-wise enthalpy equation, and optionally
        writes the updated Arrhenius factor B back to the Grid.

        Parameters
        ----------
        dt : float
            Time step in seconds.
        """
        ops = self.ops

        # Sync velocity from the momentum solution.
        # SSA velocities are in m/yr; enthalpy needs m/s.
        ops.broadcast_velocity()
        sec_per_yr = cp.float32(self.SEC_PER_YR)
        ops.enthalpy_velocity.u3d /= sec_per_yr
        ops.enthalpy_velocity.v3d /= sec_per_yr

        ops.compute_sigma_dot()
        ops.compute_frictional_heating()

        # Enthalpy solve
        ops.set_rhs(dt)
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

