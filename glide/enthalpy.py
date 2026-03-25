"""
Enthalpy model for GLIDE.

Solves the enthalpy advection-diffusion equation in terrain-following
(sigma) coordinates using a column-wise Newton/Thomas smoother
within the multigrid framework.

The enthalpy equation (Aschwanden et al. 2012):

    rho_i (dE/dt + u dE/dx + v dE/dy + sigma_dot dE/dsigma)
        - (1/h^2) d/dsigma(K dE/dsigma) = phi - rho_w L Dw(omega)

is discretized with:
    - Horizontal: finite volume, upwind fluxes on MAC grid
    - Vertical: finite differences on non-uniform sigma nodes
    - Column-wise Newton/Thomas solve as multigrid smoother
"""
import cupy as cp
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from .field import Field, Constant, GridEntity


# ========================================================
# Physical constants
# ========================================================
RHO_I = 910.0       # Ice density (kg/m^3)
RHO_W = 1000.0      # Water density (kg/m^3)
C_I = 2009.0         # Heat capacity of ice (J/(kg*K))
K_I = 2.1            # Thermal conductivity of ice (W/(m*K))
L_HEAT = 3.34e5      # Latent heat of fusion (J/kg)
T_REF = 223.15       # Reference temperature (K)
T_MELT = 273.15      # Melting point at standard pressure (K)
BETA_CC = 7.9e-8     # Clausius-Clapeyron constant (K/Pa)
GRAVITY = 9.81       # Gravitational acceleration (m/s^2)
K_COLD = K_I / C_I   # Diffusivity for cold ice (m^2/s)
K_TEMP_FACTOR = 1e-5 # Temperate diffusivity reduction factor
K_TEMP = K_COLD * K_TEMP_FACTOR


def enthalpy_from_temperature(T, depth=0.0):
    """Convert temperature to enthalpy (cold ice only)."""
    return C_I * (T - T_REF)


def temperature_from_enthalpy(E, depth=0.0):
    """Convert enthalpy to temperature (clipped at pressure melting point)."""
    T_pmp = T_MELT - BETA_CC * RHO_I * GRAVITY * depth
    E_pmp = C_I * (T_pmp - T_REF)
    T = E / C_I + T_REF
    return cp.minimum(T, T_pmp)


def water_content_from_enthalpy(E, depth=0.0):
    """Extract water content from enthalpy."""
    T_pmp = T_MELT - BETA_CC * RHO_I * GRAVITY * depth
    E_pmp = C_I * (T_pmp - T_REF)
    return cp.maximum(E - E_pmp, 0.0) / L_HEAT


def make_sigma(nz, q=2.0):
    """
    Generate non-uniform sigma levels bunched toward the bed.

    Parameters
    ----------
    nz : int
        Number of sigma levels.
    q : float
        Bunching exponent. q=1 is uniform, q>1 bunches toward bed (sigma=0).

    Returns
    -------
    sigma : cp.ndarray, shape (nz,)
        Sigma node positions in [0, 1].
    """
    zeta = np.linspace(0.0, 1.0, nz)
    sigma = zeta ** q
    return cp.array(sigma, dtype=cp.float32)


# ========================================================
# Dataclasses for enthalpy state and parameters
# ========================================================
@dataclass
class EnthalpyState:
    """Enthalpy field and related 3D state variables."""
    E: cp.ndarray | None = None          # (ny, nx, nz) enthalpy
    E_prev: cp.ndarray | None = None     # (ny, nx, nz) previous time step

    def __repr__(self):
        if self.E is not None:
            return f'EnthalpyState: E {self.E.shape}'
        return 'EnthalpyState: uninitialized'


@dataclass
class EnthalpyForcing:
    """Boundary conditions and forcing for the enthalpy equation."""
    E_surface: cp.ndarray | None = None  # (ny, nx) surface enthalpy BC
    Q_geo: cp.ndarray | None = None      # (ny, nx) geothermal heat flux
    phi_strain: cp.ndarray | None = None # (ny, nx, nz) strain heating

    drain_rate: Constant = field(
        default_factory=lambda: Constant(
            value=cp.float32(0.01),
            name='drain_rate',
            units='a^{-1}',
            attrs={'long_name': 'Water drainage rate in temperate ice'})
    )

    def __repr__(self):
        shapes = []
        if self.E_surface is not None:
            shapes.append(f'E_surface {self.E_surface.shape}')
        if self.Q_geo is not None:
            shapes.append(f'Q_geo {self.Q_geo.shape}')
        if self.phi_strain is not None:
            shapes.append(f'phi_strain {self.phi_strain.shape}')
        return f'EnthalpyForcing: {", ".join(shapes)}'


@dataclass
class EnthalpyVelocity:
    """3D velocity fields for enthalpy advection."""
    u3d: cp.ndarray | None = None       # (nz, ny, nx+1) x-velocity
    v3d: cp.ndarray | None = None       # (nz, ny+1, nx) y-velocity
    sigma_dot: cp.ndarray | None = None # (ny, nx, nz) sigma velocity

    def __repr__(self):
        if self.u3d is not None:
            return f'EnthalpyVelocity: u3d {self.u3d.shape}, v3d {self.v3d.shape}'
        return 'EnthalpyVelocity: uninitialized'


@dataclass
class ColumnSmootherConfig:
    """Configuration for the column-wise Newton/Thomas smoother."""
    omega: cp.float32 = cp.float32(1.0)
    n_newton: int = 3
    relaxation: cp.float32 = cp.float32(1.0)
    hook_func: Callable[[int], None] = field(
        default_factory=lambda: lambda i: None)


# ========================================================
# Enthalpy operators (CUDA kernel interface)
# ========================================================
class EnthalpyOperators:
    """
    GPU-accelerated operators for the enthalpy equation.

    Provides:
    - compute_residual: full enthalpy residual at all (i,j,k)
    - column_smooth: column-wise Newton/Thomas smoother
    - column_sweep: repeated smoothing with solution update
    - compute_sigma_dot: sigma velocity from 3D velocity field
    - set_rhs: set forcing / previous time step data

    Parallels the ForwardOperators class for the SSA solver.
    """

    def __init__(self, grid, nz=11, sigma_q=2.0, use_fast_math=True):
        self.grid = grid
        self.nz = nz

        # Sigma levels
        self.sigma = make_sigma(nz, q=sigma_q)

        # Compile CUDA kernels
        cuda_dir = Path(__file__).parent / "cuda"
        # Include common.cu for get_cell helper
        cuda_files = ['common.cu', 'enthalpy.cu']
        cuda_source = '\n'.join((cuda_dir / f).read_text() for f in cuda_files)

        options = ("--use_fast_math",) if use_fast_math else ()
        self.kernels = cp.RawModule(code=cuda_source, options=options)

        ny, nx = grid.ny, grid.nx

        # Enthalpy state
        self.enthalpy_state = EnthalpyState(
            E=cp.zeros((ny, nx, nz), dtype=cp.float32),
            E_prev=cp.zeros((ny, nx, nz), dtype=cp.float32),
        )

        # 3D velocity
        self.enthalpy_velocity = EnthalpyVelocity(
            u3d=cp.zeros((nz, ny, nx + 1), dtype=cp.float32),
            v3d=cp.zeros((nz, ny + 1, nx), dtype=cp.float32),
            sigma_dot=cp.zeros((ny, nx, nz), dtype=cp.float32),
        )

        # Forcing / BCs
        self.enthalpy_forcing = EnthalpyForcing(
            E_surface=cp.zeros((ny, nx), dtype=cp.float32),
            Q_geo=cp.zeros((ny, nx), dtype=cp.float32),
            phi_strain=cp.zeros((ny, nx, nz), dtype=cp.float32),
        )

        # Work arrays
        self.r_E = cp.zeros((ny, nx, nz), dtype=cp.float32)
        self.delta_E = cp.zeros((ny, nx, nz), dtype=cp.float32)
        self.Q_fh = cp.zeros((ny, nx), dtype=cp.float32)

        self.smoother_config = ColumnSmootherConfig()

    @property
    def _column_launch_config(self):
        """Launch config: one thread per column."""
        n_columns = self.grid.ny * self.grid.nx
        block_size = 256
        grid_size = (n_columns + block_size - 1) // block_size
        return (grid_size,), (block_size,)

    def broadcast_velocity(self):
        """
        Broadcast 2D SSA velocities to all sigma layers.

        Copies the depth-averaged u, v from the grid state
        into the 3D velocity arrays for enthalpy advection.
        """
        u2d = self.grid.state.u.data  # (ny, nx+1)
        v2d = self.grid.state.v.data  # (ny+1, nx)

        for k in range(self.nz):
            self.enthalpy_velocity.u3d[k, :, :] = u2d
            self.enthalpy_velocity.v3d[k, :, :] = v2d

    def compute_sigma_dot(self):
        """Compute sigma velocity from 3D velocity field."""
        kernel = self.kernels.get_function('compute_sigma_dot')
        grid_size, block_size = self._column_launch_config

        grid = self.grid
        vel = self.enthalpy_velocity

        kernel(grid_size, block_size,
               (vel.sigma_dot,
                vel.u3d, vel.v3d,
                grid.state.H.data,
                grid.geometry.bed.data,
                grid.forcing.smb.data,
                self.sigma,
                grid.dx, cp.float32(0.0),  # dt not needed for sigma_dot
                grid.ny, grid.nx, self.nz))

    def compute_frictional_heating(self):
        """
        Compute basal frictional heating Q_fh = beta * |u_b|^(m+1).

        Uses the bed-level (k=0) velocities.
        """
        grid = self.grid
        sliding = grid.sliding
        u_bed = self.enthalpy_velocity.u3d[0, :, :]  # (ny, nx+1)
        v_bed = self.enthalpy_velocity.v3d[0, :, :]  # (ny+1, nx)

        # Interpolate to cell centers
        u_cell = 0.5 * (u_bed[:, 1:] + u_bed[:, :-1])
        v_cell = 0.5 * (v_bed[1:, :] + v_bed[:-1, :])
        speed_sq = u_cell**2 + v_cell**2 + sliding.u_reg.value**2
        speed = cp.sqrt(speed_sq)

        m = sliding.m.value
        self.Q_fh[:, :] = sliding.beta.data * speed ** (m + 1.0)

    def compute_residual(self, dt):
        """Compute the enthalpy residual at all grid points."""
        kernel = self.kernels.get_function('enthalpy_compute_residual')
        grid_size, block_size = self._column_launch_config

        state = self.enthalpy_state
        vel = self.enthalpy_velocity
        forcing = self.enthalpy_forcing
        grid = self.grid

        self.r_E.fill(0)
        kernel(grid_size, block_size,
               (self.r_E,
                state.E, state.E_prev,
                vel.u3d, vel.v3d, vel.sigma_dot,
                grid.state.H.data,
                forcing.phi_strain,
                forcing.E_surface,
                forcing.Q_geo,
                self.Q_fh,
                self.sigma,
                grid.dx, cp.float32(dt),
                forcing.drain_rate.value,
                grid.ny, grid.nx, self.nz))

        return cp.linalg.norm(self.r_E)

    def column_smooth(self, dt):
        """
        One application of the column-wise Newton/Thomas smoother.

        Freezes horizontal neighbors, solves each column's tridiagonal
        system via Newton iteration with the Thomas algorithm.
        """
        kernel = self.kernels.get_function('enthalpy_column_smooth')
        grid_size, block_size = self._column_launch_config

        state = self.enthalpy_state
        vel = self.enthalpy_velocity
        forcing = self.enthalpy_forcing
        grid = self.grid
        cfg = self.smoother_config

        self.delta_E.fill(0)
        kernel(grid_size, block_size,
               (self.delta_E,
                state.E, state.E_prev,
                vel.u3d, vel.v3d, vel.sigma_dot,
                grid.state.H.data,
                forcing.phi_strain,
                forcing.E_surface,
                forcing.Q_geo,
                self.Q_fh,
                self.sigma,
                grid.dx, cp.float32(dt),
                forcing.drain_rate.value,
                grid.ny, grid.nx, self.nz,
                cfg.n_newton, cfg.relaxation))

    def column_sweep(self, dt, n_iter):
        """
        Repeated column smoothing with solution update.

        Parameters
        ----------
        dt : float
            Time step size.
        n_iter : int
            Number of smoothing sweeps.
        """
        for iteration in range(n_iter):
            self.column_smooth(dt)
            self.enthalpy_state.E[:] += (
                self.smoother_config.omega * self.delta_E)
            self.smoother_config.hook_func(iteration)

    def set_rhs(self, dt):
        """Store previous time step enthalpy and update forcing."""
        self.enthalpy_state.E_prev[:] = self.enthalpy_state.E

    def set_surface_enthalpy_from_temperature(self, T_surface):
        """
        Set surface enthalpy BC from a temperature field.

        Parameters
        ----------
        T_surface : array-like, shape (ny, nx)
            Surface temperature in Kelvin. Capped at T_melt.
        """
        T_capped = cp.minimum(cp.asarray(T_surface, dtype=cp.float32),
                              cp.float32(T_MELT))
        self.enthalpy_forcing.E_surface[:] = C_I * (T_capped - T_REF)

    def initialize_from_temperature(self, T_field):
        """
        Initialize the 3D enthalpy field from a temperature field.

        Parameters
        ----------
        T_field : array-like, shape (ny, nx, nz) or scalar
            Temperature in Kelvin.
        """
        T = cp.asarray(T_field, dtype=cp.float32)
        if T.ndim == 0:
            T = cp.full((self.grid.ny, self.grid.nx, self.nz),
                        T.item(), dtype=cp.float32)
        self.enthalpy_state.E[:] = C_I * (T - T_REF)
        self.enthalpy_state.E_prev[:] = self.enthalpy_state.E

    def get_temperature(self):
        """
        Extract temperature from enthalpy (clipped at pressure melting point).

        Returns
        -------
        T : cp.ndarray, shape (ny, nx, nz)
        """
        E = self.enthalpy_state.E
        H = self.grid.state.H.data
        T = cp.zeros_like(E)
        for k in range(self.nz):
            sigma_k = float(self.sigma[k])
            depth = (1.0 - sigma_k) * H
            T_pmp = T_MELT - BETA_CC * RHO_I * GRAVITY * depth
            T[:, :, k] = cp.minimum(E[:, :, k] / C_I + T_REF, T_pmp)
        return T

    def get_water_content(self):
        """
        Extract water content from enthalpy.

        Returns
        -------
        omega : cp.ndarray, shape (ny, nx, nz)
        """
        E = self.enthalpy_state.E
        H = self.grid.state.H.data
        omega = cp.zeros_like(E)
        for k in range(self.nz):
            sigma_k = float(self.sigma[k])
            depth = (1.0 - sigma_k) * H
            omega[:, :, k] = water_content_from_enthalpy(E[:, :, k], depth)
        return omega

    def get_arrhenius_factor(self):
        """
        Compute the Arrhenius factor A(T, omega) from enthalpy.

        Returns the depth-averaged B = A^{-1/n} for use in the SSA solver.

        Returns
        -------
        B_avg : cp.ndarray, shape (ny, nx)
            Depth-averaged rate factor B = A^{-1/n}.
        """
        T = self.get_temperature()
        n_glen = float(self.grid.rheology.n.value)

        # Paterson-Budd law
        A_cold = 3.985e-13   # s^-1 Pa^-3, T < 263.15 K
        A_warm = 1.916e3     # s^-1 Pa^-3, T >= 263.15 K
        Q_cold = 60e3        # J/mol
        Q_warm = 139e3       # J/mol
        R_gas = 8.314        # J/(mol*K)
        T_threshold = 263.15

        # Compute A at each point
        T_pa = T  # pressure-adjusted temperature (already in the right frame)
        A_factor = cp.where(T_pa < T_threshold, A_cold, A_warm)
        Q_factor = cp.where(T_pa < T_threshold, Q_cold, Q_warm)

        T_safe = cp.maximum(T_pa, 1.0)  # avoid division by zero
        A = A_factor * cp.exp(-Q_factor / (R_gas * T_safe))

        # Water content enhancement
        E = self.enthalpy_state.E
        H = self.grid.state.H.data
        for k in range(self.nz):
            sigma_k = float(self.sigma[k])
            depth = (1.0 - sigma_k) * H
            omega = water_content_from_enthalpy(E[:, :, k], depth)
            A[:, :, k] *= (1.0 + 181.25 * omega)

        # Depth average and convert to B
        A_avg = cp.mean(A, axis=2)
        B_avg = A_avg ** (-1.0 / n_glen)

        return B_avg
