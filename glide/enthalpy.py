"""
Enthalpy model for GLIDE.

Solves the enthalpy advection-diffusion equation in terrain-following
(sigma) coordinates using a column-wise Newton/Thomas smoother
within the multigrid framework.

The enthalpy equation (Aschwanden et al. 2012):

    rho_i (dE/dt + u dE/dx + v dE/dy + omega dE/dsigma)
        - (1/h^2) d/dsigma(K dE/dsigma) = phi - rho_w L Dw(omega)

is discretized with:
    - Horizontal: finite volume, upwind fluxes on MAC grid
    - Vertical: finite differences on uniform sigma nodes
    - Column-wise Newton/Thomas solve as multigrid smoother
"""
import cupy as cp
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
K_COLD = K_I / C_I   # Enthalpy diffusion coeff k_i/c_i (kg/(m*s)), not thermal diffusivity
K_TEMP_FACTOR = 1e-1 # Temperate diffusivity reduction factor (Aschwanden et al. 2012)
K_TEMP = K_COLD * K_TEMP_FACTOR
E_SCALE = C_I * (T_MELT - T_REF)  # Enthalpy scale (J/kg) for non-dimensionalization
SEC_PER_YR = 365.25 * 86400.0
# Mass flux regularization: must match momentum solver (flux.cu: sqrt(u^2 + 10))
# where 10 is in (m/yr)^2. Convert to (m/s)^2 for the enthalpy solver.
MASS_FLUX_REG = 10.0 / (SEC_PER_YR ** 2)


def water_content_from_enthalpy(E, depth=0.0):
    """Extract water content from enthalpy."""
    T_pmp = T_MELT - BETA_CC * RHO_I * GRAVITY * depth
    E_pmp = C_I * (T_pmp - T_REF)
    return cp.maximum(E - E_pmp, 0.0) / L_HEAT


def make_sigma(nz):
    """
    Generate uniform sigma levels from bed (0) to surface (1).

    Parameters
    ----------
    nz : int
        Number of sigma levels.

    Returns
    -------
    sigma : cp.ndarray, shape (nz,)
        Sigma node positions in [0, 1].
    """
    return cp.linspace(0.0, 1.0, nz, dtype=cp.float32)


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
    Q_fh: cp.ndarray | None = None       # (ny, nx) basal frictional heating
    phi_strain: cp.ndarray | None = None # (ny, nx, nz) strain heating

    drain_rate: Constant = field(
        default_factory=lambda: Constant(
            value=cp.float32(0.01 / (365.25 * 86400.0)),
            name='drain_rate',
            units='s^{-1}',
            attrs={'long_name': 'Water drainage rate in temperate ice '
                                '(0.01 a^{-1} default)'})
    )

    h_thin: Constant = field(
        default_factory=lambda: Constant(
            value=cp.float32(25.0),
            name='h_thin',
            units='m',
            attrs={'long_name': 'Thickness below which the column is clamped '
                                'to the surface enthalpy (Dirichlet)'})
    )

    def __repr__(self):
        shapes = []
        if self.E_surface is not None:
            shapes.append(f'E_surface {self.E_surface.shape}')
        if self.Q_geo is not None:
            shapes.append(f'Q_geo {self.Q_geo.shape}')
        if self.Q_fh is not None:
            shapes.append(f'Q_fh {self.Q_fh.shape}')
        if self.phi_strain is not None:
            shapes.append(f'phi_strain {self.phi_strain.shape}')
        return f'EnthalpyForcing: {", ".join(shapes)}'


@dataclass
class EnthalpyVelocity:
    """3D velocity fields for enthalpy advection."""
    u3d: cp.ndarray | None = None       # (nz, ny, nx+1) x-velocity
    v3d: cp.ndarray | None = None       # (nz, ny+1, nx) y-velocity
    omega: cp.ndarray | None = None     # (ny, nx, nz) omega = H*omega

    def __repr__(self):
        if self.u3d is not None:
            return f'EnthalpyVelocity: u3d {self.u3d.shape}, v3d {self.v3d.shape}'
        return 'EnthalpyVelocity: uninitialized'


@dataclass
class EnthalpyTermFlags:
    """Flags controlling which physics terms are active in the enthalpy solver.

    All terms are enabled by default. Disabling terms is useful for
    debugging convergence and isolating the effect of individual physics.

    Strain heating (phi_strain) is not toggled here — it is a forcing
    field that defaults to zero. Set it externally if needed.
    """
    horizontal_advection: bool = True
    omega: bool = True
    drainage: bool = True

    @property
    def bitmask(self) -> int:
        """Pack flags into the integer bitmask expected by CUDA kernels."""
        return ((1 if self.horizontal_advection else 0)
              | ((1 if self.omega else 0) << 1)
              | ((1 if self.drainage else 0) << 2))


@dataclass
class ColumnSmootherConfig:
    """Configuration for the column-wise Newton/Thomas smoother.

    Convergence is checked after each sweep in column_sweep().
    Iteration stops when any of these conditions is met:
      - relative residual < relative_tolerance
      - absolute residual < absolute_tolerance
      - iteration count reaches n_iter (passed to column_sweep)
    This mirrors FASCDSolver.solve() in the momentum solver.
    """
    omega: cp.float32 = cp.float32(1.0)
    n_newton: int = 3
    relaxation: cp.float32 = cp.float32(1.0)
    relative_tolerance: cp.float32 = cp.float32(1e-5)
    absolute_tolerance: cp.float32 = cp.float32(1e-2)
    lf_c: cp.float32 = cp.float32(1e-12)
    report_norms: bool = False
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
    - omega field: set externally by the momentum/mass solver
    - set_rhs: set forcing / previous time step data

    Parallels the ForwardOperators class for the SSA solver.
    """

    def __init__(self, grid, nz=11, use_fast_math=True):
        self.grid = grid
        self.nz = nz

        # Sigma levels (uniform spacing)
        self.sigma = make_sigma(nz)

        # Compile CUDA kernels.
        # Physical constants are injected as #define directives so that
        # Python remains the single source of truth.
        cuda_dir = Path(__file__).parent / "cuda"
        cuda_files = ['common.cu', 'enthalpy.cu']
        constants = {
            'RHO_I': RHO_I, 'RHO_W': RHO_W, 'C_I': C_I, 'K_I': K_I,
            'L_HEAT': L_HEAT, 'T_REF': T_REF, 'T_MELT': T_MELT,
            'BETA_CC': BETA_CC, 'GRAVITY': GRAVITY,
            'K_TEMP_FACTOR': K_TEMP_FACTOR,
            'MASS_FLUX_REG': MASS_FLUX_REG,
        }
        defines = '\n'.join(f'#define {k} {v}f' for k, v in constants.items())
        defines += '\n#define K_COLD (K_I/C_I)\n'
        defines += '#define E_SCALE (C_I * (T_MELT - T_REF))\n'
        file_source = '\n'.join((cuda_dir / f).read_text() for f in cuda_files)
        cuda_source = defines + '\n' + file_source

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
            omega=cp.zeros((ny, nx, nz), dtype=cp.float32),
        )

        # Forcing / BCs
        self.enthalpy_forcing = EnthalpyForcing(
            E_surface=cp.zeros((ny, nx), dtype=cp.float32),
            Q_geo=cp.zeros((ny, nx), dtype=cp.float32),
            Q_fh=cp.zeros((ny, nx), dtype=cp.float32),
            phi_strain=cp.zeros((ny, nx, nz), dtype=cp.float32),
        )

        # Work arrays
        self.H_prev = cp.zeros((ny, nx), dtype=cp.float32)  # thickness before momentum step
        self.r_E = cp.zeros((ny, nx, nz), dtype=cp.float32)
        self.f_E = cp.zeros((ny, nx, nz), dtype=cp.float32)  # precomputed forcing
        self.delta_E = cp.zeros((ny, nx, nz), dtype=cp.float32)

        self.smoother_config = ColumnSmootherConfig()
        self.term_flags = EnthalpyTermFlags()

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

    def compute_omega(self, dh_dt, bmb=None):
        """
        Compute omega = H * sigma_dot by integrating the sigma-space
        continuity equation upward from the bed.

        Uses the actual dH/dt (passed by the caller) instead of
        estimating it from SMB - div(Hu). This ensures exact
        consistency with the realized thickness change from the
        momentum step, so uniform enthalpy produces zero residual.

        Parameters
        ----------
        dh_dt : cp.ndarray, shape (ny, nx)
            Thickness change rate in m/s. For coupled runs this is
            (H_new - H_prev) / dt from the momentum step. For
            standalone runs, pass SMB (m/s) for a steady-state estimate.
        bmb : cp.ndarray, shape (ny, nx), optional
            Basal mass balance in m/s. Defaults to 0.
        """
        kernel = self.kernels.get_function('compute_omega')
        grid_size, block_size = self._column_launch_config

        grid = self.grid
        vel = self.enthalpy_velocity

        if bmb is None:
            bmb = cp.zeros((grid.ny, grid.nx), dtype=cp.float32)

        kernel(grid_size, block_size,
               (vel.omega,
                vel.u3d, vel.v3d,
                grid.state.H.data,
                dh_dt, bmb,
                grid.dx,
                grid.ny, grid.nx, self.nz))



    def compute_residual(self, dt):
        """Compute the enthalpy residual at all grid points."""
        kernel = self.kernels.get_function('enthalpy_compute_residual')
        grid_size, block_size = self._column_launch_config

        vel = self.enthalpy_velocity
        grid = self.grid

        self.r_E.fill(0)
        kernel(grid_size, block_size,
               (self.r_E,
                self.enthalpy_state.E, self.f_E,
                vel.u3d, vel.v3d, vel.omega,
                grid.state.H.data,
                self.enthalpy_forcing.E_surface,
                grid.dx, cp.float32(dt),
                self.enthalpy_forcing.drain_rate.value,
                self.enthalpy_forcing.h_thin.value,
                self.smoother_config.lf_c,
                cp.int32(self.term_flags.bitmask),
                cp.int32(1),  # use_forcing=true
                grid.ny, grid.nx, self.nz))

        return float(cp.max(cp.abs(self.r_E)))

    def column_smooth(self, dt):
        """
        One application of the column-wise Newton/Thomas smoother.

        Freezes horizontal neighbors, solves each column's tridiagonal
        system via Newton iteration with the Thomas algorithm.
        """
        kernel = self.kernels.get_function('enthalpy_column_smooth')
        grid_size, block_size = self._column_launch_config

        vel = self.enthalpy_velocity
        grid = self.grid
        cfg = self.smoother_config

        self.delta_E.fill(0)
        kernel(grid_size, block_size,
               (self.delta_E,
                self.enthalpy_state.E, self.f_E,
                vel.u3d, vel.v3d, vel.omega,
                grid.state.H.data,
                self.enthalpy_forcing.E_surface,
                grid.dx, cp.float32(dt),
                self.enthalpy_forcing.drain_rate.value,
                self.enthalpy_forcing.h_thin.value,
                cfg.lf_c,
                cp.int32(self.term_flags.bitmask),
                grid.ny, grid.nx, self.nz,
                cfg.n_newton, cfg.relaxation))

    def layer_smooth(self, dt):
        """
        One application of the pointwise Jacobi layer smoother.

        Computes delta_E = -R / J_diag at every node simultaneously.
        One thread per (i,j,k) — fully parallel, no data dependencies.
        """
        kernel = self.kernels.get_function('enthalpy_layer_smooth')
        ny, nx, nz = self.grid.ny, self.grid.nx, self.nz

        total_work = ny * nx * nz
        block_size = 256
        grid_size = (total_work + block_size - 1) // block_size

        vel = self.enthalpy_velocity
        grid = self.grid

        self.delta_E.fill(0)
        kernel((grid_size,), (block_size,),
               (self.delta_E,
                self.enthalpy_state.E, self.f_E,
                vel.u3d, vel.v3d, vel.omega,
                grid.state.H.data,
                self.enthalpy_forcing.E_surface,
                grid.dx, cp.float32(dt),
                self.enthalpy_forcing.drain_rate.value,
                self.enthalpy_forcing.h_thin.value,
                self.smoother_config.lf_c,
                cp.int32(self.term_flags.bitmask),
                ny, nx, nz))

    def column_sweep(self, dt, n_iter, alternating=True):
        """
        Repeated smoothing with convergence checking.

        Each iteration applies a column sweep (exact vertical solve)
        followed optionally by a layer sweep (pointwise Jacobi on
        horizontal error). Convergence is checked after each iteration
        against the full PDE residual.

        Parameters
        ----------
        dt : float
            Time step size.
        n_iter : int
            Maximum number of smoothing sweeps.
        alternating : bool
            If True (default), follow each column sweep with a layer
            sweep for faster horizontal convergence.
        """
        cfg = self.smoother_config

        # Initial residual
        r0 = float(self.compute_residual(dt))
        if cfg.report_norms:
            print(f"  Enthalpy initial: |r0| = {r0:.2e}")

        for iteration in range(n_iter):
            self.column_smooth(dt)
            self.enthalpy_state.E[:] += cfg.omega * self.delta_E

            if alternating:
                self.layer_smooth(dt)
                self.enthalpy_state.E[:] += cfg.omega * self.delta_E

            cfg.hook_func(iteration)

            # Convergence check
            r = float(self.compute_residual(dt))
            rel = r / r0 if r0 > 0 else 0.0

            if cfg.report_norms:
                print(f"  Enthalpy sweep {iteration}: "
                      f"|r|/|r0| = {rel:.2e}, |r| = {r:.2e}")

            if rel < cfg.relative_tolerance or r < cfg.absolute_tolerance:
                break

    def set_rhs(self, dt, snapshot=True):
        """Precompute the forcing array f_E.

        Optionally snapshots E_prev and H_prev from the current state,
        then precomputes f_E which encodes all E-independent forcing:

            f_E = rho_i * H_prev * E_prev / dt
                + H * phi_strain
                + (Q_geo + Q_fh) / dsig_half  [at k=0 only]

        The residual kernel then computes R = F(E) - f_E.
        On the finest grid, set_rhs is called once per time step.
        On coarse grids, f_E is overwritten by the FAS defect equation.

        Parameters
        ----------
        dt : float
            Time step size.
        snapshot : bool
            If True (default), snapshot E_prev and H_prev from the
            current state before computing f_E. Set to False when
            E_prev/H_prev were already captured earlier (e.g., by
            pre_momentum() in the coupled ThermalModel).
        """
        if snapshot:
            self.enthalpy_state.E_prev[:] = self.enthalpy_state.E
            self.H_prev[:] = self.grid.state.H.data

        dsig = 1.0 / (self.nz - 1)
        dsig_half = 0.5 * dsig

        H = self.grid.state.H.data
        forcing = self.enthalpy_forcing

        # Interior and bed nodes: time forcing + strain heating.
        # E_prev is in scaled units; phi_strain and heat fluxes are
        # physical (W/m^2 etc.) and must be divided by E_SCALE.
        e_scale = cp.float32(E_SCALE)
        for k in range(self.nz - 1):
            self.f_E[:, :, k] = (
                RHO_I * self.H_prev * self.enthalpy_state.E_prev[:, :, k]
                / cp.float32(dt)
                + H * forcing.phi_strain[:, :, k] / e_scale
            )

        # Bed node (k=0): add Neumann heat flux
        self.f_E[:, :, 0] += (forcing.Q_geo + forcing.Q_fh) / (cp.float32(dsig_half) * e_scale)

        # Surface node: Dirichlet row (f_E = 0, handled by the kernel)
        self.f_E[:, :, self.nz - 1] = 0.0

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
        self.enthalpy_forcing.E_surface[:] = C_I * (T_capped - T_REF) / E_SCALE

    def initialize_from_temperature(self, T_field):
        """
        Initialize the 3D enthalpy field from a temperature field.

        Parameters
        ----------
        T_field : array-like, shape (ny, nx, nz), (ny, nx), or scalar
            Temperature in Kelvin. 2D arrays are broadcast to all sigma levels.
        """
        T = cp.asarray(T_field, dtype=cp.float32)
        if T.ndim == 0:
            T = cp.full((self.grid.ny, self.grid.nx, self.nz),
                        T.item(), dtype=cp.float32)
        elif T.ndim == 2:
            T = cp.broadcast_to(T[:, :, None],
                                (self.grid.ny, self.grid.nx, self.nz)).copy()

        # Cap temperature at the local pressure melting point.
        # Ice cannot exceed T_pmp; excess energy would be latent heat
        # (water content), which should be set explicitly if needed.
        H = self.grid.state.H.data
        for k in range(self.nz):
            sigma_k = float(self.sigma[k])
            depth = (1.0 - sigma_k) * H
            T_pmp = T_MELT - BETA_CC * RHO_I * GRAVITY * depth
            T[:, :, k] = cp.minimum(T[:, :, k], T_pmp)

        self.enthalpy_state.E[:] = C_I * (T - T_REF) / E_SCALE
        self.enthalpy_state.E_prev[:] = self.enthalpy_state.E
        self.H_prev[:] = self.grid.state.H.data

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
            T[:, :, k] = cp.minimum(E[:, :, k] * E_SCALE / C_I + T_REF, T_pmp)
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
            omega[:, :, k] = water_content_from_enthalpy(E[:, :, k] * E_SCALE, depth)
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
            omega = water_content_from_enthalpy(E[:, :, k] * E_SCALE, depth)
            A[:, :, k] *= (1.0 + 181.25 * omega)

        # Depth average and convert to B
        A_avg = cp.mean(A, axis=2)
        B_avg = A_avg ** (-1.0 / n_glen)

        return B_avg

