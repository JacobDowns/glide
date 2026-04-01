"""
Standalone NumPy column smoother for the enthalpy equation.

Mirrors the discretization in glide/cuda/enthalpy.cu exactly:
  - Uniform sigma grid, bed at sigma=0, surface at sigma=1
  - Centered vertical diffusion with K evaluated at half-nodes
  - Upwind vertical advection
  - Neumann basal BC with half-cell control volume
  - Dirichlet surface BC
  - Frozen-K Jacobian (K held constant in derivatives, not in residual)
  - Newton/Thomas solve

This is a single-column solver — no horizontal coupling. Horizontal
advection and strain heating can be injected as external source terms.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Callable


# ---- Physical constants (same as enthalpy.cu) ----
RHO_I = 910.0
RHO_W = 1000.0
C_I = 2009.0
K_I = 2.1
L_HEAT = 3.34e5
T_REF = 223.15
T_MELT = 273.15
BETA_CC = 7.9e-8
GRAVITY = 9.81
K_COLD = K_I / C_I
K_TEMP_FACTOR = 1e-1
SECONDS_PER_YEAR = 365.25 * 24 * 3600


# ---- Helpers matching enthalpy.cu ----

def get_E_pmp(sigma, H):
    """Enthalpy at the pressure melting point."""
    depth = (1.0 - sigma) * H
    T_pmp = T_MELT - BETA_CC * RHO_I * GRAVITY * depth
    return C_I * (T_pmp - T_REF)


def get_K(E, E_pmp):
    """Diffusivity K(E) with smooth sigmoid transition."""
    z = np.clip((E - E_pmp) * 0.01, -20.0, 20.0)
    s = 1.0 / (1.0 + np.exp(-z))
    return K_COLD * (1.0 - s + K_TEMP_FACTOR * s)


def enthalpy_from_temperature(T):
    return C_I * (T - T_REF)


def temperature_from_enthalpy(E, sigma, H):
    E_pmp = get_E_pmp(sigma, H)
    T = E / C_I + T_REF
    T_pmp = E_pmp / C_I + T_REF
    return np.minimum(T, T_pmp)


def water_content_from_enthalpy(E, sigma, H):
    E_pmp = get_E_pmp(sigma, H)
    return np.maximum(E - E_pmp, 0.0) / L_HEAT


# Transition sharpness for the drainage sigmoid (same as K(E) in enthalpy.cu)
DRAIN_SHARPNESS = 0.01


def get_omega_smooth(E, E_pmp):
    """Smoothed water content using a logistic approximation of max(E - E_pmp, 0).

    softplus(x) = log(1 + exp(alpha*x)) / alpha  approximates max(x, 0).
    Using the same sharpness as the K(E) sigmoid for consistency.
    """
    alpha = DRAIN_SHARPNESS
    x = E - E_pmp
    ax = alpha * x
    # Numerically stable softplus: for large ax use x, for small use exp(ax)/alpha
    ax_clipped = np.clip(ax, -20.0, 20.0)
    softplus = np.where(ax_clipped > 10.0, x,
                        np.log1p(np.exp(ax_clipped)) / alpha)
    return softplus / L_HEAT


def get_domega_dE_smooth(E, E_pmp):
    """Derivative d(omega_smooth)/dE — the sigmoid itself."""
    alpha = DRAIN_SHARPNESS
    z = np.clip((E - E_pmp) * alpha, -20.0, 20.0)
    s = 1.0 / (1.0 + np.exp(-z))
    return s / L_HEAT


def E_history_to_fields(E_hist, cfg):
    """Convert E history (n_times, nz) to temperature and water content arrays."""
    sigma = cfg.sigma
    nz = E_hist.shape[1]
    T = np.zeros_like(E_hist)
    omega = np.zeros_like(E_hist)
    for k in range(nz):
        T[:, k] = np.minimum(E_hist[:, k] / C_I + T_REF,
                              T_MELT - BETA_CC * RHO_I * GRAVITY * (1 - sigma[k]) * cfg.H)
        E_pmp_k = get_E_pmp(sigma[k], cfg.H)
        omega[:, k] = np.maximum(E_hist[:, k] - E_pmp_k, 0.0) / L_HEAT
    return T, omega


# ---- Column configuration ----

@dataclass
class ColumnConfig:
    """Parameters for a single-column enthalpy problem."""
    H: float = 1000.0               # Ice thickness (m)
    nz: int = 21                     # Number of sigma levels
    dt: float = 1.0                  # Time step (seconds)
    T_surface: float = 243.15       # Surface temperature (K)
    Q_geo: float = 0.042            # Geothermal heat flux (W/m^2)
    Q_fh: float = 0.0              # Basal frictional heating (W/m^2)
    sigma_dot: np.ndarray | None = None   # (nz,) sigma velocity
    phi_strain: np.ndarray | None = None  # (nz,) strain heating
    horiz_source: np.ndarray | None = None  # (nz,) external horizontal advection source
    horiz_source_jac: np.ndarray | None = None  # (nz,) diagonal Jacobian of horiz source
    drain_rate: float = 0.0         # Drainage rate (s^-1), 0 = off
    n_newton: int = 3               # Newton steps per smooth
    relaxation: float = 1.0         # Newton relaxation factor

    @property
    def sigma(self):
        return np.linspace(0.0, 1.0, self.nz)

    @property
    def dsig(self):
        return 1.0 / (self.nz - 1)

    @property
    def E_surface(self):
        T_s = min(self.T_surface, T_MELT)
        return C_I * (T_s - T_REF)

    def with_dt_years(self, dt_years):
        """Return a copy with dt set from a value in years."""
        import copy
        cfg = copy.copy(self)
        cfg.dt = dt_years * SECONDS_PER_YEAR
        return cfg


# ---- Residual computation ----

def compute_residual(E, E_prev, cfg: ColumnConfig):
    """
    Compute the column residual r[k] for k = 0, ..., nz-1.

    Matches enthalpy_compute_residual in enthalpy.cu for a single column
    with no horizontal neighbors.

    Returns
    -------
    r : ndarray, shape (nz,)
    """
    nz = cfg.nz
    dsig = cfg.dsig
    h = cfg.H
    h_inv = 1.0 / h
    h2_inv = h_inv * h_inv
    dt = cfg.dt
    sigma = cfg.sigma

    phi = cfg.phi_strain if cfg.phi_strain is not None else np.zeros(nz)
    horiz = cfg.horiz_source if cfg.horiz_source is not None else np.zeros(nz)
    sd = cfg.sigma_dot if cfg.sigma_dot is not None else np.zeros(nz)

    r = np.zeros(nz)

    for k in range(nz):
        # Surface: Dirichlet, residual = 0
        if k == nz - 1:
            r[k] = 0.0
            continue

        E_k = E[k]
        E_pmp_k = get_E_pmp(sigma[k], h)

        # Time derivative
        rk = RHO_I * (E_k - E_prev[k]) / dt

        # Horizontal advection (external source)
        rk += horiz[k]

        if k == 0:
            # ---- Bed boundary ----
            E_kp1 = E[1]
            E_pmp_0 = get_E_pmp(0.0, h)
            E_pmp_1 = get_E_pmp(dsig, h)

            # Bed sigma advection (one-sided)
            sd_k = sd[0]
            if sd_k < 0.0:
                rk += RHO_I * sd_k * (E_kp1 - E_k) / dsig

            # Bed diffusion with half-cell control volume
            K_half = get_K(0.5 * (E_k + E_kp1), 0.5 * (E_pmp_0 + E_pmp_1))
            dsig_half = 0.5 * dsig
            rk += (-h2_inv * K_half * (E_kp1 - E_k) / dsig
                    - (cfg.Q_geo + cfg.Q_fh) * h_inv) / dsig_half

            # Strain heating
            rk -= phi[0]

            # Drainage (smoothed)
            omega = get_omega_smooth(E_k, E_pmp_0)
            rk += RHO_W * L_HEAT * cfg.drain_rate * omega

        else:
            # ---- Interior layers (k = 1 .. nz-2) ----
            E_km1 = E[k - 1]
            E_kp1 = E[k + 1] if k < nz - 1 else cfg.E_surface

            E_pmp_km1 = get_E_pmp(sigma[max(k - 1, 0)], h)
            E_pmp_kp1 = get_E_pmp(sigma[min(k + 1, nz - 1)], h)

            # Sigma advection (upwind)
            sd_k = sd[k]
            sd_pos = max(sd_k, 0.0)
            sd_neg = min(sd_k, 0.0)
            rk += RHO_I * (sd_pos * (E_k - E_km1) + sd_neg * (E_kp1 - E_k)) / dsig

            # Interior diffusion
            K_upper = get_K(0.5 * (E_k + E_kp1), 0.5 * (E_pmp_k + E_pmp_kp1))
            K_lower = get_K(0.5 * (E_k + E_km1), 0.5 * (E_pmp_k + E_pmp_km1))
            dsig2_inv = 1.0 / (dsig * dsig)
            rk += -h2_inv * dsig2_inv * (K_upper * (E_kp1 - E_k)
                                         - K_lower * (E_k - E_km1))

            # Strain heating + drainage (smoothed)
            rk -= phi[k]
            omega = get_omega_smooth(E_k, E_pmp_k)
            rk += RHO_W * L_HEAT * cfg.drain_rate * omega

        r[k] = rk

    return r


# ---- Column Newton/Thomas solve ----

def column_solve(E, E_prev, cfg: ColumnConfig):
    """
    Newton/Thomas solve for a single column.

    Mirrors enthalpy_column_smooth in enthalpy.cu. Freezes K in the
    Jacobian, uses the Thomas algorithm for the tridiagonal system.

    Parameters
    ----------
    E : ndarray, shape (nz,)
        Current enthalpy (modified in place across Newton steps).
    E_prev : ndarray, shape (nz,)
        Previous time step enthalpy.
    cfg : ColumnConfig

    Returns
    -------
    E_new : ndarray, shape (nz,)
        Updated enthalpy after n_newton Newton steps.
    info : dict
        Newton convergence history.
    """
    nz = cfg.nz
    dsig = cfg.dsig
    h = cfg.H
    h_inv = 1.0 / h
    h2_inv = h_inv * h_inv
    dt = cfg.dt
    sigma = cfg.sigma

    phi = cfg.phi_strain if cfg.phi_strain is not None else np.zeros(nz)
    horiz = cfg.horiz_source if cfg.horiz_source is not None else np.zeros(nz)
    horiz_jac = cfg.horiz_source_jac if cfg.horiz_source_jac is not None else np.zeros(nz)
    sd = cfg.sigma_dot if cfg.sigma_dot is not None else np.zeros(nz)

    E_local = E.copy()
    E_s = cfg.E_surface

    newton_residuals = []

    for newton in range(cfg.n_newton):
        a = np.zeros(nz)
        b = np.zeros(nz)
        c = np.zeros(nz)
        rhs = np.zeros(nz)

        # ---- Surface: Dirichlet ----
        b[nz - 1] = 1.0
        rhs[nz - 1] = -(E_local[nz - 1] - E_s)

        # ---- Bed: k = 0 ----
        E_k = E_local[0]
        E_kp1 = E_local[1]
        E_pmp_0 = get_E_pmp(0.0, h)
        E_pmp_1 = get_E_pmp(dsig, h)

        # Residual
        rk = RHO_I * (E_k - E_prev[0]) / dt
        rk += horiz[0] + horiz_jac[0] * (E_k - E[0])

        # Bed sigma advection
        sd_k = sd[0]
        adv_d_E_k = 0.0
        adv_d_E_kp1 = 0.0
        if sd_k < 0.0:
            rk += RHO_I * sd_k * (E_kp1 - E_k) / dsig
            adv_d_E_k = -RHO_I * sd_k / dsig
            adv_d_E_kp1 = RHO_I * sd_k / dsig

        # Bed diffusion
        K_half = get_K(0.5 * (E_k + E_kp1), 0.5 * (E_pmp_0 + E_pmp_1))
        dsig_half = 0.5 * dsig
        rk += (-h2_inv * K_half * (E_kp1 - E_k) / dsig
                - (cfg.Q_geo + cfg.Q_fh) * h_inv) / dsig_half

        # Frozen-K diffusion Jacobian
        diff_coeff = h2_inv * K_half / (dsig * dsig_half)
        diff_d_E_k = diff_coeff
        diff_d_E_kp1 = -diff_coeff

        # Source terms (smoothed drainage)
        rk -= phi[0]
        omega_0 = get_omega_smooth(E_k, E_pmp_0)
        rk += RHO_W * L_HEAT * cfg.drain_rate * omega_0
        drain_d_E_k = RHO_W * L_HEAT * cfg.drain_rate * get_domega_dE_smooth(E_k, E_pmp_0)

        # Assemble bed row
        b[0] = (RHO_I / dt + diff_d_E_k + adv_d_E_k
                + drain_d_E_k + horiz_jac[0])
        c[0] = diff_d_E_kp1 + adv_d_E_kp1
        rhs[0] = -rk

        # ---- Interior: k = 1 .. nz-2 ----
        for k in range(1, nz - 1):
            E_k = E_local[k]
            E_km1 = E_local[k - 1]
            E_kp1 = E_local[k + 1] if k < nz - 1 else E_s

            E_pmp_k = get_E_pmp(sigma[k], h)
            E_pmp_km1 = get_E_pmp(sigma[k - 1], h)
            E_pmp_kp1 = get_E_pmp(sigma[min(k + 1, nz - 1)], h)

            # Residual
            rk = RHO_I * (E_k - E_prev[k]) / dt
            rk += horiz[k] + horiz_jac[k] * (E_k - E[k])

            # Sigma advection (upwind)
            sd_k = sd[k]
            sd_pos = max(sd_k, 0.0)
            sd_neg = min(sd_k, 0.0)
            dsig_inv = 1.0 / dsig
            rk += RHO_I * (sd_pos * (E_k - E_km1)
                           + sd_neg * (E_kp1 - E_k)) * dsig_inv

            adv_d_km1 = -RHO_I * sd_pos * dsig_inv
            adv_d_kp1 = RHO_I * sd_neg * dsig_inv
            adv_d_k = RHO_I * (sd_pos - sd_neg) * dsig_inv

            # Interior diffusion
            K_upper = get_K(0.5 * (E_k + E_kp1),
                            0.5 * (E_pmp_k + E_pmp_kp1))
            K_lower = get_K(0.5 * (E_k + E_km1),
                            0.5 * (E_pmp_k + E_pmp_km1))
            dsig2_inv = 1.0 / (dsig * dsig)
            rk += -h2_inv * dsig2_inv * (K_upper * (E_kp1 - E_k)
                                         - K_lower * (E_k - E_km1))

            # Frozen-K diffusion Jacobian
            diff_lower = h2_inv * dsig2_inv * K_lower
            diff_upper = h2_inv * dsig2_inv * K_upper
            diff_d_km1 = -diff_lower
            diff_d_kp1 = -diff_upper
            diff_d_k = diff_lower + diff_upper

            # Source terms (smoothed drainage)
            rk -= phi[k]
            omega_k = get_omega_smooth(E_k, E_pmp_k)
            rk += RHO_W * L_HEAT * cfg.drain_rate * omega_k
            drain_d_k = RHO_W * L_HEAT * cfg.drain_rate * get_domega_dE_smooth(E_k, E_pmp_k)

            # Assemble
            a[k] = diff_d_km1 + adv_d_km1
            c[k] = diff_d_kp1 + adv_d_kp1
            b[k] = (RHO_I / dt + diff_d_k + adv_d_k
                    + drain_d_k + horiz_jac[k])
            rhs[k] = -rk

        # ---- Thomas algorithm (forward elimination) ----
        for k in range(1, nz):
            if abs(b[k - 1]) < 1e-30:
                continue
            w = a[k] / b[k - 1]
            b[k] -= w * c[k - 1]
            rhs[k] -= w * rhs[k - 1]

        # ---- Back substitution ----
        dE = np.zeros(nz)
        dE[nz - 1] = rhs[nz - 1] / b[nz - 1]
        for k in range(nz - 2, -1, -1):
            dE[k] = (rhs[k] - c[k] * dE[k + 1]) / b[k]

        # ---- Apply correction ----
        E_local += cfg.relaxation * dE
        newton_residuals.append(np.linalg.norm(dE))

    info = {'newton_corrections': newton_residuals}
    return E_local, info


# ---- Sweep driver (repeated column solves with residual tracking) ----

def column_sweep(E, E_prev, cfg: ColumnConfig, n_iter=50,
                 rtol=1e-10, atol=1e-10, report=False):
    """
    Repeated column solves with convergence checking.

    For a single isolated column this is just repeated Newton solves
    on the same system (no Jacobi iteration needed since there are no
    horizontal neighbors). This is useful for studying how the Newton
    iteration converges on different column configurations.

    Parameters
    ----------
    E : ndarray, shape (nz,)
        Initial enthalpy guess.
    E_prev : ndarray, shape (nz,)
        Previous time step enthalpy.
    cfg : ColumnConfig
    n_iter : int
        Maximum number of sweeps.
    rtol, atol : float
        Convergence tolerances on the residual norm.
    report : bool
        Print convergence info.

    Returns
    -------
    E : ndarray, shape (nz,)
        Converged enthalpy.
    history : dict
        Convergence history with residual norms.
    """
    E = E.copy()
    residual_norms = []

    r0 = np.linalg.norm(compute_residual(E, E_prev, cfg))
    if report:
        print(f"  Initial: |r0| = {r0:.6e}")
    residual_norms.append(r0)

    for iteration in range(n_iter):
        E, newton_info = column_solve(E, E_prev, cfg)

        r = np.linalg.norm(compute_residual(E, E_prev, cfg))
        residual_norms.append(r)
        rel = r / r0 if r0 > 0 else 0.0

        if report:
            print(f"  Sweep {iteration}: |r|/|r0| = {rel:.6e}, |r| = {r:.6e}")

        if rel < rtol or r < atol:
            break

    history = {
        'residual_norms': np.array(residual_norms),
        'n_sweeps': iteration + 1,
        'converged': (r / r0 < rtol if r0 > 0 else True) or r < atol,
    }
    return E, history


# ---- Time stepping ----

def time_step(E, cfg: ColumnConfig, dt_years, n_steps=1,
              n_iter=20, rtol=1e-10, atol=1e-10,
              report=False, save_profiles=False):
    """
    Advance the column enthalpy forward in time.

    Parameters
    ----------
    E : ndarray, shape (nz,)
        Initial enthalpy.
    cfg : ColumnConfig
        Column configuration. The dt field is overridden by dt_years.
    dt_years : float
        Time step size in years.
    n_steps : int
        Number of time steps to take.
    n_iter : int
        Max sweeps per time step.
    rtol, atol : float
        Convergence tolerances per time step.
    report : bool
        Print per-step info.
    save_profiles : bool
        If True, store the enthalpy profile at every time step
        in the returned history as 'E_history', shape (n_steps+1, nz).

    Returns
    -------
    E : ndarray, shape (nz,)
        Enthalpy after n_steps time steps.
    time_history : dict
        Time in years and residual norms at each step.
        If save_profiles=True, also contains 'E_history'.
    """
    cfg_step = cfg.with_dt_years(dt_years)
    E = E.copy()

    times = [0.0]
    residuals = []
    if save_profiles:
        E_history = [E.copy()]

    for step in range(n_steps):
        E_prev = E.copy()
        E, hist = column_sweep(E, E_prev, cfg_step,
                               n_iter=n_iter, rtol=rtol, atol=atol)
        times.append(times[-1] + dt_years)
        residuals.append(hist['residual_norms'][-1])
        if save_profiles:
            E_history.append(E.copy())

        if report and (step % max(1, n_steps // 20) == 0 or step == n_steps - 1):
            print(f"  Step {step}: t = {times[-1]:.1f} yr, "
                  f"|r| = {residuals[-1]:.2e}, "
                  f"sweeps = {hist['n_sweeps']}")

    result = {
        'times': np.array(times),
        'residuals': np.array(residuals),
    }
    if save_profiles:
        result['E_history'] = np.array(E_history)  # (n_steps+1, nz)
    return E, result


# ---- Convenience initializers ----

def initialize_linear_profile(cfg: ColumnConfig):
    """Initialize with the steady-state linear (pure diffusion) profile."""
    sigma = cfg.sigma
    E_s = cfg.E_surface
    # T(sigma) = T_s + Q_geo*H/k_i * (1 - sigma)
    T_profile = cfg.T_surface + (cfg.Q_geo * cfg.H / K_I) * (1.0 - sigma)
    # Cap at pressure melting point
    for k in range(cfg.nz):
        T_pmp = T_MELT - BETA_CC * RHO_I * GRAVITY * (1.0 - sigma[k]) * cfg.H
        T_profile[k] = min(T_profile[k], T_pmp)
    return enthalpy_from_temperature(T_profile)


def initialize_uniform_temperature(T, cfg: ColumnConfig):
    """Initialize with a uniform temperature."""
    sigma = cfg.sigma
    T_arr = np.full(cfg.nz, T)
    for k in range(cfg.nz):
        T_pmp = T_MELT - BETA_CC * RHO_I * GRAVITY * (1.0 - sigma[k]) * cfg.H
        T_arr[k] = min(T_arr[k], T_pmp)
    return enthalpy_from_temperature(T_arr)


def initialize_from_surface(cfg: ColumnConfig):
    """Initialize the entire column to the surface enthalpy."""
    return np.full(cfg.nz, cfg.E_surface)
