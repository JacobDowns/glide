"""
Stefan reference example: cold-to-temperate basal transition.

A flat ice slab with high geothermal heat flux can develop a temperate
basal layer. The reference profile plotted here is a steady 1D enthalpy
solution consistent with the model's cold/temperate diffusivities in the
sharp-transition limit:

  Cold zone (sigma* < sigma < 1):
    E varies linearly with slope -Q_geo H / K_cold.
    The CTS is where E first reaches E_pmp(sigma).

  Temperate zone (sigma < sigma*):
    E continues increasing downward with slope -Q_geo H / K_temp.
    Temperature is clipped to T_pmp(sigma), and the excess enthalpy
    above E_pmp becomes water content.

This is the quantity that should be compared against the modeled steady
enthalpy. Unlike temperature, enthalpy is not capped at E_pmp in the
temperate layer.
"""
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from glide.grid import Grid
from glide.enthalpy import (
    EnthalpyOperators, C_I, K_I, T_REF, T_MELT, RHO_I, GRAVITY, BETA_CC,
    L_HEAT, K_COLD, K_TEMP
)

# ========================================================
# Parameters
# ========================================================
ny, nx = 4, 4
dx = 1000.0
H_ice = 1000.0
T_surface = 223.15     # K (-50 C)
Q_geo = 1.0         # W/m^2 (high — triggers temperate base)
nz = 64
dt = 1000.0 * 365.25 * 86400.0
n_smooth = 20
snapshot_times_kyr = [0, 2, 10, 30, 100, 200]

# ========================================================
# Grid and operator setup
# ========================================================
grid = Grid(ny, nx, cp.float32(dx))
grid.state.H.set(cp.full((ny, nx), H_ice, dtype=cp.float32))
grid.state.H_prev.set(cp.full((ny, nx), H_ice, dtype=cp.float32))
grid.geometry.bed.set(cp.zeros((ny, nx), dtype=cp.float32))
grid.forcing.smb.set(cp.zeros((ny, nx), dtype=cp.float32))
grid.sliding.beta.set(cp.ones((ny, nx), dtype=cp.float32))
grid.sliding.m.set(1.0)
grid.sliding.u_reg.set(1.0)

ops = EnthalpyOperators(grid, nz=nz, sigma_q=1.0)
ops.initialize_from_temperature(T_surface)
ops.set_surface_enthalpy_from_temperature(
    cp.full((ny, nx), T_surface, dtype=cp.float32))
ops.enthalpy_forcing.Q_geo[:, :] = Q_geo
ops.enthalpy_forcing.phi_strain.fill(0)
ops.enthalpy_forcing.drain_rate.set(0.0)
ops.enthalpy_velocity.u3d.fill(0)
ops.enthalpy_velocity.v3d.fill(0)
ops.enthalpy_velocity.sigma_dot.fill(0)
ops.Q_fh.fill(0)

# ========================================================
# Semi-analytical enthalpy reference profile
# ========================================================
sigma = cp.asnumpy(ops.sigma)
depth = (1.0 - sigma) * H_ice

def T_pmp(sig):
    """Pressure melting point at sigma level."""
    d = (1 - sig) * H_ice
    return T_MELT - BETA_CC * RHO_I * GRAVITY * d

def E_pmp(sig):
    """Enthalpy at pressure melting point at sigma level."""
    return C_I * (T_pmp(sig) - T_REF)

T_analytical = np.zeros_like(sigma)
E_analytical = np.zeros_like(sigma)
omega_analytical = np.zeros_like(sigma)

T_pmp_profile = np.array([T_pmp(s) for s in sigma])
E_pmp_profile = np.array([E_pmp(s) for s in sigma])

T_pmp_bed = T_pmp(0.0)
E_pmp_bed = E_pmp(0.0)
Q_crit = K_I * (T_pmp_bed - T_surface) / H_ice

if Q_geo <= Q_crit:
    regime = 'cold bed'
    E_analytical[:] = C_I * (T_surface - T_REF) + Q_geo * H_ice / K_COLD * (1.0 - sigma)
    sigma_star = None
else:
    regime = 'temperate basal layer expected'

    def cts_residual(sig_star):
        return Q_geo * H_ice * (1.0 - sig_star) / K_I - (T_pmp(sig_star) - T_surface)

    sigma_star = brentq(cts_residual, 0.0, 1.0 - 1e-10)
    E_cts = E_pmp(sigma_star)

    for k, sig in enumerate(sigma):
        if sig <= sigma_star:
            E_analytical[k] = E_cts + Q_geo * H_ice / K_TEMP * (sigma_star - sig)
        else:
            E_analytical[k] = C_I * (T_surface - T_REF) + Q_geo * H_ice / K_COLD * (1.0 - sig)

omega_analytical[:] = np.maximum(E_analytical - E_pmp_profile, 0.0) / L_HEAT
T_analytical[:] = np.minimum(E_analytical / C_I + T_REF, T_pmp_profile)

print(f"  Regime:           {regime}")
print(f"  T_pmp at bed:     {T_pmp_bed:.2f} K")
print(f"  T_surface:        {T_surface:.2f} K")
print(f"  Q_geo:            {Q_geo:.3f} W/m^2")
print(f"  Q_crit:           {Q_crit:.3f} W/m^2")
print(f"  K_cold:           {K_COLD:.3e} m^2/s")
print(f"  K_temp:           {K_TEMP:.3e} m^2/s")
if sigma_star is not None:
    print(f"  Reference CTS:    sigma* = {sigma_star:.4f}")
    print(f"  Reference omega_bed: {omega_analytical[0]:.3e}")

# ========================================================
# Time stepping with snapshots
# ========================================================
i_col, j_col = ny // 2, nx // 2
dt_kyr = dt / (1000.0 * 365.25 * 86400.0)

snapshots_E = {}
snapshots_T = {}
snapshots_omega = {}
step = 0
time_kyr = 0.0

E_init = cp.asnumpy(ops.enthalpy_state.E[i_col, j_col, :])
T_init = cp.asnumpy(ops.get_temperature()[i_col, j_col, :])
omega_init = cp.asnumpy(ops.get_water_content()[i_col, j_col, :])
snapshots_E[0.0] = E_init.copy()
snapshots_T[0.0] = T_init.copy()
snapshots_omega[0.0] = omega_init.copy()

for target_kyr in snapshot_times_kyr[1:]:
    while time_kyr < target_kyr:
        ops.set_rhs(dt)
        ops.column_sweep(dt, n_smooth)
        step += 1
        time_kyr += dt_kyr
    E_snap = cp.asnumpy(ops.enthalpy_state.E[i_col, j_col, :])
    T_snap = cp.asnumpy(ops.get_temperature()[i_col, j_col, :])
    omega_snap = cp.asnumpy(ops.get_water_content()[i_col, j_col, :])
    snapshots_E[target_kyr] = E_snap.copy()
    snapshots_T[target_kyr] = T_snap.copy()
    snapshots_omega[target_kyr] = omega_snap.copy()
    print(f"  t = {target_kyr:6.0f} kyr, T_bed = {snapshots_T[target_kyr][0]:.2f} K")

# ========================================================
# Plot: 3x2 (temperature / enthalpy / water content) x (profiles / error)
# sigma on y-axis: 0 = bed (bottom), 1 = surface (top)
# ========================================================
fig, axs = plt.subplots(3, 2, figsize=(12, 14), sharey=True)
(ax_T, ax_T_err), (ax_E, ax_E_err), (ax_W, ax_W_err) = axs

cmap = plt.cm.coolwarm
colors = cmap(np.linspace(0.1, 0.9, len(snapshots_T)))

# --- Temperature profiles ---
for (t_kyr, T_profile), color in zip(snapshots_T.items(), colors):
    ax_T.plot(T_profile, sigma, color=color, linewidth=1.5,
              label=f't = {t_kyr:.0f} kyr')
ax_T.plot(T_analytical, sigma, 'k--', linewidth=2, label='Stefan reference')
ax_T.plot(T_pmp_profile, sigma, 'r:', linewidth=1.5, alpha=0.7,
          label=r'$T_{\mathrm{pmp}}(\sigma)$')
if sigma_star is not None:
    ax_T.axhline(sigma_star, color='green', linestyle='--', linewidth=1.5, alpha=0.7,
                 label=rf'Reference CTS $\sigma^*$ = {sigma_star:.3f}')
ax_T.set_xlabel('Temperature (K)')
ax_T.set_ylabel(r'$\sigma$ (0 = bed, 1 = surface)')
ax_T.set_title('Temperature')
ax_T.set_xlim(T_surface - 5, T_pmp_bed + 5)
ax_T.legend(fontsize=7, loc='upper right')
ax_T.grid(True, alpha=0.3)

# --- Temperature error ---
T_final = snapshots_T[snapshot_times_kyr[-1]]
T_err = T_final - T_analytical
ax_T_err.plot(T_err, sigma, 'b-', linewidth=2)
if sigma_star is not None:
    ax_T_err.axhline(sigma_star, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
ax_T_err.set_xlabel('Temperature Error (K)')
ax_T_err.set_title(f'Temperature Error at t = {snapshot_times_kyr[-1]} kyr')
ax_T_err.axvline(0, color='k', linestyle='-', alpha=0.3)
ax_T_err.grid(True, alpha=0.3)
max_T_err = np.max(np.abs(T_err[:-1]))
ax_T_err.text(0.95, 0.05,
              f'Max error: {max_T_err:.2e} K\n'
              f'T_bed = {T_final[0]:.2f} K\n'
              f'T_pmp = {T_pmp_bed:.2f} K\n'
              f'Q_crit = {Q_crit:.3f} W/m^2',
              transform=ax_T_err.transAxes, ha='right', fontsize=10,
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# --- Enthalpy profiles ---
for (t_kyr, E_profile), color in zip(snapshots_E.items(), colors):
    ax_E.plot(E_profile / 1e3, sigma, color=color, linewidth=1.5,
              label=f't = {t_kyr:.0f} kyr')
ax_E.plot(E_analytical / 1e3, sigma, 'k--', linewidth=2,
          label='Semi-analytical reference')
ax_E.plot(E_pmp_profile / 1e3, sigma, 'r:', linewidth=1.5, alpha=0.7,
          label=r'$E_{\mathrm{pmp}}(\sigma)$')
if sigma_star is not None:
    ax_E.axhline(sigma_star, color='green', linestyle='--', linewidth=1.5, alpha=0.7,
                 label=rf'Reference CTS $\sigma^*$ = {sigma_star:.3f}')
ax_E.set_xlabel('Enthalpy (kJ/kg)')
ax_E.set_ylabel(r'$\sigma$ (0 = bed, 1 = surface)')
ax_E.set_title('Enthalpy')
ax_E.set_xscale('symlog', linthresh=10.0)
ax_E.legend(fontsize=7, loc='upper right')
ax_E.grid(True, alpha=0.3)

# --- Enthalpy error ---
E_final = snapshots_E[snapshot_times_kyr[-1]]
E_err = E_final - E_analytical
ax_E_err.plot(E_err, sigma, 'b-', linewidth=2)
if sigma_star is not None:
    ax_E_err.axhline(sigma_star, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
ax_E_err.set_xlabel('Enthalpy Error (J/kg)')
ax_E_err.set_title(f'Enthalpy Error at t = {snapshot_times_kyr[-1]} kyr')
ax_E_err.set_xscale('symlog', linthresh=1e3)
ax_E_err.axvline(0, color='k', linestyle='-', alpha=0.3)
ax_E_err.grid(True, alpha=0.3)
max_E_err = np.max(np.abs(E_err[:-1]))
ax_E_err.text(0.95, 0.05,
              f'Max error: {max_E_err:.2f} J/kg\n'
              f'E_bed = {E_final[0]/1e3:.2f} kJ/kg\n'
              f'E_pmp = {E_pmp_bed/1e3:.2f} kJ/kg',
              transform=ax_E_err.transAxes, ha='right', fontsize=10,
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# --- Water-content profiles ---
for (t_kyr, omega_profile), color in zip(snapshots_omega.items(), colors):
    ax_W.plot(omega_profile, sigma, color=color, linewidth=1.5,
              label=f't = {t_kyr:.0f} kyr')
ax_W.plot(omega_analytical, sigma, 'k--', linewidth=2,
          label='Semi-analytical reference')
if sigma_star is not None:
    ax_W.axhline(sigma_star, color='green', linestyle='--', linewidth=1.5, alpha=0.7,
                 label=rf'Reference CTS $\sigma^*$ = {sigma_star:.3f}')
ax_W.set_xlabel('Water Content (mass fraction)')
ax_W.set_ylabel(r'$\sigma$ (0 = bed, 1 = surface)')
ax_W.set_title('Water Content')
ax_W.set_xscale('symlog', linthresh=1e-6)
ax_W.legend(fontsize=7, loc='upper right')
ax_W.grid(True, alpha=0.3)

# --- Water-content error ---
omega_final = snapshots_omega[snapshot_times_kyr[-1]]
omega_err = omega_final - omega_analytical
ax_W_err.plot(omega_err, sigma, 'b-', linewidth=2)
if sigma_star is not None:
    ax_W_err.axhline(sigma_star, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
ax_W_err.set_xlabel('Water Content Error')
ax_W_err.set_title(f'Water Content Error at t = {snapshot_times_kyr[-1]} kyr')
ax_W_err.set_xscale('symlog', linthresh=1e-6)
ax_W_err.axvline(0, color='k', linestyle='-', alpha=0.3)
ax_W_err.grid(True, alpha=0.3)
max_omega_err = np.max(np.abs(omega_err[:-1]))
ax_W_err.text(0.95, 0.05,
              f'Max error: {max_omega_err:.2e}\n'
              f'omega_bed = {omega_final[0]:.3e}\n'
              f'omega_ref = {omega_analytical[0]:.3e}',
              transform=ax_W_err.transAxes, ha='right', fontsize=10,
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

fig.suptitle('Polythermal Column: Enthalpy Reference Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('examples/thermal/stefan.png', dpi=150)
plt.show()
print(f"\nSaved: examples/thermal/stefan.png")
