"""
Prescribed-velocity dome thermal experiment.

Uses a perfectly radially symmetric velocity field (no momentum solver)
to test horizontal advection in the enthalpy solver in isolation.

The velocity is prescribed as radial outflow:
    u(x,y) = V(r) * x/r,  v(x,y) = V(r) * y/r
where V(r) increases linearly from 0 at the center to V_max at the margin.

This produces a perfectly symmetric dome setup that lets us isolate
whether any temperature asymmetry comes from the enthalpy solver itself
or from the momentum solver's velocity field.

Run:
    python examples/thermal/prescribed_velocity_dome.py
"""
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from glide.grid import Grid
from glide.enthalpy import (
    EnthalpyOperators, T_MELT, BETA_CC, RHO_I, GRAVITY, C_I, T_REF
)

# ========================================================
# Parameters
# ========================================================
ny, nx = 128, 128
dx = 1000.0
nz = 11

DOME_RADIUS = 50000.0
DOME_HEIGHT = 2000.0

# Prescribed radial velocity: V(r) = V_MAX * r / R
V_MAX = 100.0  # m/yr at the margin

# Forcing
SMB_CENTER = 0.5   # m/yr
SMB_EDGE = -2.0    # m/yr
Q_GEO = 0.1       # W/m^2

# Surface temperature
T_SEA_LEVEL = 268.15
LAPSE_RATE = -6.5e-3
T_INIT = 253.15

# Time stepping
DT_YR = 5.0
N_STEPS = 500
N_SMOOTH = 20
SEC_PER_YR = 365.25 * 86400.0

OUT_DIR = Path('prescribed_velocity_dome_output')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ========================================================
# Build dome geometry
# ========================================================
x = (cp.arange(nx, dtype=cp.float32) + 0.5) * dx - 0.5 * nx * dx
y = (cp.arange(ny, dtype=cp.float32) + 0.5) * dx - 0.5 * ny * dx
xx, yy = cp.meshgrid(x, y)
radius = cp.sqrt(xx**2 + yy**2)
radial_fraction = cp.clip(radius / DOME_RADIUS, 0.0, 1.0)
hemisphere = cp.sqrt(cp.clip(1.0 - radial_fraction**2, 0.0, 1.0))

thickness = DOME_HEIGHT * hemisphere
bed = cp.zeros((ny, nx), dtype=cp.float32)
smb = cp.float32(SMB_EDGE) + cp.float32(SMB_CENTER - SMB_EDGE) * hemisphere


def surface_temperature(H):
    surface_elev = H  # flat bed
    return cp.minimum(
        cp.float32(T_SEA_LEVEL) + cp.float32(LAPSE_RATE) * surface_elev,
        cp.float32(T_MELT))


# ========================================================
# Build prescribed radially symmetric velocity on MAC grid
# ========================================================
def build_radial_velocity(H):
    """
    Prescribed radial outflow: V(r) = V_MAX * r / R.
    u and v on MAC facets, uniform in sigma (SSA-like).
    Returns u3d (nz, ny, nx+1) and v3d (nz, ny+1, nx) in m/s.
    """
    v_max_si = V_MAX / SEC_PER_YR  # m/s

    # u on vertical facets: x-positions at j+1/2
    x_u = (cp.arange(nx + 1, dtype=cp.float32)) * dx - 0.5 * nx * dx
    y_u = (cp.arange(ny, dtype=cp.float32) + 0.5) * dx - 0.5 * ny * dx
    xx_u, yy_u = cp.meshgrid(x_u, y_u)
    r_u = cp.sqrt(xx_u**2 + yy_u**2)
    r_u = cp.maximum(r_u, 1.0)  # avoid division by zero
    V_r_u = v_max_si * cp.minimum(r_u / DOME_RADIUS, cp.float32(1.0))
    u_2d = V_r_u * xx_u / r_u  # u = V(r) * x/r

    # v on horizontal facets: y-positions at i+1/2
    x_v = (cp.arange(nx, dtype=cp.float32) + 0.5) * dx - 0.5 * nx * dx
    y_v = (cp.arange(ny + 1, dtype=cp.float32)) * dx - 0.5 * ny * dx
    xx_v, yy_v = cp.meshgrid(x_v, y_v)
    r_v = cp.sqrt(xx_v**2 + yy_v**2)
    r_v = cp.maximum(r_v, 1.0)
    V_r_v = v_max_si * cp.minimum(r_v / DOME_RADIUS, cp.float32(1.0))
    v_2d = V_r_v * yy_v / r_v  # v = V(r) * y/r

    # Broadcast to all sigma layers
    u3d = cp.zeros((nz, ny, nx + 1), dtype=cp.float32)
    v3d = cp.zeros((nz, ny + 1, nx), dtype=cp.float32)
    for k in range(nz):
        u3d[k, :, :] = u_2d
        v3d[k, :, :] = v_2d

    return u3d, v3d


# ========================================================
# Initialize enthalpy solver (no momentum solver)
# ========================================================
grid = Grid(ny, nx, cp.float32(dx))
grid.state.H.set(thickness)
grid.state.H_prev.set(thickness)
grid.geometry.bed.set(bed)
grid.forcing.smb.set(smb)
grid.sliding.beta.set(cp.ones((ny, nx), dtype=cp.float32))
grid.sliding.m.set(1.0)
grid.sliding.u_reg.set(1.0)

ops = EnthalpyOperators(grid, nz=nz)
ops.smoother_config.report_norms = True
ops.smoother_config.n_newton = 5
ops.smoother_config.lf_c = cp.float32(1e-4)
ops.enthalpy_forcing.h_thin.set(50.0)

T_surf = surface_temperature(thickness)
ops.initialize_from_temperature(T_INIT)
ops.set_surface_enthalpy_from_temperature(T_surf)
ops.enthalpy_forcing.Q_geo[:, :] = Q_GEO
ops.enthalpy_forcing.Q_fh.fill(0)
ops.enthalpy_forcing.phi_strain.fill(0)

# Set prescribed velocity
u3d, v3d = build_radial_velocity(thickness)
ops.enthalpy_velocity.u3d[:] = u3d
ops.enthalpy_velocity.v3d[:] = v3d

# Compute omega from continuity (uses the prescribed velocity)
smb_si = smb / cp.float32(SEC_PER_YR)
ops.compute_omega(smb_si)

print(f"Prescribed velocity dome:")
print(f"  Grid: {ny}x{nx}, nz={nz}, dx={dx:.0f} m")
print(f"  V_max = {V_MAX} m/yr")
print(f"  T_surface: center = {float(T_surf[ny//2, nx//2]):.1f} K, "
      f"margin = {T_SEA_LEVEL:.1f} K")
print(f"  Term flags: 0x{ops.term_flags.bitmask:x}")

# Check velocity symmetry
u_np = cp.asnumpy(u3d[0])
v_np = cp.asnumpy(v3d[0])
# u should be antisymmetric in x: u(x,y) = -u(-x,y)
u_asym = np.max(np.abs(u_np + u_np[:, ::-1]))
# v should be antisymmetric in y: v(x,y) = -v(x,-y)
v_asym = np.max(np.abs(v_np + v_np[::-1, :]))
print(f"  Velocity antisymmetry: u={u_asym:.2e}, v={v_asym:.2e}")

# ========================================================
# Time stepping (no momentum solver — static geometry)
# ========================================================
dt_sec = DT_YR * SEC_PER_YR
t_yr = 0.0

times = [0.0]
T_bed_center = []
T_bed_margin = []

sigma = cp.asnumpy(ops.sigma)
i_c, j_c = ny // 2, nx // 2
j_m = nx // 2 + int(0.8 * DOME_RADIUS / dx)
j_m = min(j_m, nx - 1)

snapshot_steps = [0, N_STEPS // 4, N_STEPS // 2, N_STEPS]
snapshots = {}

# Save initial
T_3d = cp.asnumpy(ops.get_temperature())
snapshots[0] = {'t_yr': 0.0, 'T_3d': T_3d.copy(),
                'H': cp.asnumpy(thickness)}

print(f"\n{'step':>5s}  {'t (yr)':>8s}  {'T_bed ctr':>10s}  {'T_bed mrg':>10s}  "
      f"{'T_asym':>10s}  {'|r|':>10s}")
print("-" * 62)

for step in range(1, N_STEPS + 1):
    # Static geometry: H doesn't change, so H_prev = H always.
    # (No momentum solver — we're testing enthalpy in isolation.)
    ops.set_rhs(dt_sec)
    ops.column_sweep(dt_sec, N_SMOOTH)

    t_yr += DT_YR
    times.append(t_yr)

    # Diagnostics
    T_3d = cp.asnumpy(ops.get_temperature())
    T_mid = T_3d[:, :, nz // 2]

    T_bc = T_3d[i_c, j_c, 0]
    T_bm = T_3d[i_c, j_m, 0]
    T_bed_center.append(T_bc)
    T_bed_margin.append(T_bm)

    # Symmetry: 180-degree rotation
    T_rot = T_mid[::-1, ::-1]
    T_asym = np.max(np.abs(T_mid - T_rot))

    # Also check x-reflection symmetry
    T_xflip = T_mid[:, ::-1]
    T_x_asym = np.max(np.abs(T_mid - T_xflip))

    # y-reflection symmetry
    T_yflip = T_mid[::-1, :]
    T_y_asym = np.max(np.abs(T_mid - T_yflip))

    r = float(cp.max(cp.abs(ops.r_E)))

    if step % 20 == 0 or step == 1:
        print(f"{step:5d}  {t_yr:8.1f}  {T_bc:10.2f}  {T_bm:10.2f}  "
              f"{T_asym:10.2e}  {r:10.2e}  "
              f"x_asym={T_x_asym:.2e}  y_asym={T_y_asym:.2e}")

    if step in snapshot_steps:
        snapshots[step] = {'t_yr': t_yr, 'T_3d': T_3d.copy(),
                           'H': cp.asnumpy(grid.state.H.data)}

print(f"\nDone. {N_STEPS} steps, final t = {t_yr:.0f} yr")

# ========================================================
# Plots
# ========================================================
x_km = cp.asnumpy(x) / 1000.0
y_km = cp.asnumpy(y) / 1000.0

# 1. Time series
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(times[1:], T_bed_center, 'r-', label='Center')
ax1.plot(times[1:], T_bed_margin, 'b-', label='Margin')
ax1.set_xlabel('Time (yr)')
ax1.set_ylabel('Basal temperature (K)')
ax1.set_title('Basal temperature evolution')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Final basal T map
H_np = cp.asnumpy(grid.state.H.data)
T_bed_2d = T_3d[:, :, 0].copy()
T_bed_2d[H_np < 50] = np.nan
im = ax2.pcolormesh(x_km, y_km, T_bed_2d, cmap='coolwarm', shading='auto')
ax2.set_xlabel('x (km)')
ax2.set_ylabel('y (km)')
ax2.set_title(f'Basal T at t = {t_yr:.0f} yr')
ax2.set_aspect('equal')
plt.colorbar(im, ax=ax2, label='T (K)')

plt.tight_layout()
plt.savefig(OUT_DIR / 'time_series.png', dpi=150)
print(f"Saved {OUT_DIR / 'time_series.png'}")

# 2. Cross-sections at snapshots
snap_keys = sorted(snapshots.keys())
fig2, axes2 = plt.subplots(len(snap_keys), 1,
                            figsize=(12, 3.5 * len(snap_keys)), sharex=True)
if len(snap_keys) == 1:
    axes2 = [axes2]

T_all = []
for snap in snapshots.values():
    T_row = snap['T_3d'][i_c, :, :]
    H_row = snap['H'][i_c, :]
    for j in range(nx):
        if H_row[j] >= 50:
            T_all.extend(T_row[j, :].tolist())
T_vmin, T_vmax = min(T_all), max(T_all)

for idx, step_key in enumerate(snap_keys):
    ax = axes2[idx]
    snap = snapshots[step_key]
    T_row = snap['T_3d'][i_c, :, :]
    H_row = snap['H'][i_c, :]
    T_masked = T_row.copy()
    for j in range(nx):
        if H_row[j] < 50:
            T_masked[j, :] = np.nan
    im = ax.pcolormesh(x_km, sigma, T_masked.T,
                       cmap='RdYlBu_r', shading='auto',
                       vmin=T_vmin, vmax=T_vmax)
    plt.colorbar(im, ax=ax, label='T (K)')
    ax.set_ylabel(r'$\sigma$')
    ax.set_title(f't = {snap["t_yr"]:.0f} yr')

axes2[-1].set_xlabel('x (km)')
fig2.suptitle('Temperature Cross-Section (y=0), Prescribed Radial Velocity',
              fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / 'cross_sections.png', dpi=150)
print(f"Saved {OUT_DIR / 'cross_sections.png'}")

# 3. Symmetry comparison: T along x-axis vs y-axis
fig3, ax3 = plt.subplots(1, 1, figsize=(8, 5))
T_bed_xaxis = T_3d[i_c, :, 0]
T_bed_yaxis = T_3d[:, j_c, 0]
H_xaxis = H_np[i_c, :]
H_yaxis = H_np[:, j_c]
# Mask thin ice
T_bed_xaxis[H_xaxis < 50] = np.nan
T_bed_yaxis[H_yaxis < 50] = np.nan

ax3.plot(x_km, T_bed_xaxis, 'r-', label='Along x-axis (y=0)')
ax3.plot(cp.asnumpy(y) / 1000.0, T_bed_yaxis, 'b--', label='Along y-axis (x=0)')
ax3.set_xlabel('Distance from center (km)')
ax3.set_ylabel('Basal temperature (K)')
ax3.set_title('Basal T: x-axis vs y-axis (should overlap for radial symmetry)')
ax3.legend()
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / 'symmetry_check.png', dpi=150)
print(f"Saved {OUT_DIR / 'symmetry_check.png'}")

plt.show()
