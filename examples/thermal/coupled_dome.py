"""
Coupled momentum-thermal dome example.

A hemispherical ice dome on a flat bed evolves under its own weight
with two-way coupling between the SSA momentum solver and the enthalpy
solver:

  Momentum  -->  velocity u, v  -->  Enthalpy advection + frictional heating
  Enthalpy  -->  Arrhenius B(T) -->  Momentum viscosity

The dome starts with a uniform cold temperature and warms from below
via geothermal heat flux. As the base warms, the ice softens (B decreases),
which accelerates flow, which in turn increases frictional heating and
advective heat transport.

Outputs:
  - VTI files for ParaView (thickness, velocity, basal temperature, B)
  - Summary plot of dome evolution
"""
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from glide.model import IceDynamics, ThermalModel
try:
    from glide.io import VTIWriter
    HAS_VTI = True
except ImportError:
    HAS_VTI = False
from glide.enthalpy import T_MELT, BETA_CC, RHO_I, GRAVITY

# ========================================================
# Parameters
# ========================================================
# Grid
n_levels = 4
ny, nx = 128, 128
dx = 2500.0           # m

# Dome geometry
DOME_RADIUS = 100000.0 # m
DOME_HEIGHT = 3000.0   # m

# Forcing
SMB_CENTER = 5.       # m/yr ice equivalent (accumulation at center)
SMB_EDGE = -5.0        # m/yr ice equivalent (ablation at margin)
Q_GEO = 0.0           # W/m^2 (geothermal heat flux)
T_SURFACE = 248.15     # K (-25 C)

# Rheology
# GLIDE works in "head" units: B_head = B_SI / (rho_i * g)
RHO_ICE = 917.0
G = 9.81
N_GLEN = 3.0

# Sliding (head units, comparable to Greenland example values)
BETA_SLIDING = 1.0

# Thermal
NZ = 21                # sigma levels
SIGMA_Q = 2.0          # bunching toward bed
N_SMOOTH = 10          # enthalpy smoothing sweeps

# Time stepping
DT_YR = 10.0           # years
N_STEPS = 100
SEC_PER_YR = 365.25 * 86400.0

# Output
OUT_DIR = Path('coupled_dome_output')

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
smb = (cp.float32(SMB_EDGE)
       + cp.float32(SMB_CENTER - SMB_EDGE) * hemisphere)

# ========================================================
# Initialize momentum solver
# ========================================================
model = IceDynamics(n_levels=n_levels, ny=ny, nx=nx, dx=cp.float32(dx))
mg = model.mg

mg.geometry.bed.set(bed)
mg.state.H.set(thickness)
mg.state.H_prev.set(thickness)
mg.forcing.smb.set(smb)

# B will be set from the thermal model after initialization below
mg.rheology.B.set(cp.ones((ny, nx), dtype=cp.float32))  # placeholder
mg.rheology.n.set(N_GLEN)
mg.rheology.eps_reg.set(1e-6)

beta_field = cp.full((ny, nx), BETA_SLIDING, dtype=cp.float32)
mg.sliding.beta.set(beta_field)
mg.sliding.m.set(1.0 / N_GLEN)
mg.sliding.u_reg.set(1.0)
mg.calving.calving_rate.set(0.0)

# Solver configuration
model.forward_solver.fas_options.set(
    coarsest_steps=200,
    pre_steps=5,
    post_steps=20,
    finest_steps=0,
    relative_tolerance=5e-3,
    absolute_tolerance=1.0,
    report_norms=False,
)

# ========================================================
# Initialize thermal solver
# ========================================================
grid = mg.levels[0]
thermal = ThermalModel(grid, nz=NZ, sigma_q=SIGMA_Q,
                       n_smooth=N_SMOOTH, update_rheology=True)

thermal.ops.smoother_config.report_norms = True
thermal.initialize(T_surface=T_SURFACE, Q_geo=Q_GEO)

# Set initial B from the Paterson-Budd law at the initial temperature,
# so there is no discontinuity when the thermal model starts updating B.
B_init = thermal.ops.get_arrhenius_factor() / thermal.B_scale
mg.rheology.B.set(B_init)
print(f"  Initial B (GLIDE units): {float(B_init[ny//2, nx//2]):.4e}")

# ========================================================
# Output setup
# ========================================================
OUT_DIR.mkdir(parents=True, exist_ok=True)

if HAS_VTI:
    origin = (float(x[0]), float(y[0]))
    vti_writer = VTIWriter(
        out_dir=str(OUT_DIR / 'vti'),
        base='dome',
        dx=dx,
        origin=origin,
    )
    (OUT_DIR / 'vti').mkdir(parents=True, exist_ok=True)

# ========================================================
# Time stepping
# Note: SSA solver uses dt in years (matching smb in m/yr),
#       enthalpy solver uses dt in seconds (SI units).
# ========================================================
dt_yr = cp.float32(DT_YR)
dt_sec = DT_YR * SEC_PER_YR
t_yr = 0.0

# Spin up the momentum solver for a few steps before thermal coupling.
# The initial hemisphere dome is very steep, producing extreme first-step
# velocities that can cause slow convergence in the enthalpy smoother.
# A short momentum-only spin-up lets the geometry relax.
N_SPINUP = 5
print(f"  Momentum spin-up ({N_SPINUP} steps, no thermal coupling)...")
for _ in range(N_SPINUP):
    model.forward(cp.float32(t_yr), dt_yr)
    t_yr += DT_YR
print(f"    done (t = {t_yr:.0f} yr)")

# Re-initialize thermal state from the relaxed geometry
thermal.initialize(T_surface=T_SURFACE, Q_geo=Q_GEO)
B_init = thermal.ops.get_arrhenius_factor() / thermal.B_scale
mg.rheology.B.set(B_init)

# Storage for time-series plot
times = [t_yr]
vol = [float(cp.sum(grid.state.H.data) * dx**2)]
T_bed_center = [float(cp.asnumpy(thermal.temperature[ny//2, nx//2, 0]))]
B_center = [float(cp.asnumpy(grid.rheology.B.data[ny//2, nx//2]))]
max_speed = [0.0]

# Snapshots for spatial map plots (at selected steps)
snapshot_steps = {1, N_STEPS // 4, N_STEPS // 2, N_STEPS}
snapshots_map = {}  # step -> dict of 2D fields

print(f"\n  {'step':>5s}  {'t (yr)':>8s}  {'vol (km3)':>10s}  "
      f"{'T_bed (K)':>10s}  {'B_center':>12s}  {'max |u| (m/yr)':>14s}")
print(f"  {'-'*68}")

for step in range(N_STEPS):
    # --- Momentum solve (dt in years) ---
    model.forward(cp.float32(t_yr), dt_yr)

    # --- Thermal solve (dt in seconds) ---
    thermal.step(dt_sec)

    t_yr += DT_YR

    # --- Diagnostics ---
    u = cp.asnumpy(grid.state.u.data)
    v = cp.asnumpy(grid.state.v.data)
    u_c = 0.5 * (u[:, 1:] + u[:, :-1])
    v_c = 0.5 * (v[1:, :] + v[:-1, :])
    speed = np.sqrt(u_c**2 + v_c**2)  # already m/yr (SSA uses year-based units)

    H_np = cp.asnumpy(grid.state.H.data)
    volume_km3 = float(np.sum(H_np) * dx**2 / 1e9)
    T_bed = float(cp.asnumpy(thermal.temperature[ny//2, nx//2, 0]))
    B_val = float(cp.asnumpy(grid.rheology.B.data[ny//2, nx//2]))
    max_spd = float(np.max(speed))

    times.append(t_yr)
    vol.append(volume_km3)
    T_bed_center.append(T_bed)
    B_center.append(B_val)
    max_speed.append(max_spd)

    if (step + 1) % 10 == 0 or step == 0:
        print(f"  {step+1:5d}  {t_yr:8.1f}  {volume_km3:10.1f}  "
              f"{T_bed:10.2f}  {B_val:12.4e}  {max_spd:14.2f}")

    # --- Store spatial snapshot ---
    if (step + 1) in snapshot_steps:
        T_3d = cp.asnumpy(thermal.temperature)
        # Thin marginal ice has unreliable thermal state — clamp it
        thin = H_np < 100.0
        T_bed = np.clip(T_3d[:, :, 0], T_SURFACE - 5.0, T_MELT)
        T_avg = np.clip(np.mean(T_3d, axis=2), T_SURFACE - 5.0, T_MELT)
        T_bed[thin] = T_SURFACE
        T_avg[thin] = T_SURFACE
        snapshots_map[step + 1] = {
            't_yr': t_yr,
            'T_bed': T_bed,
            'T_avg': T_avg,
            'B': cp.asnumpy(grid.rheology.B.data).copy(),
            'H': H_np.copy(),
            'speed': speed.copy(),
        }

    # --- Write VTI snapshot ---
    if (step + 1) % 10 == 0 and HAS_VTI:
        T_bed_2d = cp.asnumpy(thermal.temperature[:, :, 0])
        B_2d = cp.asnumpy(grid.rheology.B.data)
        snapshot = {
            'H': H_np,
            'speed': speed,
            'T_bed': T_bed_2d,
            'B': B_2d,
        }
        vti_writer.write_step(step + 1, t_yr, snapshot)
        vti_writer.write_pvd()

print(f"\n  Done. {N_STEPS} steps, final t = {t_yr:.0f} yr")

# ========================================================
# Summary plot
# ========================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# Volume
axes[0, 0].plot(times, vol, 'b-', linewidth=2)
axes[0, 0].set_ylabel('Volume (km$^3$)')
axes[0, 0].set_title('Ice Volume')
axes[0, 0].grid(True, alpha=0.3)

# Basal temperature at dome center
T_pmp_center = T_MELT - BETA_CC * RHO_I * GRAVITY * DOME_HEIGHT
axes[0, 1].plot(times, T_bed_center, 'r-', linewidth=2)
axes[0, 1].axhline(T_pmp_center, color='k', linestyle='--', alpha=0.5,
                    label=f'$T_{{pmp}}$ = {T_pmp_center:.1f} K')
axes[0, 1].set_ylabel('Temperature (K)')
axes[0, 1].set_title('Basal Temperature (dome center)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Rate factor B at dome center
axes[1, 0].plot(times, B_center, 'g-', linewidth=2)
axes[1, 0].set_ylabel('B (Pa$^{-n}$ s$^{-1}$)')
axes[1, 0].set_xlabel('Time (yr)')
axes[1, 0].set_title('Arrhenius B (dome center)')
axes[1, 0].grid(True, alpha=0.3)

# Max speed
axes[1, 1].plot(times, max_speed, 'm-', linewidth=2)
axes[1, 1].set_ylabel('Speed (m/yr)')
axes[1, 1].set_xlabel('Time (yr)')
axes[1, 1].set_title('Max Surface Speed')
axes[1, 1].grid(True, alpha=0.3)

fig.suptitle('Coupled Momentum-Thermal Dome Evolution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / 'coupled_dome_summary.png', dpi=150)
plt.show()
print(f"  Saved: {OUT_DIR / 'coupled_dome_summary.png'}")

# ========================================================
# Spatial maps at selected time steps
# Rows: T_bed, T_avg, B, speed
# Columns: one per snapshot time
# ========================================================
x_km = cp.asnumpy(x) / 1000.0
y_km = cp.asnumpy(y) / 1000.0
snap_keys = sorted(snapshots_map.keys())
n_snaps = len(snap_keys)

# Compute global color ranges from ice-covered cells across all snapshots
all_T_bed, all_T_avg, all_B, all_spd = [], [], [], []
for snap in snapshots_map.values():
    ice = snap['H'] >= 100.0
    all_T_bed.append(snap['T_bed'][ice])
    all_T_avg.append(snap['T_avg'][ice])
    all_B.append(snap['B'][ice])
    all_spd.append(snap['speed'][ice])

field_specs = [
    ('T_bed',  'Basal Temperature (K)',          'coolwarm',
     np.min(np.concatenate(all_T_bed)), np.max(np.concatenate(all_T_bed))),
    ('T_avg',  'Column-Avg Temperature (K)',     'coolwarm',
     np.min(np.concatenate(all_T_avg)), np.max(np.concatenate(all_T_avg))),
    ('B',      'Arrhenius B (GLIDE units)',       'viridis',
     np.min(np.concatenate(all_B)),     np.max(np.concatenate(all_B))),
    ('speed',  'Speed (m/yr)',                    'magma',
     0.0,                               np.max(np.concatenate(all_spd))),
]
n_rows = len(field_specs)

fig2, axes2 = plt.subplots(n_rows, n_snaps, figsize=(4 * n_snaps, 3.5 * n_rows),
                            squeeze=False, sharex=True, sharey=True)

for col, step_key in enumerate(snap_keys):
    snap = snapshots_map[step_key]
    H_mask = snap['H'] < 1.0

    for row, (field_name, label, cmap, vmin, vmax) in enumerate(field_specs):
        ax = axes2[row, col]
        data = snap[field_name].copy()
        data[H_mask] = np.nan

        im = ax.pcolormesh(x_km, y_km, data, cmap=cmap, shading='auto',
                           vmin=vmin, vmax=vmax)
        cb = fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.yaxis.get_major_formatter().set_useOffset(False)
        cb.ax.yaxis.get_major_formatter().set_scientific(False)

        if row == 0:
            ax.set_title(f't = {snap["t_yr"]:.0f} yr', fontsize=11)
        if col == 0:
            ax.set_ylabel(label, fontsize=10)
        ax.set_aspect('equal')

for ax in axes2[-1, :]:
    ax.set_xlabel('x (km)')

fig2.suptitle('Spatial Fields at Selected Times', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / 'coupled_dome_maps.png', dpi=150)
plt.show()
print(f"  Saved: {OUT_DIR / 'coupled_dome_maps.png'}")
