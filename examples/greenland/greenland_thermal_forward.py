"""
Coupled Greenland forward simulation with enthalpy.

Mirrors greenland_forward.py for the momentum / mass-balance setup but
adds a coupled enthalpy solver via ThermalModel. The thermal forcing is
intentionally simple:

  - Surface temperature: elevation-dependent lapse rate, capped at T_melt.
    T_surf(s) = T_SEA_LEVEL + LAPSE_RATE * s.
  - Geothermal heat flux: uniform (no spatial map).
  - Frictional heating: enabled (computed from beta and basal speed).

The script follows the coupled_dome.py pattern:
  1. Momentum-only spin-up for a few steps so the initial geometry relaxes
     against the inverse-derived beta before any thermal feedback.
  2. Re-initialize the thermal state from the relaxed geometry.
  3. Run the coupled loop: pre_momentum() -> momentum solve ->
     surface BC update -> thermal solve. The thermal model updates B
     via the Arrhenius factor each step.

Run interactively or execute as a script. Modify the paths and parameters
below to match your setup.
"""
from pathlib import Path

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

try:
    import pyproj
    CRS = pyproj.CRS("EPSG:3413")
except ImportError:
    CRS = None

from scipy.ndimage import gaussian_filter

from glide.model import IceDynamics, ThermalModel
from glide.data import load_greenland_preprocessed
from glide.io import ZarrWriter, VTIWriter
from glide.enthalpy import T_MELT, BETA_CC, RHO_I, GRAVITY

# ========================================================
# Parameters
# ========================================================
# Thermal forcing
T_SEA_LEVEL = 278.15      # K  
LAPSE_RATE  = -6.5e-3     # K/m (atmospheric lapse rate)
Q_GEO       = 0.005        # W/m^2 (uniform geothermal flux, ~Greenland mean)

# Enthalpy solver
NZ          = 9          # sigma levels
N_SMOOTH    = 30          # max column-sweep iterations per step
H_THIN      = 25.0        # m, columns thinner than this are clamped to T_surf

# Spin-up / time stepping
N_SPINUP    = 5           # momentum-only steps before thermal coupling
DT_YR       = 10.0        # years per step (start small; can be made dynamic later)
T_END       = 2000.0      # years
VTI_INTERVAL_YR = 100.0   # write VTI output every N years
SEC_PER_YR  = 365.25 * 86400.0

# Path to optimized beta (set to None to use a uniform value)
BETA_PATH   = "./inverse/level_0/beta_opt.nc"

# ========================================================
# Load dataset and build the multigrid model
# ========================================================
dataset = load_greenland_preprocessed()
ny, nx, dx = dataset.ny, dataset.nx, dataset.dx

# ny and nx must both divide by 2^(n_levels - 1) cleanly!
model = IceDynamics(n_levels=6, ny=ny, nx=nx, dx=dx,
        x0=dataset.x[0].item(), y0=dataset.y[0].item(),
        crs=CRS)
mg = model.mg

# ========================================================
# Initialize geometry and state
# ========================================================
thk = gaussian_filter(dataset.thickness.values, 1)
mg.state.H.set(thk)
mg.state.H_prev.set(thk)

bed = gaussian_filter(dataset.bed.values, 1)
mg.geometry.bed.set(bed)
mg.geometry.flotation_reg_driving.set(0.1)

# Rheology: start with the same uniform B as greenland_forward.py.
# The thermal model overrides this once update_rheology is enabled.
B = cp.zeros((ny, nx), dtype=cp.float32)
B.fill(1e-17 ** (-1.0 / 3.0) / (917 * 9.81))
mg.rheology.B.set(B)
mg.rheology.eps_reg.set(1e-6)
mg.rheology.n.set(3.0)

# Sliding
if BETA_PATH:
    import xarray as xr
    beta = cp.array(xr.load_dataarray(BETA_PATH))
else:
    beta = cp.zeros((ny, nx), dtype=cp.float32)
    beta.fill(2.5)
mg.sliding.beta.set(beta)
mg.sliding.m.set(1. / 3.)
mg.sliding.water_drag.set(1e-4)

# Calving
mg.calving.calving_rate.set(2000.0)

# Surface mass balance
smb = dataset.smb.values
mg.forcing.smb.set(smb)

# ========================================================
# Multigrid solver options (same as greenland_forward.py)
# ========================================================
model.forward_solver.fas_options.set(
        coarsest_steps=200, pre_steps=10,
        post_steps=150, finest_steps=0,
        relative_tolerance=1e-2, absolute_tolerance=10.0,
        report_norms=True)

# ========================================================
# Thermal model
# ========================================================
grid = mg.levels[0]
thermal = ThermalModel(grid, nz=NZ, n_smooth=N_SMOOTH,
                       update_rheology=False,     # off during spin-up
                       frictional_heating=True)

thermal.ops.smoother_config.report_norms = True
thermal.ops.smoother_config.omega = cp.float32(1.0)
thermal.ops.smoother_config.n_newton = 5
thermal.ops.smoother_config.relaxation = cp.float32(1.0)
thermal.ops.smoother_config.lf_c = cp.float32(1e-4)
thermal.ops.smoother_config.absolute_tolerance = cp.float32(1e-3)
thermal.ops.smoother_config.relative_tolerance = cp.float32(1e-7)
thermal.ops.enthalpy_forcing.h_thin.set(H_THIN)


def surface_temperature(H, bed):
    """Elevation-dependent surface temperature, capped at T_melt.

    T_surf(s) = T_SEA_LEVEL + LAPSE_RATE * s, where s = bed + H.
    Capping at T_melt is done implicitly by set_surface_enthalpy_from_temperature.
    """
    surface_elev = bed + H
    return cp.float32(T_SEA_LEVEL) + cp.float32(LAPSE_RATE) * surface_elev


# Seed thermal state from the *initial* geometry. The column is initialized
# at surface T (cold throughout) — Q_geo will warm the base during the run.
T_surf_init = surface_temperature(grid.state.H.data, grid.geometry.bed.data)
thermal.initialize(T_surface=T_surf_init, T_field=T_surf_init, Q_geo=Q_GEO)

# All enthalpy physics on by default; make it explicit for clarity.
thermal.ops.term_flags.horizontal_advection = True
thermal.ops.term_flags.omega = True
thermal.ops.term_flags.drainage = True

# ========================================================
# Output writers
# ========================================================
vti_2d = VTIWriter('thermal_forward/vti_2d/', base='greenland', dx=mg[0].dx,
        static_fields={'bed': mg[0].geometry.bed,
                       'beta': mg[0].sliding.beta},
        dynamic_fields={'H': mg[0].state.H,
                        'U': [mg[0].state.u, mg[0].state.v],
                        'mask': mg[0].state.mask,
                        'B': mg[0].rheology.B,
                        'T_bed':  lambda: thermal.temperature[:, :, 0],
                        'T_surf': lambda: thermal.temperature[:, :, -1],
                        'T_avg':  lambda: thermal.temperature.mean(axis=2)})
vti_2d.initialize(mg[0])

vti_3d = VTIWriter('thermal_forward/vti_3d/', base='greenland_thermal',
        dx=mg[0].dx, dz=1.0 / (NZ - 1),
        dynamic_fields={'E':             thermal.ops.enthalpy_state.E,
                        'T':             lambda: thermal.temperature,
                        'omega':         thermal.ops.enthalpy_velocity.omega,
                        'water_content': lambda: thermal.water_content})
vti_3d.initialize(None)

zarr_writer = ZarrWriter('thermal_forward/example_run.zarr',
        static_fields={'bed': mg[0].geometry.bed,
                       'beta': mg[0].sliding.beta},
        dynamic_fields={'H': mg[0].state.H,
                        'u': mg[0].state.u,
                        'v': mg[0].state.v,
                        'mask': mg[0].state.mask,
                        'B': mg[0].rheology.B})
zarr_writer.initialize(mg[0], overwrite=True)

# ========================================================
# Momentum-only spin-up
# ========================================================
dt_yr  = cp.float32(DT_YR)
dt_sec = float(DT_YR) * SEC_PER_YR

t = cp.float32(0.0)

print(f"Momentum spin-up ({N_SPINUP} steps, no thermal coupling)...")
for k in range(N_SPINUP):
    print(f"  spin-up step {k+1}/{N_SPINUP} at t={float(t):.1f}")
    model.forward(t, dt_yr)
    t += dt_yr

# Re-initialize thermal state from the relaxed geometry. The fixed B from
# greenland_forward stays in place until the first coupled step, after which
# the Arrhenius feedback takes over.
T_surf_relaxed = surface_temperature(grid.state.H.data, grid.geometry.bed.data)
thermal.initialize(T_surface=T_surf_relaxed,
                   T_field=T_surf_relaxed,
                   Q_geo=Q_GEO)
thermal.update_rheology = True

# Seed B from the (still cold) thermal state so the first coupled step
# doesn't shock the momentum solver with a discontinuous B change.
mg.rheology.B.set(thermal.ops.get_arrhenius_factor() / thermal.B_scale)

# ========================================================
# Coupled forward simulation
# ========================================================
_u0 = 0.5 * (grid.state.u.data[:, 1:] + grid.state.u.data[:, :-1])
_v0 = 0.5 * (grid.state.v.data[1:, :] + grid.state.v.data[:-1, :])
times          = [float(t)]
vol_km3        = [float(cp.sum(grid.state.H.data) * dx * dx / 1e9)]
max_speed_yr   = [float(cp.max(cp.sqrt(_u0 * _u0 + _v0 * _v0)))]
mean_T_bed     = [float(cp.mean(thermal.temperature[:, :, 0]))]
frac_at_pmp    = [0.0]

while float(t) < T_END:
    print(f"Coupled solve at t={float(t):.1f}  dt={DT_YR:.1f}")

    # Snapshot E and H *before* the momentum step so the conservative
    # time derivative rho_i * (H*E - H_prev*E_prev) / dt sees the
    # correct pre-step thickness.
    thermal.pre_momentum()

    # Momentum / mass-balance solve (dt in years).
    model.forward(t, dt_yr)

    # Update the surface BC from the evolving geometry.
    T_surf = surface_temperature(grid.state.H.data, grid.geometry.bed.data)
    thermal.ops.set_surface_enthalpy_from_temperature(T_surf)

    # Enthalpy solve (dt in seconds).
    thermal.step(dt_sec)

    t += dt_yr

    # Diagnostics (cell-centered speed, basal-temperature stats over ice).
    u_c = 0.5 * (grid.state.u.data[:, 1:] + grid.state.u.data[:, :-1])
    v_c = 0.5 * (grid.state.v.data[1:, :] + grid.state.v.data[:-1, :])
    speed = cp.sqrt(u_c * u_c + v_c * v_c)
    H = grid.state.H.data
    ice = H > 100.0
    T_bed = thermal.temperature[:, :, 0]
    # Pressure-melting point at the bed (depth = full column thickness).
    T_pmp_bed = T_MELT - BETA_CC * RHO_I * GRAVITY * H
    at_pmp = ice & (T_bed >= T_pmp_bed - 0.1)

    times.append(float(t))
    vol_km3.append(float(cp.sum(H) * dx * dx / 1e9))
    max_speed_yr.append(float(cp.max(speed)))
    mean_T_bed.append(float(cp.mean(T_bed[ice])) if cp.any(ice) else float('nan'))
    frac_at_pmp.append(float(cp.sum(at_pmp)) / float(cp.sum(ice)) if cp.any(ice) else 0.0)

    vti_2d.append(mg[0], time=t)
    vti_2d.write_pvd()
    vti_3d.append(None, time=t)
    vti_3d.write_pvd()
    zarr_writer.append(mg[0], time=t)

zarr_writer.consolidate_metadata()

# ========================================================
# Summary plots
# ========================================================
out_dir = Path('thermal_forward')
out_dir.mkdir(exist_ok=True)

# --- Time series ---
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(times, vol_km3, 'b-', lw=2)
axes[0, 0].set_ylabel('Volume (km$^3$)')
axes[0, 0].set_title('Ice Volume')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(times, max_speed_yr, 'm-', lw=2)
axes[0, 1].set_ylabel('max |u| (m/yr)')
axes[0, 1].set_title('Maximum Surface Speed')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(times, mean_T_bed, 'r-', lw=2)
axes[1, 0].axhline(T_MELT, color='k', ls='--', alpha=0.5, label='T$_{melt}$')
axes[1, 0].set_ylabel('Temperature (K)')
axes[1, 0].set_xlabel('Time (yr)')
axes[1, 0].set_title('Mean Basal Temperature (ice-covered)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(times, [100 * f for f in frac_at_pmp], 'g-', lw=2)
axes[1, 1].set_ylabel('Bed at PMP (% of ice area)')
axes[1, 1].set_xlabel('Time (yr)')
axes[1, 1].set_title('Temperate-Bed Fraction')
axes[1, 1].grid(True, alpha=0.3)

fig.suptitle('Coupled Greenland Forward Run', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(out_dir / 'summary_timeseries.png', dpi=300)
print(f"Saved: {out_dir / 'summary_timeseries.png'}")

# --- Final-state maps ---
H_np = cp.asnumpy(grid.state.H.data)
T_bed_np = cp.asnumpy(thermal.temperature[:, :, 0])
B_np = cp.asnumpy(grid.rheology.B.data)
u_c = 0.5 * (grid.state.u.data[:, 1:] + grid.state.u.data[:, :-1])
v_c = 0.5 * (grid.state.v.data[1:, :] + grid.state.v.data[:-1, :])
speed_np = cp.asnumpy(cp.sqrt(u_c * u_c + v_c * v_c))

ice_mask = H_np < 1.0
T_bed_np = np.where(ice_mask, np.nan, T_bed_np)
speed_plot = np.where(ice_mask, np.nan, speed_np)

T_vmin = float(np.nanmin(T_bed_np))
T_vmax = float(np.nanmax(T_bed_np))

fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
for ax, data, title, cmap in [
    (axes2[0, 0], H_np,       'Thickness (m)',      'cividis'),
    (axes2[0, 1], speed_plot, 'Surface speed (m/yr, log)', 'magma'),
    (axes2[1, 0], T_bed_np,   'Basal temperature (K)', 'coolwarm'),
    (axes2[1, 1], B_np,       'Arrhenius B (GLIDE units)', 'viridis'),
]:
    if 'log' in title:
        from matplotlib.colors import LogNorm
        im = ax.imshow(np.maximum(data, 1.0), origin='upper', cmap=cmap,
                       norm=LogNorm(vmin=1.0, vmax=max(10.0, np.nanmax(data))))
    elif 'Basal temperature' in title:
        im = ax.imshow(data, origin='upper', cmap=cmap,
                       vmin=T_vmin, vmax=T_vmax)
    else:
        im = ax.imshow(data, origin='upper', cmap=cmap)
    fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_aspect('equal')

fig2.suptitle(f'Final state at t = {float(t):.0f} yr', fontsize=14, fontweight='bold')
plt.tight_layout()
fig2.savefig(out_dir / 'summary_final_maps.png', dpi=300)
print(f"Saved: {out_dir / 'summary_final_maps.png'}")
