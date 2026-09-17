"""
Greenland thermal EQUILIBRIUM run (uncoupled).

Static geometry + static velocity: solve the momentum balance once, freeze the
geometry and velocity, then spin the enthalpy solver to steady state at a LARGE
time step.  Implicit Euler on the enthalpy equation is unconditionally stable
(no CFL limit), so a few hundred cheap thermal-only steps at dt ~ 1 kyr span the
~10^5 yr diffusive equilibration -- far cheaper than a fully-coupled transient at
dt = 10 yr (which would need ~10^4 momentum+thermal steps).

Intended as a starting point for ISMIP-style thermal initialisation.  Everything
you would swap for a different experiment (geometry, velocity source, SMB,
geothermal flux, surface temperature, stress balance) is marked "SWAP" below.
Produces the equilibrium 3D temperature, basal temperature and temperate-bed
fraction, plus a convergence plot.

    # optional but recommended first: build real Q_geo / surface-T maps
    python greenland_thermal_forcing.py
    python greenland_thermal_equilibrium.py

This example is deliberately UNCOUPLED: the velocity is frozen at the initial
momentum solve.  For a thermomechanically-consistent equilibrium, wrap steps
(5)-(6) in an outer loop that re-solves momentum from the updated Arrhenius B.
"""
from pathlib import Path

import cupy as cp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

try:
    import pyproj
    CRS = pyproj.CRS("EPSG:3413")
except ImportError:
    CRS = None

from glide.model import IceDynamics, ThermalModel
from glide.data import load_greenland_preprocessed
from glide.io import VTIWriter
from glide.enthalpy import T_MELT, BETA_CC, RHO_I, GRAVITY

# ======================================================================
# Configuration
# ======================================================================
STRESS_SCHEME = "diva"        # 'ssa' | 'molho' | 'diva'
NZ            = 9             # sigma levels
N_RELAX       = 8             # momentum steps to relax geometry+velocity before freezing
DT_RELAX_YR   = 10.0
DT_SPINUP_KYR = 1.0           # thermal-only step (large; implicit Euler is unconditional)
MAX_STEPS     = 300           # cap on thermal steps
TOL_DTBED     = 1.0e-3        # convergence: mean |dT_bed| per step over ice (K)
H_THIN        = 25.0
SEC_PER_YR    = 365.25 * 86400.0
OUT           = Path("thermal_equilibrium")

# ======================================================================
# Inputs -- SWAP these for a different experiment
# ======================================================================
dataset = load_greenland_preprocessed()
ny, nx, dx = dataset.ny, dataset.nx, dataset.dx

thickness = gaussian_filter(dataset.thickness.values, 1)   # SWAP: geometry
bed       = gaussian_filter(dataset.bed.values, 1)         # SWAP: geometry
smb       = dataset.smb.values                             # SWAP: surface mass balance [m/yr]

# Basal sliding coefficient (SWAP): an inverted beta if available, else uniform.
import os
BETA_PATH = "./inverse/level_0/beta_opt.nc"
if os.path.exists(BETA_PATH):
    import xarray as xr
    beta = cp.array(xr.load_dataarray(BETA_PATH))
else:
    beta = cp.full((ny, nx), 2.5, dtype=cp.float32)

# Thermal forcing (SWAP): Martos/SeaRISE maps from greenland_thermal_forcing.py,
# else uniform geothermal flux and a lapse-rate surface temperature.
import xarray as xr
if os.path.exists("./data/greenland_q_geo.nc"):
    q_geo = cp.array(xr.load_dataarray("./data/greenland_q_geo.nc"))       # [W/m^2]
else:
    q_geo = cp.full((ny, nx), 0.05, dtype=cp.float32)
if os.path.exists("./data/greenland_tsurf.nc"):
    t_surf = cp.array(xr.load_dataarray("./data/greenland_tsurf.nc"))      # [K]
else:
    surf_elev = cp.asarray(bed + thickness, dtype=cp.float32)
    t_surf = cp.float32(278.15) + cp.float32(-6.5e-3) * surf_elev
print(f"scheme={STRESS_SCHEME}  Q_geo mean={float(cp.mean(q_geo)):.4f} W/m^2  "
      f"T_surf mean={float(cp.mean(t_surf)):.1f} K")

# ======================================================================
# Build the momentum model
# ======================================================================
model = IceDynamics(n_levels=6, ny=ny, nx=nx, dx=dx,
                    x0=dataset.x[0].item(), y0=dataset.y[0].item(),
                    crs=CRS, stress_scheme=STRESS_SCHEME)
mg = model.mg
mg.state.H.set(thickness); mg.state.H_prev.set(thickness)
mg.geometry.bed.set(bed); mg.geometry.depth.set(np.maximum(-bed, 0.0))
mg.geometry.sigmoid_c.set(0.1); mg.geometry.sigmoid_k.set(3.0)
Bg = cp.full((ny, nx), 1e-17 ** (-1.0 / 3.0) / (917 * 9.81), dtype=cp.float32)
mg.rheology.B.set(Bg); mg.rheology.eps_reg.set(1e-6); mg.rheology.n.set(3.0)
if STRESS_SCHEME == "diva":
    mg.rheology.n_sigma.set(6.0); mg.rheology.eps_reg_shear.set(1e-6)
mg.sliding.beta.set(beta); mg.sliding.m.set(1. / 3.); mg.sliding.water_drag.set(1e-4)
mg.calving.calving_rate.set(2000.0); mg.forcing.smb.set(smb)
model.forward_solver.fas_options.set(
    coarsest_steps=200, pre_steps=10,
    post_steps=(30 if STRESS_SCHEME == "diva" else 150), finest_steps=0,
    relative_tolerance=1e-2, absolute_tolerance=10.0, report_norms=False)
model.forward_solver.vanka_options.omega.set(cp.float32(0.25))
model.forward_solver.vanka_options.newton_options.momentum_damping.set(
    cp.float32(0.1 if STRESS_SCHEME in ("diva", "molho") else 0.01))
grid = mg.levels[0]

# ======================================================================
# (1) Relax geometry + velocity with a short momentum spin-up, then FREEZE.
# ======================================================================
print(f"Momentum relaxation ({N_RELAX} steps)...")
dt_relax = cp.float32(DT_RELAX_YR); t = cp.float32(0.0)
for _ in range(N_RELAX):
    model.forward(t, dt_relax); t += dt_relax
# From here the geometry (H) and velocity (grid.state.u/v) are held fixed --
# the thermal loop never calls model.forward() again.
H = grid.state.H.data
ice = H > 100.0
n_ice = float(cp.sum(ice))
T_pmp_bed = T_MELT - BETA_CC * RHO_I * GRAVITY * H

# ======================================================================
# Thermal solver, initialised isothermal at the surface temperature.
# ======================================================================
thermal = ThermalModel(grid, nz=NZ, n_smooth=40,
                       update_rheology=False,      # velocity is frozen -> B feedback inert
                       frictional_heating=True)
thermal.ops.enthalpy_forcing.h_thin.set(H_THIN)
thermal.initialize(T_surface=t_surf, T_field=t_surf, Q_geo=q_geo)


def basal_stats():
    Tb = thermal.temperature[:, :, 0]
    frac = 100.0 * float(cp.sum(ice & (Tb >= T_pmp_bed - 0.1))) / n_ice
    return float(cp.mean(Tb[ice])), frac


# ======================================================================
# (2) Thermal-only spin-up to steady state at a large time step.
# ======================================================================
dt_sec = DT_SPINUP_KYR * 1000.0 * SEC_PER_YR
kyr_axis, mean_tbed_hist, thawed_hist = [], [], []
prev_tbed = thermal.temperature[:, :, 0].copy()
print(f"Thermal spin-up: dt={DT_SPINUP_KYR} kyr, up to {MAX_STEPS} steps "
      f"(static geometry + velocity)...")
for k in range(MAX_STEPS):
    thermal.pre_momentum()
    thermal.step(dt_sec)
    tbed = thermal.temperature[:, :, 0]
    d_tbed = float(cp.mean(cp.abs(tbed - prev_tbed)[ice]))
    prev_tbed = tbed.copy()
    mean_tbed, thawed = basal_stats()
    kyr_axis.append((k + 1) * DT_SPINUP_KYR)
    mean_tbed_hist.append(mean_tbed); thawed_hist.append(thawed)
    if k % 10 == 0 or d_tbed < TOL_DTBED:
        print(f"  step {k:3d}  {(k + 1) * DT_SPINUP_KYR:6.0f} kyr  "
              f"meanTbed={mean_tbed:6.2f} K  thawed={thawed:5.1f}%  d|Tbed|={d_tbed:.2e}")
    if d_tbed < TOL_DTBED:
        print(f"  converged at {(k + 1) * DT_SPINUP_KYR:.0f} kyr")
        break

# ======================================================================
# Outputs: equilibrium fields + convergence plot.
# ======================================================================
OUT.mkdir(exist_ok=True)
T3d = thermal.temperature                     # (ny, nx, nz) K
Tbed_np = cp.asnumpy(T3d[:, :, 0])
thawed_mask = cp.asnumpy(ice & (T3d[:, :, 0] >= T_pmp_bed - 0.1))
H_np = cp.asnumpy(H)

# save the equilibrium temperature as an initial condition for coupled runs
xr.Dataset(
    {"T": (("y", "x", "z"), cp.asnumpy(T3d)),
     "T_bed": (("y", "x"), Tbed_np),
     "water_content": (("y", "x", "z"), cp.asnumpy(thermal.water_content))},
    coords={"y": dataset.y.values, "x": dataset.x.values,
            "z": np.linspace(0.0, 1.0, NZ)},
).to_netcdf(OUT / "greenland_thermal_equilibrium.nc")

vti = VTIWriter(str(OUT / "vti_3d") + "/", base="greenland_equilibrium",
                dx=grid.dx, dz=1.0 / (NZ - 1),
                dynamic_fields={"T": lambda: thermal.temperature,
                                "water_content": lambda: thermal.water_content})
vti.initialize(None); vti.append(None, time=kyr_axis[-1]); vti.write_pvd()

# convergence
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].plot(kyr_axis, mean_tbed_hist, "r-", lw=2)
ax[0].set_xlabel("spin-up time (kyr)"); ax[0].set_ylabel("mean basal T (K)")
ax[0].axhline(T_MELT, color="k", ls="--", alpha=0.5, label="T$_{melt}$"); ax[0].legend()
ax[0].set_title("Mean basal temperature"); ax[0].grid(alpha=0.3)
ax[1].plot(kyr_axis, thawed_hist, "g-", lw=2)
ax[1].set_xlabel("spin-up time (kyr)"); ax[1].set_ylabel("temperate bed (% of ice area)")
ax[1].set_title("Temperate-bed fraction"); ax[1].grid(alpha=0.3)
fig.suptitle(f"Greenland thermal equilibrium ({STRESS_SCHEME.upper()}, uncoupled)",
             fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "convergence.png", dpi=150)

# equilibrium maps
Tb_plot = np.where(H_np > 1.0, Tbed_np, np.nan)
fig2, ax2 = plt.subplots(1, 2, figsize=(12, 8))
im = ax2[0].imshow(Tb_plot, origin="upper", cmap="coolwarm")
ax2[0].set_title("Equilibrium basal temperature (K)"); fig2.colorbar(im, ax=ax2[0], fraction=0.046)
ax2[1].imshow(np.where(H_np > 1.0, thawed_mask, np.nan), origin="upper", cmap="Blues")
ax2[1].set_title("Thawed bed (temperate)")
for a in ax2:
    a.set_xticks([]); a.set_yticks([])
fig2.suptitle(f"Greenland thermal equilibrium ({STRESS_SCHEME.upper()})", fontweight="bold")
fig2.tight_layout(); fig2.savefig(OUT / "equilibrium_maps.png", dpi=150)
print(f"saved outputs to {OUT}/  (netCDF, VTI, convergence.png, equilibrium_maps.png)")
