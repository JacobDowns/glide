"""
Verify the enthalpy solver across stress balances: SSA vs MOLHO vs DIVA.

For each scheme this runs, on real Greenland forcing (Martos Q_geo + SeaRISE
surface T -- see greenland_thermal_forcing.py) and that scheme's own inverted
beta (see greenland_inverse.py, run once per scheme):

  * an UNCOUPLED thermal equilibrium (static geometry + velocity, large-dt
    thermal-only spin-up -- as in greenland_thermal_equilibrium.py), and
  * a short COUPLED transient (as in greenland_thermal_forward.py).

then draws the 3-way comparison figures (equilibrium basal-T maps + convergence,
transient volume + temperate-bed curves).

Prerequisites (run once each):
    python greenland_thermal_forcing.py
    for s in ssa molho diva:  SCHEME=$s python greenland_inverse.py   # -> inverse_<s>/
Then:
    python greenland_thermal_compare.py

The thermal model resolves the full vertical velocity profile: u_bar for SSA
(plug), and u(sigma) = u_b + (u_s-u_b)*(1-(1-sigma)^(n+1)) for MOLHO/DIVA, plus
strain (deformational) heating from that shear.  So SSA carries no shear or strain
heating while MOLHO/DIVA do -- the temperate-bed differences below (SSA ~39% vs
MOLHO/DIVA ~51-55% at equilibrium) are that near-basal deformational warming,
which the plug-flow SSA cannot represent.
"""
import os
import cupy as cp
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import gaussian_filter

from glide.model import IceDynamics, ThermalModel
from glide.enthalpy import T_MELT, BETA_CC, RHO_I, GRAVITY
from glide.data import load_greenland_preprocessed

SCHEMES = ["ssa", "molho", "diva"]
COL = {"ssa": "#1f77b4", "molho": "#d62728", "diva": "#2ca02c"}
SEC_PER_YR = 365.25 * 86400.0
OUT = "thermal_compare"

ds = load_greenland_preprocessed()
ny, nx, dx = ds.ny, ds.nx, ds.dx
thk = gaussian_filter(ds.thickness.values, 1)
bed = gaussian_filter(ds.bed.values, 1)
q_geo = cp.array(xr.load_dataarray("./data/greenland_q_geo.nc"))
t_surf = cp.array(xr.load_dataarray("./data/greenland_tsurf.nc"))


def make_model(scheme):
    m = IceDynamics(n_levels=6, ny=ny, nx=nx, dx=dx,
                    x0=ds.x[0].item(), y0=ds.y[0].item(), crs=None, stress_scheme=scheme)
    mg = m.mg
    mg.state.H.set(thk); mg.state.H_prev.set(thk)
    mg.geometry.bed.set(bed); mg.geometry.depth.set(np.maximum(-bed, 0.0))
    mg.geometry.sigmoid_c.set(0.1); mg.geometry.sigmoid_k.set(3.0)
    mg.rheology.B.set(cp.full((ny, nx), 1e-17 ** (-1.0 / 3.0) / (917 * 9.81), dtype=cp.float32))
    mg.rheology.eps_reg.set(1e-6); mg.rheology.n.set(3.0)
    if scheme == "diva":
        mg.rheology.n_sigma.set(6.0); mg.rheology.eps_reg_shear.set(1e-6)
    mg.sliding.beta.set(cp.array(xr.load_dataarray(f"./inverse_{scheme}/level_0/beta_opt.nc")))
    mg.sliding.m.set(1. / 3.); mg.sliding.water_drag.set(1e-4)
    mg.calving.calving_rate.set(2000.0); mg.forcing.smb.set(ds.smb.values)
    m.forward_solver.fas_options.set(coarsest_steps=200, pre_steps=10,
        post_steps=(30 if scheme == "diva" else 150), finest_steps=0,
        relative_tolerance=1e-2, absolute_tolerance=10.0, report_norms=False)
    m.forward_solver.vanka_options.omega.set(cp.float32(0.25))
    m.forward_solver.vanka_options.newton_options.momentum_damping.set(
        cp.float32(0.1 if scheme in ("diva", "molho") else 0.01))
    return m, mg, mg.levels[0]


def thermal_for(grid):
    th = ThermalModel(grid, nz=9, n_smooth=40, update_rheology=False, frictional_heating=True)
    th.ops.enthalpy_forcing.h_thin.set(25.0)
    return th


def equilibrium(scheme):
    m, mg, grid = make_model(scheme)
    dt_yr = cp.float32(10.0); t = cp.float32(0.0)
    for _ in range(8):                       # relax geometry + velocity, then freeze
        m.forward(t, dt_yr); t += dt_yr
    H = grid.state.H.data; ice = H > 100.0; n_ice = float(cp.sum(ice))
    T_pmp = T_MELT - BETA_CC * RHO_I * GRAVITY * H
    th = thermal_for(grid); th.initialize(T_surface=t_surf, T_field=t_surf, Q_geo=q_geo)
    dt_sec = 1000.0 * SEC_PER_YR; prev = th.temperature[:, :, 0].copy()
    kyr, thaw, meanT = [], [], []
    for k in range(250):                     # thermal-only spin-up (implicit Euler, no CFL)
        th.pre_momentum(); th.step(dt_sec)
        Tb = th.temperature[:, :, 0]
        d = float(cp.mean(cp.abs(Tb - prev)[ice])); prev = Tb.copy()
        kyr.append((k + 1) * 1.0)
        thaw.append(100 * float(cp.sum(ice & (Tb >= T_pmp - 0.1))) / n_ice)
        meanT.append(float(cp.mean(Tb[ice])))
        if d < 1e-3:
            break
    print(f"[{scheme}] equilibrium: {kyr[-1]:.0f} kyr  thawed {thaw[-1]:.1f}%  meanTbed {meanT[-1]:.1f} K")
    return dict(kyr=np.array(kyr), thaw=np.array(thaw), Tbed=cp.asnumpy(Tb), H=cp.asnumpy(H))


def transient(scheme, years=500.0):
    m, mg, grid = make_model(scheme)
    dt_yr = cp.float32(10.0); dt_sec = 10.0 * SEC_PER_YR; t = cp.float32(0.0)
    th = thermal_for(grid)
    for _ in range(5):
        m.forward(t, dt_yr); t += dt_yr
    th.initialize(T_surface=t_surf, T_field=t_surf, Q_geo=q_geo); th.update_rheology = True
    mg.rheology.B.set(th.ops.get_arrhenius_factor() / th.B_scale)
    tt, vol, thaw = [float(t)], [float(cp.sum(grid.state.H.data) * dx * dx / 1e9)], []

    def rec():
        H = grid.state.H.data; ice = H > 100.0
        T_pmp = T_MELT - BETA_CC * RHO_I * GRAVITY * H; Tb = th.temperature[:, :, 0]
        thaw.append(100 * float(cp.sum(ice & (Tb >= T_pmp - 0.1))) / float(cp.sum(ice)))
    rec()
    while float(t) < years:
        th.pre_momentum(); m.forward(t, dt_yr)
        th.ops.set_surface_enthalpy_from_temperature(t_surf); th.step(dt_sec); t += dt_yr
        tt.append(float(t)); vol.append(float(cp.sum(grid.state.H.data) * dx * dx / 1e9)); rec()
    print(f"[{scheme}] transient: dV {100*(vol[-1]-vol[0])/vol[0]:+.2f}%  thawed {thaw[-1]:.1f}%")
    return dict(t=np.array(tt), vol=np.array(vol), thaw=np.array(thaw))


def main():
    os.makedirs(OUT, exist_ok=True)
    E = {s: equilibrium(s) for s in SCHEMES}
    T = {s: transient(s) for s in SCHEMES}
    ice = E["diva"]["H"] > 10

    # equilibrium figure
    fig = plt.figure(figsize=(15, 9)); gs = fig.add_gridspec(2, 3, height_ratios=[2, 1.1])
    for c, s in enumerate(SCHEMES):
        ax = fig.add_subplot(gs[0, c])
        im = ax.imshow(np.where(ice, E[s]["Tbed"], np.nan), origin="upper", cmap="coolwarm", vmin=244, vmax=273.2)
        ax.set_title(f"{s.upper()}  (thawed {E[s]['thaw'][-1]:.0f}%)"); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, cax=fig.add_axes([0.92, 0.55, 0.015, 0.32]), label="basal T (K)")
    axc = fig.add_subplot(gs[1, 0:2])
    for s in SCHEMES:
        axc.plot(E[s]["kyr"], E[s]["thaw"], color=COL[s], lw=2, label=f"{s.upper()} ({E[s]['thaw'][-1]:.1f}%)")
    axc.set_xlabel("spin-up time (kyr)"); axc.set_ylabel("temperate bed (%)")
    axc.set_title("Equilibrium convergence"); axc.legend(); axc.grid(alpha=.3)
    axd = fig.add_subplot(gs[1, 2])
    dd = np.where(ice, E["diva"]["Tbed"], np.nan) - np.where(ice, E["ssa"]["Tbed"], np.nan)
    im2 = axd.imshow(dd, origin="upper", cmap="PuOr_r", norm=TwoSlopeNorm(0, -6, 6))
    axd.set_title("DIVA − SSA basal T (K)"); axd.set_xticks([]); axd.set_yticks([])
    fig.colorbar(im2, ax=axd, fraction=0.046)
    fig.suptitle("Greenland thermal EQUILIBRIUM — SSA / MOLHO / DIVA (real 3D velocity + strain heating)",
                 fontweight="bold", fontsize=14)
    fig.savefig(f"{OUT}/compare_equilibrium.png", dpi=130, bbox_inches="tight")

    # transient figure
    fig2, ax2 = plt.subplots(1, 2, figsize=(14, 5.5))
    for s in SCHEMES:
        v = T[s]["vol"]
        ax2[0].plot(T[s]["t"], 100 * (v - v[0]) / v[0], color=COL[s], lw=2.3,
                    label=f"{s.upper()} ({100*(v[-1]-v[0])/v[0]:+.2f}%)")
        ax2[1].plot(T[s]["t"], T[s]["thaw"], color=COL[s], lw=2.3,
                    label=f"{s.upper()} ({T[s]['thaw'][-1]:.1f}%)")
    ax2[0].set_title("Ice-volume change"); ax2[0].set_ylabel("%")
    ax2[1].set_title("Temperate-bed fraction"); ax2[1].set_ylabel("% of ice area")
    for a in ax2:
        a.set_xlabel("year"); a.legend(); a.grid(alpha=.3)
    fig2.suptitle("Greenland coupled TRANSIENT — SSA / MOLHO / DIVA (real 3D velocity + strain heating)",
                  fontweight="bold", fontsize=13.5)
    fig2.tight_layout(); fig2.savefig(f"{OUT}/compare_transient.png", dpi=130, bbox_inches="tight")
    print(f"saved {OUT}/compare_equilibrium.png, {OUT}/compare_transient.png")


if __name__ == "__main__":
    main()
