"""ISMIP-HOM experiment C, SSA against DIVA.

Experiment C (Pattyn et al. 2008): a slab of uniform 1000 m thickness on a 0.1 degree surface
slope, with basal drag varying sinusoidally in both directions,

    beta = 1000 + 1000*sin(2*pi*x/L)*sin(2*pi*y/L)   [kPa a / m]

for length scales L from 5 to 160 km, and A = 1e-16 Pa^-3 a^-1. GLIDE imposes u = v = 0 on the
domain edges rather than periodic boundaries, so the domain is tiled 5x5 in L and only the
central tile is reported -- the same trick the existing ismip-hom.py example uses.

WHAT THIS DOES AND DOES NOT ESTABLISH.  It contrasts the two stress balances on a benchmark
geometry, which is worth having: the difference is the vertical shear DIVA adds, and its size
and length-scale dependence are physically interpretable. It is NOT a validation against
ISMIP-HOM, for two reasons worth being explicit about:

  1. ISMIP-HOM has no analytical solution. It is a model intercomparison; the reference is the
     spread of the participating full-Stokes models, published as figures in Pattyn et al. 2008
     and reproduced for DIVA in Goldberg (2011). Comparing against it needs those numbers, from
     the ISMIP-HOM archive or digitised from the figures. None of that is in this repository, and
     nothing here should be read as agreeing or disagreeing with full Stokes.

  2. The published diagnostic for experiment C is SURFACE velocity, which this now reports:
     u_s = u_b + tau_b*F1 with F1 = H*int (zeta/eta) dzeta (Arthern's I_1), accumulated by the
     coefficient kernel and exposed as state.u_s. Under SSA the surface and depth-averaged
     speeds coincide, so the SSA column is the same quantity either way; under DIVA they differ
     by the vertical shear, which is the whole point. Both are printed below.

For an analytic check of the machinery DIVA adds, see tests/diva_slab_test.py -- the uniform slab
does have a closed-form solution, and F2 matches it to float32 round-off.

    uv run python examples/ismip-hom/ismip_hom_c_ssa_vs_diva.py
"""
import contextlib
import io

import cupy as cp
import numpy as np

from glide.model import IceDynamics

LENGTH_SCALES = [5000, 10000, 20000, 40000, 80000, 160000]
BASE_RES = 128
TILES = 5
N_LEVELS = 6
RHO_I, GRAV = cp.float32(917.0), cp.float32(9.81)


def run(L, stress_balance, n_sigma=8.0):
    ny = nx = BASE_RES * TILES
    x = cp.linspace(0, TILES * L, nx, dtype=cp.float32)
    y = cp.linspace(0, TILES * L, ny, dtype=cp.float32)
    dx = (x[1] - x[0]).item()
    X, Y = cp.meshgrid(x, y)

    srf = 1000.0 * cp.ones((ny, nx), dtype=cp.float32) - cp.tan(cp.deg2rad(0.1)) * X + 10000
    bed = srf - 1000.0
    beta = (1000 * cp.sin(2 * cp.pi * X / L) * cp.sin(2 * cp.pi * Y / L) + 1000) / (RHO_I * GRAV)
    B = cp.ones((ny, nx), dtype=cp.float32)
    B.fill((1e-16 ** -(1. / 3)) / (RHO_I * GRAV))

    model = IceDynamics(n_levels=N_LEVELS, ny=ny, nx=nx, dx=dx)
    model.mg.geometry.bed.set(bed)
    model.mg.rheology.B.set(B)
    model.mg.sliding.beta.set(beta)
    model.mg.sliding.m.set(1.0)
    model.mg.sliding.u_reg.set(1.0)
    model.mg.state.H.set(srf - bed)
    model.mg.state.H_prev.set(srf - bed)
    model.mg.rheology.stress_balance.set(stress_balance)
    model.mg.rheology.n_sigma.set(n_sigma)
    model.forward_solver.fas_options.set(
        coarsest_steps=300, pre_steps=10, post_steps=20, finest_steps=0,
        relative_tolerance=5e-3, absolute_tolerance=1.0, report_norms=True)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model.forward(0.0, cp.float32(0.1))
    cycles = [l for l in buf.getvalue().splitlines() if "V-cycle" in l]
    resid = float(cycles[-1].split("|r|/|r0| =")[1].split(",")[0]) if cycles else float('nan')
    r_ub = np.nan
    if cycles and "|r_Ub|/|Ubar| =" in cycles[-1]:
        r_ub = float(cycles[-1].split("|r_Ub|/|Ubar| =")[1].split(",")[0])

    # the central tile, along y = L/4 (modulo the tiling)
    ys = int((TILES // 2 + 1. / 4) * BASE_RES)
    xs = slice(TILES // 2 * BASE_RES, (TILES // 2 + 1) * BASE_RES)
    u = cp.asnumpy(model.mg[0].state.u.data)
    v = cp.asnumpy(model.mg[0].state.v.data)
    # u_b and u_s are cell-centred SPEEDS, so they must be compared against the cell-centred
    # speed, not against the x-component of the facet velocity.
    speed = np.hypot(0.5 * (u[:, :-1] + u[:, 1:]), 0.5 * (v[:-1, :] + v[1:, :]))[ys, xs]
    if stress_balance > 0.5:
        u_b = cp.asnumpy(model.mg[0].state.u_b.data)[ys, xs]
        u_s = cp.asnumpy(model.mg[0].state.u_s.data)[ys, xs]
    else:
        u_b = u_s = speed          # SSA: no vertical shear, so all three coincide
    return speed, u_s, u_b, resid, r_ub, len(cycles)


def main():
    print("ISMIP-HOM experiment C along y = L/4, central tile.  SURFACE speed is the published")
    print("diagnostic; the depth-averaged column is shown alongside since that is what the")
    print("momentum balance solves and the two differ only under DIVA.\n")
    hdr = (f"{'L (km)':>7} | {'SSA surf':>9} {'DIVA surf':>10} {'diff':>7} | "
           f"{'DIVA ubar':>10} {'us/ubar':>8} | {'u_b/ubar':>9} | {'|r_Ub|':>8}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for L in LENGTH_SCALES:
        sp_ssa, us_ssa, _, r_s, _, n_s = run(L, 0.0)
        sp_div, us_div, ub_div, r_d, rub, n_d = run(L, 1.0)
        rel = (us_div.mean() - us_ssa.mean()) / us_ssa.mean()
        print(f"{L / 1000:7g} | {us_ssa.mean():9.3f} {us_div.mean():10.3f} {rel:+6.1%} | "
              f"{sp_div.mean():10.3f} {us_div.mean() / sp_div.mean():8.3f} | "
              f"{ub_div.mean() / sp_div.mean():9.3f} | {rub:8.1e}")
        rows.append((L, us_ssa, us_div, sp_div, ub_div))

    print("""
Reading it: the DIVA/SSA difference is the vertical shear SSA cannot represent. us/ubar > 1 is
the surface moving faster than the column mean, and u_b/ubar is the sliding fraction -- near 1
means plug flow with little for DIVA to add, well below 1 means internal deformation matters.
Both schemes solve the same 2-D operator, so every difference is in eta_bar and beta_eff.

One systematic bias remains in the DIVA numbers and it suppresses deformation, so the surface
difference is if anything an underestimate: eps_reg = 1e-6 regularizes the viscosity over the
upper part of the column (18% at tau_b ~ 120 kPa, and more at lower driving stress), costing
~0.1-8% in F2 depending on the driving stress -- quantified in tests/diva_slab_test.py. The
quadrature bias that used to sit alongside it is gone: the vertical integrals are now
Gauss-Legendre and exact for integer n.""")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(13, 7), sharex=True)
        for ax, (L, us_ssa, us_div, sp_div, ub_div) in zip(axs.ravel(), rows):
            s = np.linspace(0, 1, BASE_RES)
            ax.plot(s, us_ssa, label="SSA (surface = mean)", lw=1.6)
            ax.plot(s, us_div, label="DIVA surface", lw=1.6)
            ax.plot(s, sp_div, label="DIVA depth-avg", lw=1.2, ls=":")
            ax.plot(s, ub_div, label="DIVA basal", lw=1.0, ls="--")
            ax.set_title(f"L = {L / 1000:g} km")
            ax.set_xlim(0, 1)
            ax.grid(alpha=.3)
        axs[0, 0].legend()
        for ax in axs[-1]:
            ax.set_xlabel("$x/L$")
        for ax in axs[:, 0]:
            ax.set_ylabel(r"speed  (m a$^{-1}$)")
        fig.suptitle("ISMIP-HOM C: surface, depth-averaged and basal speed along $y=L/4$")
        fig.tight_layout()
        out = "ismip_hom_c_ssa_vs_diva.png"
        fig.savefig(out, dpi=130)
        print(f"\nwrote {out}")
    except Exception as exc:                                  # plotting is a convenience only
        print(f"\n(plot skipped: {exc})")


if __name__ == '__main__':
    main()
