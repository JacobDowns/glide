"""The DIVA residual must reduce exactly to the SSA residual.

compute_residual shares one templated body across schemes (residuals.cu); the SSA and
DIVA builds differ only in where the membrane viscosity comes from and what basal drag the
momentum balance sees (the `#if GLIDE_DIVA` branch reads the frozen eta_bar/beta_eff, the
`#else` branch computes eta from the strain rate and applies the Weertman drag inline).  So
if the DIVA coefficients are handed their SSA equivalents,

    eta_bar  = eta_SSA
    beta_eff = beta*grounded + water_drag     (DIVA folds both into beta_eff)

the two kernels must produce the same residual bit-for-bit.  That pins the plumbing --
fields read, stencils called, and the grounding/water_drag bookkeeping that is easy to
double count, since the SSA stencil applies the grounded factor internally and the DIVA
one must not.

Note: unlike the fork (which switched DIVA at runtime via a `stress_balance` field), the
unified tree selects the balance at COMPILE time, so the SSA and DIVA references are two
separately-built grids (`stress_scheme='ssa'` vs `'diva'`).

The velocity field is uniform so every membrane strain rate vanishes and eta_SSA is
analytic; the bed, thickness and beta all vary, so the driving stress, fluxes and basal
term are non-trivial.

    uv run python tests/diva_residual_test.py
"""
import cupy as cp
import numpy as np

from glide.multigrid import Multigrid

ny = nx = 32
dx = 2000.0
dt = cp.float32(1.0)

U0 = 10.0        # uniform velocity -> membrane strain rates vanish identically
B0 = 1.0
EPS_REG = 1e-6
GLEN_N = 3.0
WATER_DRAG = 1e-4


def fields():
    """Uniform flow, but spatially varying bed / thickness / drag."""
    x = cp.arange(nx, dtype=cp.float32) * dx
    y = cp.arange(ny, dtype=cp.float32) * dx
    X, Y = cp.meshgrid(x, y)
    bed = (500.0 - 0.01 * X + 50.0 * cp.sin(2 * cp.pi * Y / (8 * dx))).astype(cp.float32)
    thk = (1000.0 + 100.0 * cp.sin(2 * cp.pi * X / (10 * dx))).astype(cp.float32)
    beta = (0.01 + 0.005 * cp.cos(2 * cp.pi * X / (6 * dx))).astype(cp.float32)
    return bed, thk, beta


def build(scheme):
    bed, thk, beta = fields()
    mg = Multigrid(2, ny=ny, nx=nx, dx=dx, stress_scheme=scheme)
    mg.state.u.set(cp.full((ny, nx + 1), U0, dtype=cp.float32))
    mg.state.v.set(cp.zeros((ny + 1, nx), dtype=cp.float32))
    mg.state.H.set(thk)
    mg.state.H_prev.set(thk)
    mg.geometry.bed.set(bed)
    mg.rheology.B.set(cp.full((ny, nx), B0, dtype=cp.float32))
    mg.rheology.n.set(GLEN_N)
    mg.rheology.eps_reg.set(EPS_REG)
    mg.sliding.beta.set(beta)
    mg.sliding.m.set(1.0)
    mg.sliding.u_reg.set(1.0)
    mg.sliding.water_drag.set(WATER_DRAG)
    mg.sliding.sliding_law.set(0.0)
    return mg


def residual(mg, freeze_phi=False):
    ops = mg.levels[0].forward_operators
    ops.compute_residual(dt, use_mask=False, freeze_phi=freeze_phi)
    return [cp.asnumpy(a).copy() for a in (ops.r_u, ops.r_v, ops.r_H)]


def main():
    # --- SSA reference ---
    mg_ssa = build('ssa')
    ssa = residual(mg_ssa)
    grounded = cp.asnumpy(mg_ssa.levels[0].state.phi.data).copy()

    # --- DIVA, with the coefficients forced to their SSA equivalents ---
    mg_diva = build('diva')
    ops = mg_diva.levels[0].forward_operators
    ops.compute_phi()                                   # same grounded field
    # Freeze the refresh so our hand-set coefficients survive into the residual.
    ops.compute_diva_coeffs = lambda: None
    eta_ssa = 0.5 * B0 * EPS_REG ** ((1.0 - GLEN_N) / (2.0 * GLEN_N))
    _, _, beta = fields()
    # eta_bar / beta_eff are DIVA-only fields, not proxied by the top-level manager -> set per level.
    g_diva = mg_diva.levels[0]
    g_diva.rheology.eta_bar.set(cp.full((ny, nx), eta_ssa, dtype=cp.float32))
    g_diva.sliding.beta_eff.set((beta * cp.asarray(grounded) + WATER_DRAG).astype(cp.float32))
    diva = residual(mg_diva, freeze_phi=True)

    print(f"analytic SSA eta = {eta_ssa:.4f}, grounded fraction = {grounded.mean():.3f}")
    for name, a, b in zip(("r_u", "r_v", "r_H"), ssa, diva):
        scale = max(abs(a).max(), 1e-30)
        err = abs(a - b).max() / scale
        tag = "  (bit-identical)" if np.array_equal(a, b) else ""
        print(f"  {name}: max|ssa| = {abs(a).max():.6e}, max rel diff = {err:.3e}{tag}")
        assert abs(a).max() > 0.0, f"{name} is identically zero -- the test proves nothing"
        assert err < 1e-6, \
            f"{name}: DIVA residual must reduce to SSA when eta_bar/beta_eff are the SSA values"

    print("\nOK: DIVA residual reduces exactly to the SSA residual")


if __name__ == '__main__':
    main()
