"""Finite-difference verification of the rheology (B) adjoint gradient.

Builds a small synthetic ice-flow problem, defines a scalar loss J on the
modeled velocity, gets dJ/dB two ways:
  * the new adjoint kernel  (GlideStepB -> compute_gradient_B), and
  * central finite differences  (J(B + e_k) - J(B - e_k)) / (2 e_k),
and compares them at a set of interior cells.

Run:
    PYTHONPATH=/home/jupyter-jdowns/scratch/glide \
        /opt/glamacles/venv/bin/python /home/jupyter-jdowns/scratch/glide/tests/fd_check_B.py
"""
import cupy as cp
import numpy as np
import torch
import pyproj

from glide.model import IceDynamics
from glide.torch import GlideStepB

DEVICE = "cuda"
cp.random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def build_small(ny=64, nx=64, n_levels=3, dx=2000.0):
    """A small, fully grounded dome so the SSA solve is well posed and smooth."""
    yy, xx = np.mgrid[0:ny, 0:nx].astype("float64")
    cy, cx = ny / 2, nx / 2
    r2 = ((yy - cy) / (ny / 2)) ** 2 + ((xx - cx) / (nx / 2)) ** 2
    thk = (1800.0 * np.clip(1.0 - r2, 0.02, 1.0)).astype("float32")   # dome, >0
    bed = (300.0 + 40.0 * np.sin(xx / 6.0) * np.cos(yy / 5.0)).astype("float32")  # above sea level -> grounded

    model = IceDynamics(n_levels=n_levels, ny=ny, nx=nx, dx=dx,
                        x0=0.0, y0=0.0, crs=pyproj.CRS("EPSG:3413"))
    mg = model.mg
    mg.state.H.set(thk); mg.state.H_prev.set(thk)
    mg.geometry.bed.set(bed)
    B0 = 1e-17 ** (-1 / 3) / (917 * 9.81)
    Bfield = cp.full((ny, nx), B0, cp.float32)
    mg.rheology.B.set(Bfield); mg.rheology.eps_reg.set(1e-6); mg.rheology.n.set(3.0)
    beta = cp.full((ny, nx), 3.0, cp.float32)
    mg.sliding.beta.set(beta); mg.sliding.m.set(1 / 3); mg.sliding.water_drag.set(1e-4)
    mg.calving.calving_rate.set(0.0)
    mg.forcing.smb.set(cp.zeros((ny, nx), cp.float32))

    # TIGHT convergence: FD needs J(B) smooth, and the adjoint identity holds only
    # at a converged forward/adjoint state.
    model.forward_solver.fas_options.set(
        coarsest_steps=400, pre_steps=40, post_steps=400, finest_steps=40,
        relative_tolerance=1e-12, absolute_tolerance=1e-6, report_norms=False)
    model.adjoint_solver.fas_options.set(
        coarsest_steps=400, pre_steps=40, post_steps=400, finest_steps=40,
        relative_tolerance=1e-12, absolute_tolerance=1e-8, report_norms=False)
    return model, thk, bed, B0


def main():
    ny = nx = 64
    level = 0
    model, thk, bed, B0 = build_small(ny=ny, nx=nx, n_levels=3)

    # Fixed inputs as torch tensors.
    H_prev = torch.tensor(thk, device=DEVICE)
    bedt = torch.tensor(bed, device=DEVICE)
    beta = torch.full((ny, nx), 3.0, device=DEVICE)
    smb = torch.zeros((ny, nx), device=DEVICE)

    # B field with mild spatial variation so dJ/dB varies cell to cell.
    Bnp = B0 * (1.0 + 0.15 * np.random.randn(ny, nx)).astype("float32")
    B = torch.tensor(Bnp, device=DEVICE)

    # Random loss weights on the staggered velocity components (u:(ny,nx+1), v:(ny+1,nx)).
    Wu = torch.tensor(np.random.randn(ny, nx + 1).astype("float32"), device=DEVICE)
    Wv = torch.tensor(np.random.randn(ny + 1, nx).astype("float32"), device=DEVICE)

    def solve_J(Bt, warm=None):
        """Forward-solve for velocity given B; return (J, u, v) with no autograd."""
        if warm is not None:
            model.mg.state.u.set(cp.asarray(warm[0]), start_level=level)
            model.mg.state.v.set(cp.asarray(warm[1]), start_level=level)
        with torch.no_grad():
            u, v, H, M = GlideStepB.apply(cp.float32(0.0), cp.float32(10.0), model, level,
                                          H_prev, bedt, beta, Bt, smb)
        J = (u * Wu).sum() + (v * Wv).sum()
        return float(J.item()), cp.asarray(u.detach()), cp.asarray(v.detach())

    # --- Base converged solve + adjoint gradient ---
    model.mg.state.u.data.fill(0.0); model.mg.state.v.data.fill(0.0)
    Bg = B.clone().requires_grad_(True)
    u, v, H, M = GlideStepB.apply(cp.float32(0.0), cp.float32(10.0), model, level,
                                  H_prev, bedt, beta, Bg, smb)
    J = (u * Wu).sum() + (v * Wv).sum()
    J.backward()
    g_adj = Bg.grad.detach().cpu().numpy()
    base_u = cp.asarray(u.detach()); base_v = cp.asarray(v.detach())
    print(f"J(base) = {J.item():.6e}   |g_adj| max = {np.abs(g_adj).max():.3e}")

    # --- Finite differences at a set of interior cells ---
    rng = np.random.default_rng(1)
    cells = [(int(rng.integers(8, ny - 8)), int(rng.integers(8, nx - 8))) for _ in range(12)]
    eps = 1e-2 * B0                       # relative step ~1e-2

    print(f"\n{'cell':>12} {'adjoint':>14} {'fd':>14} {'abs_err':>11} {'rel_err':>9}")
    rels = []
    for (i, j) in cells:
        Bp = B.clone(); Bp[i, j] += eps
        Bm = B.clone(); Bm[i, j] -= eps
        Jp, _, _ = solve_J(Bp, warm=(base_u, base_v))
        Jm, _, _ = solve_J(Bm, warm=(base_u, base_v))
        fd = (Jp - Jm) / (2 * eps)
        a = g_adj[i, j]
        abs_err = abs(a - fd)
        rel = abs_err / (abs(fd) + 1e-30)
        rels.append(rel)
        print(f"({i:3d},{j:3d})  {a:14.5e} {fd:14.5e} {abs_err:11.2e} {rel:9.2%}")

    rels = np.array(rels)
    print(f"\nmedian rel err {np.median(rels):.2%}   max rel err {rels.max():.2%}")
    ok = np.median(rels) < 0.05
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    main()
