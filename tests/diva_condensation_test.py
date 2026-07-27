"""The augmented U_b unknown, and the static condensation that eliminates it.

DIVA carries the basal speed as a sixth local unknown, giving the block system

    [ A    b ] [ du   ]   [ r    ]
    [ c^T  d ] [ dU_b ] = [ r_Ub ]

whose (2,2) block is the scalar d = 1 + f'(U_b)*F2 >= 1. Because that block is 1x1 and
never singular, the smoother eliminates U_b analytically instead of solving a 6x6:

    (A - b c^T/d) du = r - b*r_Ub/d ,    dU_b = (r_Ub - c.du)/d .

Two things are checked:

  1. the condensation is exact -- on random systems the condensed 5x5 solve must
     reproduce the full 6x6 solve (this is what lu_6x6_solve is for);
  2. it matters only for nonlinear sliding. For a linear law c'(U) = 0, so
     b = dr/dU_b vanishes identically and the condensation is a no-op: secant and
     tangent drag coincide. For regularized Coulomb they differ, and the solve must
     still converge.

    uv run python tests/diva_condensation_test.py
"""
import contextlib
import io
from pathlib import Path

import cupy as cp
import numpy as np

from glide.multigrid import Multigrid, FASCDSolver

CUDA = Path(__file__).resolve().parents[1] / "glide" / "cuda"
CUDA_FILES = ['common.cu', 'viscosity.cu', 'stress.cu', 'flux.cu',
              'residuals.cu', 'vanka.cu', 'grad.cu', 'diva.cu']

PROBE = r'''
extern "C" __global__
void condense_probe(const float* A6, const float* rhs6,
                    float* x_full, float* x_cond, int nsys)
{
    int s = blockIdx.x*blockDim.x + threadIdx.x;
    if (s >= nsys) return;

    const float* A   = A6   + 36*s;
    const float* rhs = rhs6 + 6*s;

    lu_6x6_solve(A, rhs, x_full + 6*s);

    float A5[25], r5[5], bv[5], cv[5];
    for (int a = 0; a < 5; ++a) {
        for (int b = 0; b < 5; ++b) A5[a*5 + b] = A[a*6 + b];
        bv[a] = A[a*6 + 5];
        cv[a] = A[5*6 + a];
        r5[a] = rhs[a];
    }
    float d     = A[5*6 + 5];
    float r_Ub  = rhs[5];
    float inv_d = 1.0f/d;

    for (int a = 0; a < 5; ++a) {
        r5[a] -= bv[a]*r_Ub*inv_d;
        for (int b = 0; b < 5; ++b) A5[a*5 + b] -= bv[a]*cv[b]*inv_d;
    }

    float du[5];
    lu_5x5_solve(A5, r5, du);

    float dot = 0.0f;
    for (int a = 0; a < 5; ++a) dot += cv[a]*du[a];
    for (int a = 0; a < 5; ++a) x_cond[6*s + a] = du[a];
    x_cond[6*s + 5] = (r_Ub - dot)*inv_d;
}
'''


def test_condensation_is_exact(nsys=200):
    src = "\n".join((CUDA / f).read_text() for f in CUDA_FILES)
    mod = cp.RawModule(code=src + PROBE, options=("--use_fast_math",))

    rng = np.random.RandomState(0)
    A = rng.randn(nsys, 6, 6).astype(np.float32) + np.eye(6, dtype=np.float32) * 12.0
    A[:, 5, 5] = 1.0 + np.abs(rng.randn(nsys))      # d = 1 + f'*F2 >= 1, as in the real block
    rhs = rng.randn(nsys, 6).astype(np.float32)

    x_full = cp.zeros((nsys, 6), dtype=cp.float32)
    x_cond = cp.zeros((nsys, 6), dtype=cp.float32)
    mod.get_function("condense_probe")((nsys // 64 + 1,), (64,),
        (cp.asarray(A.reshape(-1)), cp.asarray(rhs.reshape(-1)), x_full, x_cond, nsys))

    f, c = cp.asnumpy(x_full), cp.asnumpy(x_cond)
    err = np.abs(f - c).max() / max(np.abs(f).max(), 1e-30)
    print(f"[condensation] full 6x6 vs condensed 5x5: max rel diff = {err:.3e} ({nsys} systems)")
    assert err < 1e-4, "static condensation does not reproduce the 6x6 solve"


def solve(sliding_law, vcycles=15):
    """Converged DIVA solve on the standard slab under the requested sliding law."""
    ny = nx = 128
    dx = 2000.0
    L = 20000.0
    rho_i, grav = cp.float32(917.0), cp.float32(9.81)

    x = cp.arange(nx, dtype=cp.float32) * dx
    y = cp.arange(ny, dtype=cp.float32) * dx
    X, Y = cp.meshgrid(x, y)
    srf = 1000.0 * cp.ones((ny, nx), dtype=cp.float32) - cp.tan(cp.deg2rad(0.1)) * X + 10000
    bed = srf - 1000.0
    thk = srf - bed
    beta = (1000 * cp.sin(2 * cp.pi * X / L) * cp.sin(2 * cp.pi * Y / L) + 1000) / (rho_i * grav)
    B = cp.ones((ny, nx), dtype=cp.float32)
    B.fill((1e-16 ** -(1. / 3)) / (rho_i * grav))

    mg = Multigrid(5, ny=ny, nx=nx, dx=dx)
    mg.geometry.bed.set(bed)
    mg.rheology.B.set(B)
    mg.state.H.set(thk)
    mg.state.H_prev.set(thk)
    mg.rheology.stress_balance.set(1.0)
    mg.rheology.n_sigma.set(8.0)
    mg.sliding.u_reg.set(1.0)
    mg.sliding.sliding_law.set(sliding_law)
    if sliding_law < 0.5:
        mg.sliding.beta.set(beta)
        mg.sliding.m.set(1.0)
    else:
        mg.sliding.beta.set(beta * 120.0)                                  # tau_max
        mg.sliding.u_c.set(cp.full((ny, nx), 100.0, dtype=cp.float32))

    solver = FASCDSolver(mg)
    solver.vanka_options.omega.set(0.5)
    solver.vanka_options.newton_options.relaxation.set(0.5)
    solver.vanka_options.newton_options.steps.set(30)
    solver.fas_options.coarsest_steps.set(200)
    solver.fas_options.pre_steps.set(10)
    solver.fas_options.post_steps.set(50)
    solver.fas_options.finest_steps.set(150)
    solver.fas_options.maximum_vcycles.set(vcycles)
    solver.fas_options.relative_tolerance.set(1e-6)
    solver.fas_options.absolute_tolerance.set(1e-6)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        solver.solve(dt=cp.float32(1.0))
    cycles = [l for l in buf.getvalue().splitlines() if "V-cycle" in l]
    residual = float(cycles[-1].split("|r|/|r0| =")[1].split(",")[0])

    grid = mg.levels[0]
    return residual, cp.asnumpy(grid.state.u.data), cp.asnumpy(grid.state.u_b.data)


def main():
    test_condensation_is_exact()

    r_lin, u_lin, ub_lin = solve(0.0)
    print(f"[linear ]      |r|/|r0| = {r_lin:.2e}, max|u| = {abs(u_lin).max():.4f} m/a, "
          f"u_b max = {ub_lin.max():.4f}")

    r_cou, u_cou, ub_cou = solve(1.0)
    print(f"[coulomb]      |r|/|r0| = {r_cou:.2e}, max|u| = {abs(u_cou).max():.4f} m/a, "
          f"u_b max = {ub_cou.max():.4f}")

    assert np.isfinite(u_cou).all(), "Coulomb DIVA produced non-finite velocities"
    assert r_cou < 1e-2, f"Coulomb DIVA failed to converge (|r|/|r0| = {r_cou:.2e})"
    assert ub_cou.min() >= 0.0, "basal speed must be non-negative"

    print("\nOK: condensation is exact, and DIVA converges for a nonlinear sliding law")


if __name__ == '__main__':
    main()
