"""Deterministic SSA regression oracle.

Guards the existing SSA solution paths -- Weertman (sliding_law=0) and regularized
Coulomb (sliding_law=1) -- against unintended numerical change while the DIVA stress
balance is added. The DIVA work edits the shared momentum kernels (stress.cu,
residuals.cu, vanka.cu), so every one of those edits must leave these solves alone.

Unlike tests/grad_test.py this is a *reproducibility* check, not a gradient check:
grad_test draws an unseeded random perturbation direction, so its reported error
legitimately varies from run to run and it cannot detect a regression. Here the setup
is fully deterministic (cold start, converged to the float32 residual floor), so a
repeat run reproduces the velocity field bit-for-bit on the same machine.

    uv run python tests/ssa_regression_test.py            # compare to the reference
    uv run python tests/ssa_regression_test.py --write    # (re)generate the reference

The reference (tests/ssa_reference.json) stores a fingerprint per field rather than the
raw arrays: a sha256 of the bytes, which detects any bit-level change, plus max/L2/sum
so a mismatch can be judged by magnitude instead of just flagged. Regenerate it only
when a change to the SSA physics is *intended*, and say so in the commit message.
"""
import hashlib
import json
import sys
from pathlib import Path

import cupy as cp
import numpy as np

from glide.multigrid import Multigrid, FASCDSolver

REFERENCE = Path(__file__).parent / 'ssa_reference.json'

# Tolerance on the max relative velocity difference. Cold-started solves are
# bit-identical on a fixed machine/toolkit, so this only absorbs recompilation or
# driver differences -- it is far tighter than any real physics change would be.
RTOL = 1e-5

L = 20000.0                      # beta oscillation wavelength
dt = cp.float32(1.0)
ny = nx = 128
dx = 2000.0
n_levels = 5

RHO_I = cp.float32(917.0)
GRAV = cp.float32(9.81)


def build_geometry():
    """A 1 km slab on a 0.1 degree slope with an oscillating basal drag field."""
    x = cp.arange(nx, dtype=cp.float32) * dx
    y = cp.arange(ny, dtype=cp.float32) * dx
    X, Y = cp.meshgrid(x, y)

    srf = 1000.0 * cp.ones((ny, nx), dtype=cp.float32) - cp.tan(cp.deg2rad(0.1)) * X + 10000
    bed = srf - 1000.0
    thk = srf - bed

    # Strictly positive, so the drag never vanishes anywhere on the domain.
    beta = (1000 * cp.sin(2 * cp.pi * X / L) * cp.sin(2 * cp.pi * Y / L) + 1000) / (RHO_I * GRAV)

    B = cp.ones((ny, nx), dtype=cp.float32)
    B.fill((1e-16 ** -(1. / 3)) / (RHO_I * GRAV))

    return bed, thk, beta, B


def solve(sliding_law):
    """Cold-started, converged solve. Returns (u, v, H) as numpy arrays.

    Cold start (a fresh Multigrid, zero initial velocity) is what makes this
    reproducible: every call walks the identical iteration path.
    """
    bed, thk, beta, B = build_geometry()

    mg = Multigrid(n_levels, ny=ny, nx=nx, dx=dx)
    mg.geometry.bed.set(bed)
    mg.rheology.B.set(B)
    mg.state.H.set(thk)
    mg.state.H_prev.set(thk)

    mg.sliding.sliding_law.set(sliding_law)
    mg.sliding.u_reg.set(1.0)
    if sliding_law < 0.5:
        mg.sliding.beta.set(beta)                  # Weertman drag coefficient
        mg.sliding.m.set(1.0)
    else:
        # beta doubles as tau_max in Coulomb mode; scale it so the drag magnitude is
        # comparable to the Weertman case at these speeds (tau_max ~ beta*(|u| + u_c)).
        mg.sliding.beta.set(beta * 120.0)
        mg.sliding.u_c.set(cp.full((ny, nx), 100.0, dtype=cp.float32))

    solver = FASCDSolver(mg)
    solver.vanka_options.omega.set(0.5)
    solver.vanka_options.newton_options.relaxation.set(0.5)
    solver.vanka_options.newton_options.steps.set(30)
    solver.fas_options.coarsest_steps.set(200)
    solver.fas_options.pre_steps.set(10)
    solver.fas_options.post_steps.set(50)
    solver.fas_options.finest_steps.set(150)
    solver.fas_options.maximum_vcycles.set(15)
    solver.fas_options.relative_tolerance.set(1e-6)
    solver.fas_options.absolute_tolerance.set(1e-6)
    solver.solve(dt)

    grid = mg.levels[0]
    return (cp.asnumpy(grid.state.u.data),
            cp.asnumpy(grid.state.v.data),
            cp.asnumpy(grid.state.H.data))


def fingerprint(arr):
    """Hash plus summary norms. The hash catches any bit-level change; the norms say
    how big the change is when it happens."""
    return {'sha256': hashlib.sha256(arr.tobytes()).hexdigest(),
            'max': float(np.abs(arr).max()),
            'l2': float(np.linalg.norm(arr.astype(np.float64))),
            'sum': float(arr.astype(np.float64).sum())}


def compare(name, got, ref):
    """Worst relative difference across the summary norms (0 if bit-identical)."""
    if got['sha256'] == ref['sha256']:
        print(f"  {name:<18} bit-identical")
        return 0.0
    err = max(abs(got[k] - ref[k]) / max(abs(ref[k]), 1e-30) for k in ('max', 'l2', 'sum'))
    print(f"  {name:<18} CHANGED: worst norm rel diff = {err:.3e}")
    return err


def main(write=False):
    # The non-destructive guarantee: DIVA must never be the default.
    mg = Multigrid(2, ny=16, nx=16, dx=dx)
    assert all(float(g.rheology.stress_balance.value) == 0.0 for g in mg.levels), \
        "stress_balance must default to 0 (SSA) on every level"

    results = {}
    for tag, law in (('weertman', 0.0), ('coulomb', 1.0)):
        u, v, H = solve(law)
        for fld, arr in (('u', u), ('v', v), ('H', H)):
            results[f'{tag}_{fld}'] = fingerprint(arr)
        print(f"{tag}: max|u| = {np.abs(u).max():.4f} m/a, max|v| = {np.abs(v).max():.4f} m/a")

    if write:
        REFERENCE.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote reference: {REFERENCE}")
        return

    assert REFERENCE.exists(), (
        f"missing reference {REFERENCE}; regenerate with "
        f"`uv run python tests/ssa_regression_test.py --write`")
    ref = json.loads(REFERENCE.read_text())

    print("\nagainst reference:")
    worst = 0.0
    for tag in ('weertman', 'coulomb'):
        for fld in ('u', 'v', 'H'):
            key = f'{tag}_{fld}'
            worst = max(worst, compare(key, results[key], ref[key]))

    print(f"\nworst max rel diff = {worst:.3e} (tolerance {RTOL:.0e})")
    assert worst < RTOL, f"SSA solution changed: {worst:.3e} exceeds {RTOL:.0e}"
    print("OK: SSA (Weertman + regularized Coulomb) unchanged")


if __name__ == '__main__':
    main(write='--write' in sys.argv)
