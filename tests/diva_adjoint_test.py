"""The DIVA adjoint solve.

Checks that the transposed DIVA operator can actually be solved -- i.e. that the DIVA VJP
and the adjoint smoother form a consistent pair and the FAS adjoint cycle converges on it,
with the coefficients recomputed per level from the restricted forward state.

This runs whatever AdjointOperators is configured for, which by default is the **exact**
coefficient transpose -- the d(eta_bar)/du and closure paths included. The smoother remains
the frozen block, which is deliberate: it is only a preconditioner, and the frozen operator
is the one Goldberg proves self-adjoint (see notes/diva_numerics.md 5.8), so it makes a good
one. SSA works the same way -- its VJP carries d(eta)/du and its smoother does not.

Convergence is therefore the thing under test here, not correctness of the transpose;
tests/diva_dotproduct_test.py establishes that separately and much more sharply. Worth
knowing what a failure looks like: an earlier version stalled flat at 1.1e-1 regardless of
omega, which is the signature of contributions landing on Dirichlet rows that the smoother
has replaced with identities -- NOT of a weak preconditioner, which responds to omega.

    uv run python tests/diva_adjoint_test.py
"""
import contextlib
import io

import cupy as cp
import numpy as np

from glide.multigrid import Multigrid, FASCDSolver, FASAdjointSolver

ny = nx = 128
dx = 2000.0
dt = cp.float32(1.0)
n_levels = 5
L = 20000.0
RHO_I = cp.float32(917.0)
GRAV = cp.float32(9.81)


def build(scheme):
    x = cp.arange(nx, dtype=cp.float32) * dx
    y = cp.arange(ny, dtype=cp.float32) * dx
    X, Y = cp.meshgrid(x, y)
    srf = 1000.0 * cp.ones((ny, nx), dtype=cp.float32) - cp.tan(cp.deg2rad(0.1)) * X + 10000
    bed = srf - 1000.0
    thk = srf - bed
    beta = (1000 * cp.sin(2 * cp.pi * X / L) * cp.sin(2 * cp.pi * Y / L) + 1000) / (RHO_I * GRAV)
    B = cp.ones((ny, nx), dtype=cp.float32)
    B.fill((1e-16 ** -(1. / 3)) / (RHO_I * GRAV))

    mg = Multigrid(n_levels, ny=ny, nx=nx, dx=dx, stress_scheme=scheme)
    mg.geometry.bed.set(bed)
    mg.rheology.B.set(B)
    mg.state.H.set(thk)
    mg.state.H_prev.set(thk)
    mg.sliding.beta.set(beta)
    mg.sliding.m.set(1.0)
    mg.sliding.u_reg.set(1.0)
    for lvl in mg.levels:                          # DIVA-only, harmless under SSA
        lvl.rheology.n_sigma.set(cp.float32(8.0))
    return mg


def _tune(solver, vcycles):
    solver.fas_options.coarsest_steps.set(200)
    solver.fas_options.pre_steps.set(10)
    solver.fas_options.post_steps.set(50)
    solver.fas_options.finest_steps.set(150)
    solver.fas_options.maximum_vcycles.set(vcycles)
    solver.fas_options.relative_tolerance.set(1e-6)
    solver.fas_options.absolute_tolerance.set(1e-6)
    solver.vanka_options.omega.set(0.5)


def solve(scheme, vcycles=20):
    mg = build(scheme)
    grid = mg.levels[0]

    forward = FASCDSolver(mg)
    _tune(forward, 15)
    forward.vanka_options.newton_options.relaxation.set(0.5)
    forward.vanka_options.newton_options.steps.set(30)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        forward.solve(dt)
    fwd_cycles = [l for l in buf.getvalue().splitlines() if "V-cycle" in l]
    fwd_res = float(fwd_cycles[-1].split("|r|/|r0| =")[1].split(",")[0])

    # Smooth objective: L2 velocity misfit against a scaled target, so dJ/du is smooth
    # and nonzero over the whole domain.
    grid.adjoint_operators.f_u[:, :] = -(grid.state.u.data - 0.9 * grid.state.u.data)
    grid.adjoint_operators.f_v[:, :] = -(grid.state.v.data - 0.9 * grid.state.v.data)
    grid.adjoint_operators.f_H.fill(0.0)

    adjoint = FASAdjointSolver(mg)
    _tune(adjoint, vcycles)
    adjoint.vanka_options.newton_options.momentum_damping.set(cp.float32(0.1))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        adjoint.solve(dt)
    adj_cycles = [l for l in buf.getvalue().splitlines() if "V-cycle" in l]
    adj_res = float(adj_cycles[-1].split("|r|/|r0| =")[1].split(",")[0])

    return dict(fwd_res=fwd_res, adj_res=adj_res, adj_cycles=len(adj_cycles),
                lam_u=cp.asnumpy(grid.adjoint.lambda_u.data),
                lam_v=cp.asnumpy(grid.adjoint.lambda_v.data))


def main():
    for tag, scheme in (("SSA ", 'ssa'), ("DIVA", 'diva')):
        r = solve(scheme)
        print(f"{tag}: forward |r|/|r0| = {r['fwd_res']:.2e} | "
              f"adjoint |r|/|r0| = {r['adj_res']:.2e} in {r['adj_cycles']} V-cycles | "
              f"max|lambda_u| = {np.abs(r['lam_u']).max():.4e}")
        assert np.isfinite(r['lam_u']).all() and np.isfinite(r['lam_v']).all(), \
            f"{tag}: adjoint produced non-finite lambda"
        assert r['adj_res'] < 1e-4, \
            f"{tag}: adjoint solve failed to converge (|r|/|r0| = {r['adj_res']:.2e})"
        assert np.abs(r['lam_u']).max() > 0.0, f"{tag}: lambda is identically zero"

    print("\nOK: the DIVA adjoint operator is solvable and converges")


if __name__ == '__main__':
    main()
