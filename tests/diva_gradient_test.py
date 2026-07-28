"""Finite-difference check of dJ/d(beta), for SSA and DIVA.

Inversion-shaped setup: synthetic observations from a reference beta, then the gradient
of the velocity misfit evaluated at a different beta.

    J(beta) = 1/2 * sum_interior [ (u - u_obs)^2 + (v - v_obs)^2 ]

The SSA case is a **control**: GLIDE's SSA adjoint is exact, so agreement there validates
the harness (converged solves, seeded direction, sensible epsilon). Any DIVA discrepancy
can then be attributed to the approximation actually under test rather than to the test.

The DIVA adjoint is currently **frozen-coefficient**: eta_bar and beta_eff are held fixed
with respect to velocity, so lambda solves an approximate transposed system. The parameter
sensitivity d(beta_eff)/d(beta) *is* exact (implicit function theorem through the closure).
So the DIVA number is expected to be close but not exact, and the size of the gap is the
motivation for adding the exact d(eta_bar)/du and closure-through-velocity paths.

Unlike tests/grad_test.py the perturbation direction is **seeded**, the forward solves are
cold-started and converged, and the direction is masked to the interior so the boundary
guard bounds (see notes/open_questions.md Q2) cannot confound the comparison.

    uv run python tests/diva_gradient_test.py
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

# Perturbation scale. A Taylor sweep (3e-4, 1e-4, 3e-5) shows the FD error is NOT
# monotonic in eps -- float32 round-off scales like 1/eps and dominates below ~1e-4 --
# so 3e-4 is the best-conditioned point. There the SSA control reaches 1.4e-4, which is
# what establishes the harness; DIVA reaches ~1e-3, and that gap is the
# frozen-coefficient error (~0.1%).
EPS = 3e-4
SEED = 12345


def reference_beta():
    x = cp.arange(nx, dtype=cp.float32) * dx
    y = cp.arange(ny, dtype=cp.float32) * dx
    X, Y = cp.meshgrid(x, y)
    return (1000 * cp.sin(2 * cp.pi * X / L) * cp.sin(2 * cp.pi * Y / L) + 1000) / (RHO_I * GRAV)


def interior_mask():
    """Weights for the objective and the perturbation direction: strictly interior, so
    Dirichlet rows and the boundary guard bounds play no part."""
    w = np.zeros((ny, nx), dtype=np.float32)
    w[6:-6, 6:-6] = 1.0
    return w


def build(beta, stress_balance):
    x = cp.arange(nx, dtype=cp.float32) * dx
    y = cp.arange(ny, dtype=cp.float32) * dx
    X, Y = cp.meshgrid(x, y)
    srf = 1000.0 * cp.ones((ny, nx), dtype=cp.float32) - cp.tan(cp.deg2rad(0.1)) * X + 10000
    bed = srf - 1000.0
    thk = srf - bed
    B = cp.ones((ny, nx), dtype=cp.float32)
    B.fill((1e-16 ** -(1. / 3)) / (RHO_I * GRAV))

    mg = Multigrid(n_levels, ny=ny, nx=nx, dx=dx)
    mg.geometry.bed.set(bed)
    mg.rheology.B.set(B)
    mg.state.H.set(thk)
    mg.state.H_prev.set(thk)
    mg.sliding.beta.set(beta)
    mg.sliding.m.set(1.0)
    mg.sliding.u_reg.set(1.0)
    mg.rheology.stress_balance.set(stress_balance)
    mg.rheology.n_sigma.set(8.0)
    return mg


def forward(beta, stress_balance):
    """Cold-started converged solve. Returns (mg, u, v) with u,v as cupy arrays."""
    mg = build(beta, stress_balance)
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
    with contextlib.redirect_stdout(io.StringIO()):
        solver.solve(dt)
    grid = mg.levels[0]
    return mg, grid.state.u.data.copy(), grid.state.v.data.copy()


def objective(u, v, u_obs, v_obs, w_u, w_v):
    du = (u - u_obs) * w_u
    dv = (v - v_obs) * w_v
    return 0.5 * float(cp.sum(du * du) + cp.sum(dv * dv))


def run(stress_balance, tag):
    beta_true = reference_beta()
    w = interior_mask()
    # objective weights on the velocity grids (facet-centred), from the cell mask
    w_u = cp.asarray(np.pad(w, ((0, 0), (0, 1)), mode='edge'))
    w_v = cp.asarray(np.pad(w, ((0, 1), (0, 0)), mode='edge'))

    # synthetic observations, then evaluate the gradient at a different beta
    _, u_obs, v_obs = forward(beta_true, stress_balance)
    beta_0 = beta_true * 1.3

    mg, u0, v0 = forward(beta_0, stress_balance)
    grid = mg.levels[0]
    J0 = objective(u0, v0, u_obs, v_obs, w_u, w_v)

    # adjoint right-hand side: f = -dJ/dx
    grid.adjoint_operators.f_u[:, :] = -((u0 - u_obs) * w_u * w_u)
    grid.adjoint_operators.f_v[:, :] = -((v0 - v_obs) * w_v * w_v)
    grid.adjoint_operators.f_H.fill(0.0)

    adjoint = FASAdjointSolver(mg)
    adjoint.fas_options.coarsest_steps.set(200)
    adjoint.fas_options.pre_steps.set(10)
    adjoint.fas_options.post_steps.set(50)
    adjoint.fas_options.finest_steps.set(150)
    adjoint.fas_options.maximum_vcycles.set(30)
    adjoint.fas_options.relative_tolerance.set(1e-7)
    adjoint.fas_options.absolute_tolerance.set(1e-7)
    adjoint.vanka_options.omega.set(0.5)
    adjoint.vanka_options.newton_options.ssa_damping.set(cp.float32(0.1))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        adjoint.solve(dt)
    cycles = [l for l in buf.getvalue().splitlines() if "V-cycle" in l]
    adj_res = float(cycles[-1].split("|r|/|r0| =")[1].split(",")[0])

    grid.adjoint_operators.compute_gradient_beta()
    grad = cp.asnumpy(grid.sliding.beta.grad)

    # seeded, interior-only direction
    rng = np.random.RandomState(SEED)
    d = (rng.randn(ny, nx) * w).astype(np.float32)
    d_cp = cp.asarray(d)

    gvp_adj = float(np.sum(grad * d))

    _, up, vp = forward(beta_0 + EPS * d_cp, stress_balance)
    _, um, vm = forward(beta_0 - EPS * d_cp, stress_balance)
    Jp = objective(up, vp, u_obs, v_obs, w_u, w_v)
    Jm = objective(um, vm, u_obs, v_obs, w_u, w_v)
    gvp_fd = (Jp - Jm) / (2 * EPS)

    rel = abs(gvp_fd - gvp_adj) / max(abs(gvp_fd), 1e-30)
    print(f"{tag}: J0 = {J0:.6e}, adjoint |r|/|r0| = {adj_res:.2e}")
    print(f"      FD = {gvp_fd:+.6e}   adjoint = {gvp_adj:+.6e}   rel diff = {rel:.3e}")
    return rel


def main():
    # Control: the SSA adjoint is exact, so this validates the harness itself.
    rel_ssa = run(0.0, "SSA  (control, exact adjoint)")
    assert rel_ssa < 5e-3, \
        f"harness or SSA adjoint is broken: rel diff {rel_ssa:.3e} (expected ~1e-4)"

    # Under test: frozen-coefficient DIVA adjoint.
    rel_diva = run(1.0, "DIVA (frozen-coefficient adjoint)")
    # Loose bound: catches sign errors, missing chain factors and gross plumbing bugs,
    # while tolerating the deliberately omitted velocity paths.
    assert rel_diva < 1e-2, \
        f"DIVA gradient is further from FD than the frozen approximation explains: {rel_diva:.3e}"

    print(f"\nSSA control {rel_ssa:.3e} | DIVA {rel_diva:.3e}")
    print("OK: dJ/dbeta agrees with finite differences (SSA exactly; DIVA up to the "
          "frozen-coefficient approximation)")


if __name__ == '__main__':
    main()
