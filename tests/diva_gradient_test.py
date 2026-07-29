"""Finite-difference check of the sliding-parameter gradients, for SSA and DIVA.

Inversion-shaped setup: synthetic observations from a reference parameter set, then the
gradient of the velocity misfit evaluated at a perturbed parameter.

    J(p) = 1/2 * sum_interior [ (u - u_obs)^2 + (v - v_obs)^2 ]

Three parameters are covered, and each is checked under BOTH stress balances:

    beta  per-cell drag coefficient      Weertman law
    u_c   per-cell Coulomb threshold     regularized Coulomb law (inert under Weertman)
    m     global Weertman exponent       Weertman law (inert under Coulomb)

The SSA case is a **control**: GLIDE's SSA adjoint is exact, so agreement there validates
the harness (converged solves, seeded direction, sensible epsilon). Any DIVA discrepancy
can then be attributed to the code actually under test rather than to the test.

Under DIVA all three gradients come from the same cell-local expression,

    dJ/d(p)_c = W_eta_c * d(eta_bar_c)/d(p_c) + W_be_c * d(beta_eff_c)/d(p_c)

differing only in which pair of derivative fields is used (and, for m, a reduction to a
scalar). The closure derivatives themselves are checked separately and much more sharply
in tests/diva_derivs_test.py; what this test adds is that they are contracted with the
right adjoint-weighted W and assembled into the right thing.

Unlike tests/grad_test.py the perturbation direction is **seeded**, the forward solves are
cold-started and converged, and per-cell directions are masked to the interior so the
boundary guard bounds (see notes/open_questions.md Q2) cannot confound the comparison.

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

SEED = 12345

# Baselines. beta doubles as tau_max in Coulomb mode, so it is scaled there to give a
# comparable drag magnitude at these speeds (tau_max ~ beta*(|u| + u_c)) -- same
# convention as tests/ssa_regression_test.py.
U_C_0 = 100.0
M_0 = 1.0
COULOMB_BETA_SCALE = 120.0

# The FD error is NOT monotonic in eps -- two-sided truncation falls like eps^2 while
# float32 round-off grows like 1/eps -- so no single step is trustworthy.  Sweep and judge
# on the best.  Each sweep is sized to BRACKET that crossover for its own parameter, and
# the whole curve is printed so the bracketing is visible rather than asserted: expect the
# error to fall with eps, bottom out, then rise again.
#
# Sizing has to be per parameter, in RELATIVE terms.  A step that is small in absolute
# terms but tiny relative to the baseline moves J by a few ULPs and gives a meaningless
# reference -- u_c has a baseline of ~130 where beta's is ~0.14, so the same absolute eps
# differs by three orders of magnitude in what it actually perturbs.  eps/baseline is
# printed for each step.
PARAMS = {
    #        sliding_law  scalar  offset from truth   eps sweep
    'beta': (0.0,         False,  1.3,                (1e-3, 3e-4, 1e-4)),
    'u_c':  (1.0,         False,  1.3,                (3.0,  2.0,  1.0)),
    'm':    (0.0,         True,   0.7,                (1e-2, 3e-3, 1e-3)),
}


def reference_beta():
    x = cp.arange(nx, dtype=cp.float32) * dx
    y = cp.arange(ny, dtype=cp.float32) * dx
    X, Y = cp.meshgrid(x, y)
    return (1000 * cp.sin(2 * cp.pi * X / L) * cp.sin(2 * cp.pi * Y / L) + 1000) / (RHO_I * GRAV)


def interior_mask():
    """Weights for the objective and for per-cell perturbation directions: strictly
    interior, so Dirichlet rows and the boundary guard bounds play no part."""
    w = np.zeros((ny, nx), dtype=np.float32)
    w[6:-6, 6:-6] = 1.0
    return w


def baseline(law):
    """The reference ('true') sliding parameters for a given law."""
    beta = reference_beta()
    if law > 0.5:
        beta = beta * COULOMB_BETA_SCALE
    return dict(beta=beta,
                u_c=cp.full((ny, nx), U_C_0, dtype=cp.float32),
                m=M_0)


def build(params, law, stress_balance):
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
    mg.sliding.sliding_law.set(law)
    mg.sliding.beta.set(params['beta'])
    mg.sliding.u_c.set(params['u_c'])
    mg.sliding.m.set(params['m'])
    mg.sliding.u_reg.set(1.0)
    mg.rheology.stress_balance.set(stress_balance)
    mg.rheology.n_sigma.set(8.0)
    return mg


def forward(params, law, stress_balance):
    """Cold-started converged solve. Returns (mg, u, v) with u,v as cupy arrays."""
    mg = build(params, law, stress_balance)
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


def run(param, stress_balance, tag):
    law, is_scalar, offset, eps_sweep = PARAMS[param]

    w = interior_mask()
    # objective weights on the velocity grids (facet-centred), from the cell mask
    w_u = cp.asarray(np.pad(w, ((0, 0), (0, 1)), mode='edge'))
    w_v = cp.asarray(np.pad(w, ((0, 1), (0, 0)), mode='edge'))

    # synthetic observations from the reference parameters, then evaluate the gradient at a
    # parameter set offset in the one component under test
    p_true = baseline(law)
    _, u_obs, v_obs = forward(p_true, law, stress_balance)

    p_0 = dict(p_true)
    p_0[param] = p_true[param] * offset

    # seeded perturbation direction: interior-only for a field, unity for a scalar
    # `scale` is defined so that eps/scale is the size of the perturbation relative to the
    # baseline, which is the quantity that has to be resolvable in float32.
    if is_scalar:
        d = 1.0
        scale = abs(float(p_0[param]))
    else:
        rng = np.random.RandomState(SEED)
        d = (rng.randn(ny, nx) * w).astype(np.float32)
        sup = cp.asarray(w) > 0
        rms = lambda a: float(cp.sqrt(cp.mean(cp.asarray(a)[sup] ** 2)))
        scale = rms(p_0[param] * cp.asarray(w)) / rms(d)

    def perturbed(eps):
        p = dict(p_0)
        p[param] = p_0[param] + eps * (d if is_scalar else cp.asarray(d))
        return p

    mg, u0, v0 = forward(p_0, law, stress_balance)
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

    getattr(grid.adjoint_operators, f'compute_gradient_{param}')()
    if is_scalar:
        gvp_adj = float(grid.sliding.m.grad)
    else:
        gvp_adj = float(np.sum(cp.asnumpy(getattr(grid.sliding, param).grad) * d))

    print(f"{tag}: J0 = {J0:.6e}, adjoint |r|/|r0| = {adj_res:.2e}")
    best = (np.inf, None, None)
    for eps in eps_sweep:
        _, up, vp = forward(perturbed(+eps), law, stress_balance)
        _, um, vm = forward(perturbed(-eps), law, stress_balance)
        Jp = objective(up, vp, u_obs, v_obs, w_u, w_v)
        Jm = objective(um, vm, u_obs, v_obs, w_u, w_v)
        gvp_fd = (Jp - Jm) / (2 * eps)
        rel = abs(gvp_fd - gvp_adj) / max(abs(gvp_fd), 1e-30)
        print(f"      eps={eps:<8g} (={eps / scale:.1e} of baseline)  FD = {gvp_fd:+.6e}  "
              f"adjoint = {gvp_adj:+.6e}  rel diff = {rel:.3e}")
        if rel < best[0]:
            best = (rel, gvp_fd, eps)
    print(f"      best: rel diff = {best[0]:.3e} at eps = {best[2]:g}")
    return best[0]


def main():
    results = {}
    for param in PARAMS:
        law = 'Coulomb' if PARAMS[param][0] > 0.5 else 'Weertman'
        # Control first: the SSA adjoint is exact, so this validates the harness itself.
        rel_ssa = run(param, 0.0, f"dJ/d({param}) {law}  SSA  (control, exact adjoint)")
        assert rel_ssa < 1e-3, \
            f"harness or SSA dJ/d({param}) is broken: rel diff {rel_ssa:.3e}"

        rel_diva = run(param, 1.0, f"dJ/d({param}) {law}  DIVA (under test)")
        # Every parameter path is complete (both the beta_eff and the eta_bar route), so
        # DIVA should sit at the FD floor alongside the SSA control, not above it.
        assert rel_diva < 1e-3, \
            f"DIVA dJ/d({param}) above the finite-difference floor: {rel_diva:.3e}"
        results[param] = (rel_ssa, rel_diva)
        print()

    print("summary (best relative difference vs finite differences):")
    for param, (rel_ssa, rel_diva) in results.items():
        print(f"  dJ/d({param}):  SSA control {rel_ssa:.3e}   DIVA {rel_diva:.3e}")
    print("\nOK: all sliding-parameter gradients agree with finite differences, "
          "for SSA and DIVA")


if __name__ == '__main__':
    main()
