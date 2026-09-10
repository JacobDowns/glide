"""Finite-difference check of the sliding-parameter gradients, for SSA and DIVA.

Inversion-shaped setup: synthetic observations from a reference parameter set, then the
gradient of the velocity misfit evaluated at a perturbed parameter.

    J(p) = 1/2 * sum_interior [ (u - u_obs)^2 + (v - v_obs)^2 ]

The SSA case is a **control**: GLIDE's SSA adjoint is exact, so agreement there validates
the harness (converged solves, seeded direction, sensible epsilon). Any DIVA discrepancy
can then be attributed to the code actually under test rather than to the test.

Under DIVA the drag gradient comes from the cell-local expression

    dJ/d(beta)_c = W_eta_c * d(eta_bar_c)/d(beta_c) + W_be_c * d(beta_eff_c)/d(beta_c)

The closure derivatives themselves are checked separately and much more sharply in
tests/diva_derivs_test.py; what this test adds is that they are contracted with the right
adjoint-weighted W and assembled into the right thing.

SCOPE (unified tree): beta is checked under BOTH schemes (SSA is the exact-adjoint control).
u_c and m are checked under DIVA only, against finite differences directly, because this tree
makes SSA/MOLHO Weertman-only: u_c (the regularized-Coulomb threshold) exists solely as a DIVA
field, and the SSA Weertman-exponent gradient kernel is not ported.  There is therefore no SSA
control for those two -- but the beta case has already established that the FD harness is sound,
so FD is a trustworthy reference on its own.  All three DIVA gradients flow through the same
cell-local closure contraction (compute_gradient_{beta,u_c,m} -> _diva_param_gradient), whose
per-parameter closure derivatives are validated much more sharply in tests/diva_derivs_test.py.

Unlike tests/grad_test.py the perturbation direction is **seeded**, the forward solves are
cold-started and converged, and per-cell directions are masked to the interior so the
boundary guard bounds cannot confound the comparison.

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

U_C_0 = 100.0
M_0 = 1.0
# beta doubles as tau_max in Coulomb mode, so it is scaled there to give a comparable drag
# magnitude at these speeds (tau_max ~ beta*(|u| + u_c)).
COULOMB_BETA_SCALE = 120.0

# The FD error is NOT monotonic in eps -- two-sided truncation falls like eps^2 while float32
# round-off grows like 1/eps -- so no single step is trustworthy.  Sweep and judge on the best.
# ssa_control: beta has an exact-adjoint SSA control; u_c/m are DIVA-only here (see the docstring).
PARAMS = {
    #        sliding_law  scalar  offset  eps sweep              ssa_control
    'beta': (0.0,         False,  1.3,    (1e-3, 3e-4, 1e-4),    True),
    'u_c':  (1.0,         False,  1.3,    (3.0,  2.0,  1.0),     False),
    'm':    (0.0,         True,   0.7,    (1e-2, 3e-3, 1e-3),    False),
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
    if law > 0.5:                              # Coulomb: beta is tau_max, scaled for a comparable drag
        beta = beta * COULOMB_BETA_SCALE
    return dict(beta=beta,
                u_c=cp.full((ny, nx), U_C_0, dtype=cp.float32),
                m=M_0)


def build(params, law, scheme):
    x = cp.arange(nx, dtype=cp.float32) * dx
    y = cp.arange(ny, dtype=cp.float32) * dx
    X, Y = cp.meshgrid(x, y)
    srf = 1000.0 * cp.ones((ny, nx), dtype=cp.float32) - cp.tan(cp.deg2rad(0.1)) * X + 10000
    bed = srf - 1000.0
    thk = srf - bed
    B = cp.ones((ny, nx), dtype=cp.float32)
    B.fill((1e-16 ** -(1. / 3)) / (RHO_I * GRAV))

    mg = Multigrid(n_levels, ny=ny, nx=nx, dx=dx, stress_scheme=scheme)
    mg.geometry.bed.set(bed)
    mg.rheology.B.set(B)
    mg.state.H.set(thk)
    mg.state.H_prev.set(thk)
    mg.sliding.sliding_law.set(law)
    mg.sliding.beta.set(params['beta'])
    if mg.levels[0].sliding.u_c is not None:  # u_c is a DIVA-only field (None under SSA)
        mg.sliding.u_c.set(params['u_c'])
    mg.sliding.m.set(params['m'])
    mg.sliding.u_reg.set(1.0)
    for lvl in mg.levels:                     # DIVA-only, harmless under SSA
        lvl.rheology.n_sigma.set(cp.float32(8.0))
    return mg


def forward(params, law, scheme):
    """Cold-started converged solve. Returns (mg, u, v) with u,v as cupy arrays."""
    mg = build(params, law, scheme)
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


def run(param, scheme, tag):
    law, is_scalar, offset, eps_sweep, _ = PARAMS[param]

    w = interior_mask()
    # objective weights on the velocity grids (facet-centred), from the cell mask
    w_u = cp.asarray(np.pad(w, ((0, 0), (0, 1)), mode='edge'))
    w_v = cp.asarray(np.pad(w, ((0, 1), (0, 0)), mode='edge'))

    # synthetic observations from the reference parameters, then evaluate the gradient at a
    # parameter set offset in the one component under test
    p_true = baseline(law)
    _, u_obs, v_obs = forward(p_true, law, scheme)

    p_0 = dict(p_true)
    p_0[param] = p_true[param] * offset

    # seeded perturbation direction: interior-only for a field, unity for a scalar
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

    mg, u0, v0 = forward(p_0, law, scheme)
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
    adjoint.vanka_options.newton_options.momentum_damping.set(cp.float32(0.1))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        adjoint.solve(dt)
    cycles = [l for l in buf.getvalue().splitlines() if "V-cycle" in l]
    adj_res = float(cycles[-1].split("|r|/|r0| =")[1].split(",")[0])

    getattr(grid.adjoint_operators, f'compute_gradient_{param}')()
    if is_scalar:
        gvp_adj = float(getattr(grid.sliding, param).grad)
    else:
        gvp_adj = float(np.sum(cp.asnumpy(getattr(grid.sliding, param).grad) * d))

    print(f"{tag}: J0 = {J0:.6e}, adjoint |r|/|r0| = {adj_res:.2e}")
    best = (np.inf, None, None)
    for eps in eps_sweep:
        _, up, vp = forward(perturbed(+eps), law, scheme)
        _, um, vm = forward(perturbed(-eps), law, scheme)
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


# Gross-breakage bound, applied to both schemes.  Deliberately loose: its job is to catch a
# gradient that is wrong in sign or magnitude, not to measure the FD floor.
SANITY = 1e-2
# DIVA must not be materially worse than the exact-adjoint SSA control on the same problem.
DIVA_VS_CONTROL = 3.0
DIVA_FLOOR = 1e-3


def main():
    results = {}
    for param in PARAMS:
        law = 'Coulomb' if PARAMS[param][0] > 0.5 else 'Weertman'
        has_control = PARAMS[param][4]

        rel_ssa = None
        if has_control:
            # Control first: the SSA adjoint is exact, so this measures the harness.
            rel_ssa = run(param, 'ssa', f"dJ/d({param}) {law}  SSA  (control, exact adjoint)")
            assert rel_ssa < SANITY, \
                f"harness or SSA dJ/d({param}) is broken: rel diff {rel_ssa:.3e}"

        rel_diva = run(param, 'diva', f"dJ/d({param}) {law}  DIVA (under test)")
        assert rel_diva < SANITY, \
            f"DIVA dJ/d({param}) is grossly wrong: rel diff {rel_diva:.3e}"
        if has_control:
            # Judged against the control, not an absolute number: both share the same FD reference.
            bound = max(DIVA_VS_CONTROL * rel_ssa, DIVA_FLOOR)
            assert rel_diva < bound, (
                f"DIVA dJ/d({param}) is materially worse than the exact-adjoint control: "
                f"{rel_diva:.3e} vs control {rel_ssa:.3e} (bound {bound:.3e})")
        results[param] = (rel_ssa, rel_diva)
        print()

    print("summary (best relative difference vs finite differences):")
    for param, (rel_ssa, rel_diva) in results.items():
        if rel_ssa is None:
            print(f"  dJ/d({param}):  DIVA {rel_diva:.3e}   (DIVA-only; no SSA control)")
        else:
            flag = "DIVA better" if rel_diva <= rel_ssa else f"{rel_diva / rel_ssa:.1f}x control"
            print(f"  dJ/d({param}):  SSA control {rel_ssa:.3e}   DIVA {rel_diva:.3e}   ({flag})")
    print("\nOK: every DIVA sliding-parameter gradient (beta, u_c, m) agrees with finite\n"
          "    differences; beta additionally matches the exact-adjoint SSA control")


if __name__ == '__main__':
    main()
