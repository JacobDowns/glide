"""dJ/d(beta) for an objective built on SURFACE velocity, under DIVA.

Real inversions compare the model against observed SURFACE velocity. GLIDE has always used the
depth-averaged velocity as the counterpart, which is exact for SSA because the two coincide. Under
DIVA they differ by the vertical shear -- ~6% on ISMIP-HOM C, more in slow interior ice -- so the
objective has to be built on u_s, and that changes the adjoint in two places:

    dJ/dp = dJ/dp|explicit + lambda^T dr/dp ,      (dr/dx)^T lambda = -dJ/dx

  * dJ/dx = (dJ/du_s)(du_s/dx) changes the RIGHT-HAND SIDE. It is a cell-to-facet scatter of
    dus_deps and dus_dU through the analytic partials of eps_mem^2 and Ubar -- the same scatter
    the coefficient transpose uses, with A and B redefined.
  * dJ/dp|explicit is entirely NEW. u_s depends on beta, u_c and m directly through the closure,
    which ubar never did, so the usual lambda^T dr/dp is no longer the whole gradient.

THE SSA CONTROL IS THE POINT OF THIS TEST, and it comes in two forms.  Jake's suggestion, and it
is much sharper than finite differences -- it catches a sign error, a double count, or a term
applied to the wrong scheme, none of which an FD comparison at the 1e-3 level could resolve.

  * STRUCTURAL: under SSA both new pieces must be INACTIVE, returning None rather than a small
    number.  That is the difference between "this path is switched off" and "this path happens to
    be small in the configuration I tried".

  * STIFF LIMIT: the sharp one.  Stiffening the ice removes the vertical shear (it scales like
    B^-n), so u_s -> |ubar| and the explicit term -> 0.  Both schemes then compute the gradient of
    the IDENTICAL functional of beta -- the observations are taken from the SSA run so it is the
    same function, not merely the same formula -- one of them exercising the new scatter and the
    explicit term, the other using neither.  They must converge.  Measured:

        B scale        1        4       16       64      256
        |dgrad|/|grad| 1.7e-1  5.4e-2  1.7e-2  5.4e-3  2.0e-3

    monotone with no plateau, which is what says the new path is right.  It falls somewhat slower
    than the shear itself (a factor ~3.1 per step against ~4.1) -- expected, since this is a
    max-norm of a difference that the adjoint solve has propagated nonlocally, not a pointwise
    comparison of two local quantities.  A residual error in the new path would show up as a floor
    instead, since as the shear vanishes DIVA's scatter reduces term by term to the plain
    d|ubar|/d(facet) one that the SSA branch builds independently.

Then the DIVA case is checked against finite differences of the surface objective, judged against
the SSA control as in diva_gradient_test (see notes/open_questions.md Q7 for why an absolute bound
is not the right target).

    uv run python tests/diva_surface_gradient_test.py
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
EPS_SWEEP = (1e-3, 3e-4, 1e-4)
SEED = 12345


def interior_mask():
    w = np.zeros((ny, nx), dtype=np.float32)
    w[6:-6, 6:-6] = 1.0
    return w


def build(beta, stress_balance, B_scale=1.0):
    x = cp.arange(nx, dtype=cp.float32) * dx
    y = cp.arange(ny, dtype=cp.float32) * dx
    X, Y = cp.meshgrid(x, y)
    srf = 1000.0 * cp.ones((ny, nx), dtype=cp.float32) - cp.tan(cp.deg2rad(0.1)) * X + 10000
    bed = srf - 1000.0
    B = cp.ones((ny, nx), dtype=cp.float32)
    B.fill(B_scale * (1e-16 ** -(1. / 3)) / (RHO_I * GRAV))

    mg = Multigrid(n_levels, ny=ny, nx=nx, dx=dx)
    mg.geometry.bed.set(bed)
    mg.rheology.B.set(B)
    mg.state.H.set(srf - bed)
    mg.state.H_prev.set(srf - bed)
    mg.sliding.beta.set(beta)
    mg.sliding.m.set(1.0)
    mg.sliding.u_reg.set(1.0)
    mg.rheology.stress_balance.set(stress_balance)
    mg.rheology.n_sigma.set(8.0)
    return mg


def forward(beta, stress_balance, B_scale=1.0):
    mg = build(beta, stress_balance, B_scale)
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
    return mg


def surface_speed(mg, stress_balance):
    """The model's surface speed, cell-centred.  Under SSA there is no shear, so it is the
    cell-centred depth-averaged speed; under DIVA it is the closure's u_s."""
    g = mg.levels[0]
    if stress_balance > 0.5:
        return g.state.u_s.data
    u, v = g.state.u.data, g.state.v.data
    return cp.hypot(0.5 * (u[:, :-1] + u[:, 1:]), 0.5 * (v[:-1, :] + v[1:, :]))


def objective_and_cotangent(mg, obs, w, stress_balance):
    """J = 1/2 sum w*(u_s - obs)^2, and cot = dJ/d(u_s) = w*(u_s - obs)."""
    u_s = surface_speed(mg, stress_balance)
    d = (u_s - obs) * w
    return 0.5 * float(cp.sum(d * d)), d * w


def gradient(mg, obs, w, stress_balance):
    """dJ/d(beta) for the surface objective: the adjoint term plus the explicit term."""
    g = mg.levels[0]
    ao = g.adjoint_operators
    J, cot = objective_and_cotangent(mg, obs, w, stress_balance)

    if stress_balance > 0.5:
        ao.diva_surface_misfit_rhs(cot)              # fills f_u, f_v = -dJ/d(u,v)
    else:
        # SSA: u_s == cell-centred |ubar|, so scatter the cotangent to the facets by hand --
        # d|ubar|/du_l = 0.5*u_ctr/|ubar| and so on, the same averaging DIVA's kernel applies.
        u, v = g.state.u.data, g.state.v.data
        u_ctr = 0.5 * (u[:, :-1] + u[:, 1:])
        v_ctr = 0.5 * (v[:-1, :] + v[1:, :])
        spd = cp.maximum(cp.hypot(u_ctr, v_ctr), 1e-30)
        cu, cv = cot * 0.5 * u_ctr / spd, cot * 0.5 * v_ctr / spd
        ao.f_u.fill(0); ao.f_v.fill(0)
        ao.f_u[:, :-1] -= cu; ao.f_u[:, 1:] -= cu
        ao.f_v[:-1, :] -= cv; ao.f_v[1:, :] -= cv
    ao.f_H.fill(0.0)

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
    with contextlib.redirect_stdout(io.StringIO()):
        adjoint.solve(dt)

    ao.compute_gradient_beta()
    grad = cp.asnumpy(g.sliding.beta.grad).copy()
    explicit = ao.diva_surface_param_gradient(cot, 'beta')
    if explicit is not None:
        grad += cp.asnumpy(explicit)
    return J, grad


def reference_beta():
    x = cp.arange(nx, dtype=cp.float32) * dx
    y = cp.arange(ny, dtype=cp.float32) * dx
    X, Y = cp.meshgrid(x, y)
    return (1000 * cp.sin(2 * cp.pi * X / L) * cp.sin(2 * cp.pi * Y / L) + 1000) / (RHO_I * GRAV)


def run(stress_balance, tag):
    w = cp.asarray(interior_mask())
    beta_true = reference_beta()
    mg_obs = forward(beta_true, stress_balance)
    obs = surface_speed(mg_obs, stress_balance).copy()

    beta_0 = beta_true * 1.3
    mg = forward(beta_0, stress_balance)
    J0, grad = gradient(mg, obs, w, stress_balance)

    rng = np.random.RandomState(SEED)
    d = (rng.randn(ny, nx) * interior_mask()).astype(np.float32)
    gvp_adj = float(np.sum(grad * d))

    print(f"{tag}: J0 = {J0:.6e}")
    best = (np.inf, None, None)
    for eps in EPS_SWEEP:
        Jp, _ = objective_and_cotangent(forward(beta_0 + eps * cp.asarray(d), stress_balance),
                                        obs, w, stress_balance)
        Jm, _ = objective_and_cotangent(forward(beta_0 - eps * cp.asarray(d), stress_balance),
                                        obs, w, stress_balance)
        fd = (Jp - Jm) / (2 * eps)
        rel = abs(fd - gvp_adj) / max(abs(fd), 1e-30)
        print(f"      eps={eps:<8g} FD = {fd:+.6e}  adjoint = {gvp_adj:+.6e}  rel = {rel:.3e}")
        if rel < best[0]:
            best = (rel, fd, eps)
    print(f"      best: rel diff = {best[0]:.3e} at eps = {best[2]:g}")
    return best[0], grad


def stiff_limit_control():
    """The sharp control: the SAME objective, and DIVA's surface gradient must converge onto
    SSA's as the ice stiffens.

    Stiffening removes the vertical shear (it scales like B^-n), so u_s -> |ubar| and the explicit
    parameter term -> 0. Both schemes then compute the gradient of the identical functional of
    beta, one of them exercising the new cell-to-facet scatter and the explicit term and the other
    using neither. A sign error, double count or misplaced factor in the new path shows up as a gap
    that does not close, because as the shear vanishes DIVA's scatter reduces term by term to the
    d|ubar|/d(facet) scatter the SSA branch builds independently.

    The observations are taken from the SSA run so the two objectives are literally the same
    function of beta, not merely the same formula.

    Only three B scales are run here to keep the test affordable; pushed to B x 256 the gap reaches
    2.0e-3 with no sign of a floor (see the module docstring).
    """
    w = cp.asarray(interior_mask())
    beta_true = reference_beta()
    beta_0 = beta_true * 1.3
    print("stiff-ice control -- the same objective, DIVA's gradient must converge onto SSA's:")
    print(f"  {'B scale':>8} {'|grad diff|/|grad|':>19} {'max|u_s/ubar - 1|':>19}")
    prev = None
    for B_scale in (1.0, 4.0, 16.0):
        obs = surface_speed(forward(beta_true, 0.0, B_scale), 0.0).copy()

        mg_s = forward(beta_0, 0.0, B_scale)
        _, g_ssa = gradient(mg_s, obs, w, 0.0)
        mg_d = forward(beta_0, 1.0, B_scale)
        _, g_div = gradient(mg_d, obs, w, 1.0)

        gd = mg_d.levels[0]
        u, v = gd.state.u.data, gd.state.v.data
        spd = cp.maximum(cp.hypot(0.5*(u[:, :-1] + u[:, 1:]), 0.5*(v[:-1, :] + v[1:, :])), 1e-30)
        shear = float(cp.abs(gd.state.u_s.data / spd - 1.0).max())

        rel = np.abs(g_div - g_ssa).max() / max(np.abs(g_ssa).max(), 1e-30)
        print(f"  {B_scale:8g} {rel:19.3e} {shear:19.3e}")
        if prev is not None:
            assert rel < prev, (
                f"the DIVA surface gradient does not approach the SSA one as the shear vanishes: "
                f"{rel:.3e} at B x {B_scale} is not below {prev:.3e}. That points at the new "
                f"right-hand-side scatter or the explicit parameter term, not at the harness.")
        prev = rel
    assert prev < 2e-2, (
        f"the two gradients have not converged in the stiff limit: {prev:.3e}")
    print()


def main():
    stiff_limit_control()

    # ---- FD check on SSA, which validates the harness ----
    rel_ssa, grad_ssa = run(0.0, "SSA  (control: u_s == |ubar|, both new terms vanish)")
    assert rel_ssa < 1e-2, f"harness broken: SSA surface gradient vs FD is {rel_ssa:.3e}"

    # Under SSA the explicit term must be None, not merely small: it is the difference between
    # "this path is inactive" and "this path happens to be small here".
    mg = forward(reference_beta() * 1.3, 0.0)
    ao = mg.levels[0].adjoint_operators
    assert ao.diva_surface_param_gradient(cp.zeros((ny, nx), cp.float32), 'beta') is None, \
        "the explicit parameter term must be inactive under SSA, where u_s == ubar"
    assert ao.diva_surface_misfit_rhs(cp.zeros((ny, nx), cp.float32))[0] is None, \
        "the surface-misfit right-hand side must be inactive under SSA"
    print("\nSSA control: both new terms correctly inactive (returned None, not zero)\n")

    # ---- under test ----
    rel_diva, grad_diva = run(1.0, "DIVA (surface objective, under test)")
    assert rel_diva < 1e-2, f"DIVA surface gradient is grossly wrong: {rel_diva:.3e}"
    bound = max(3.0 * rel_ssa, 1e-3)
    assert rel_diva < bound, (
        f"DIVA surface gradient is materially worse than the SSA control: {rel_diva:.3e} vs "
        f"{rel_ssa:.3e} (bound {bound:.3e})")

    print(f"\nsummary: SSA control {rel_ssa:.3e}   DIVA {rel_diva:.3e}")
    print("OK: dJ/d(beta) for a surface-velocity objective agrees with finite differences")


if __name__ == '__main__':
    main()
