"""Operator-consistency tests for DIVA: JVP against finite differences, then the
adjoint identity against the JVP.

Three checks, in the order that makes them interpretable:

  1. **JVP vs FD of the residual, VELOCITY direction.** Validates compute_jvp_diva itself
     as a faithful Jacobian of compute_residual_diva. Without this, a failure in check 3
     could equally be a JVP bug.

  1b. **JVP vs FD of the residual, THICKNESS direction.** The same check with d_H != 0 and
     d_u = d_v = 0, covering the closure-through-thickness path

         H -> I1, I2 -> U_b, tau_b -> eta_bar, beta_eff -> r

     SSA is the control: there the residual still depends on H explicitly (through
     H*eta_bar, the driving stress and the fluxes) but has no closure path at all.

     This check is COARSE, and it is worth knowing by how much. Disabling the closure's H
     seed and re-running moves it only from 5.7e-5 to 1.2e-4 -- about a factor of two --
     because the residual's explicit H terms dominate its response to thickness, so the
     closure-mediated part is a small correction on top of a large signal. It will catch a
     gross error and not a subtle one. The SHARP checks for this path are elsewhere:
     tests/diva_derivs_test.py finite-differences d(eta_bar)/dH, d(beta_eff)/dH and
     d(u_s)/dH directly (agreement ~1e-4 or better), and check 3 below detects any
     asymmetry between the two operators -- the same probe drove it from 1.3e-7 to 2.6e-5.

     The one thing no combination of these can see is a term dropped from the JVP and the
     VJP simultaneously; that is what the direct closure-derivative test is for.

  2. **Dot-product test**  <J^T lam, x>  vs  <lam, J x>  for seeded random lam, x.
     This is the sharp instrument: it uses **no finite differences**, so it resolves to
     float32 round-off (~1e-7) rather than the ~1e-3 an FD gradient check tops out at.
     It is also not circular when the VJP takes the lambda-seeded shortcut: if the VJP
     returns J*lam rather than J^T*lam, the test compares lam^T J^T x against lam^T J x,
     which differ by exactly the asymmetry.

The SSA case is the control. Its VJP is an exact transpose of its JVP, so it should hit
round-off, which is what establishes that any DIVA discrepancy is signal.

DIVA is run BOTH ways, which is the point of keeping this test: with the coefficient
transpose off (eta_bar and beta_eff held fixed w.r.t. velocity, while the JVP carries the
full d(eta_bar)/du and d(beta_eff)/du through dual arithmetic) check 2 measures precisely
what those terms are worth, and with it on it should hit round-off. The frozen number comes
out near 1e-1 -- about 10% -- which a finite-difference gradient check could not resolve at
all, and which is why the frozen case is retained rather than deleted: it is the standing
evidence that the omitted terms matter. The exact case is the default in production.

    uv run python tests/diva_dotproduct_test.py
"""
import contextlib
import io

import cupy as cp
import numpy as np

from glide.multigrid import Multigrid, FASCDSolver

ny = nx = 64
dx = 2000.0
dt = cp.float32(1.0)
L = 20000.0
RHO_I = cp.float32(917.0)
GRAV = cp.float32(9.81)
SEED = 3

SSA_DOTPROD_BOUND = 1e-5     # exact transpose: round-off only
JVP_FD_BOUND = 1e-3          # limited by the FD reference, not the JVP
JVP_FD_H_BOUND = 5e-3        # thickness direction; judged against the SSA control below
DIVA_FROZEN_BOUND = 0.5      # loose: the frozen VJP is knowingly incomplete (~1e-1)
DIVA_EXACT_BOUND = 1e-5      # with the coefficient terms applied: round-off


def converged_state(stress_balance):
    x = cp.arange(nx, dtype=cp.float32) * dx
    y = cp.arange(ny, dtype=cp.float32) * dx
    X, Y = cp.meshgrid(x, y)
    srf = 1000.0 * cp.ones((ny, nx), dtype=cp.float32) - cp.tan(cp.deg2rad(0.1)) * X + 10000
    bed = srf - 1000.0
    thk = srf - bed
    beta = (1000 * cp.sin(2 * cp.pi * X / L) * cp.sin(2 * cp.pi * Y / L) + 1000) / (RHO_I * GRAV)
    B = cp.ones((ny, nx), dtype=cp.float32)
    B.fill((1e-16 ** -(1. / 3)) / (RHO_I * GRAV))

    mg = Multigrid(3, ny=ny, nx=nx, dx=dx)
    mg.geometry.bed.set(bed)
    mg.rheology.B.set(B)
    mg.state.H.set(thk)
    mg.state.H_prev.set(thk)
    mg.sliding.beta.set(beta)
    mg.sliding.m.set(1.0)
    mg.sliding.u_reg.set(1.0)
    mg.rheology.stress_balance.set(stress_balance)
    mg.rheology.n_sigma.set(8.0)

    solver = FASCDSolver(mg)
    solver.vanka_options.omega.set(0.5)
    solver.vanka_options.newton_options.relaxation.set(0.5)
    solver.vanka_options.newton_options.steps.set(30)
    solver.fas_options.coarsest_steps.set(200)
    solver.fas_options.pre_steps.set(10)
    solver.fas_options.post_steps.set(50)
    solver.fas_options.finest_steps.set(150)
    solver.fas_options.maximum_vcycles.set(12)
    solver.fas_options.relative_tolerance.set(1e-6)
    solver.fas_options.absolute_tolerance.set(1e-6)
    with contextlib.redirect_stdout(io.StringIO()):
        solver.solve(dt)
    return mg


def masks():
    """Interior-only, so Dirichlet rows and the active-set mask play no part."""
    mu = np.zeros((ny, nx + 1), dtype=np.float32); mu[5:-5, 5:-5] = 1.0
    mv = np.zeros((ny + 1, nx), dtype=np.float32); mv[5:-5, 5:-5] = 1.0
    mh = np.zeros((ny, nx), dtype=np.float32);     mh[5:-5, 5:-5] = 1.0
    return mu, mv, mh


def run(stress_balance, tag, exact_coeff_adjoint=False):
    mu, mv, mh = masks()
    rng = np.random.RandomState(SEED)
    rand = lambda m: (rng.randn(*m.shape) * m).astype(np.float32)
    xu, xv, xH = rand(mu), rand(mv), rand(mh)
    lu, lv, lH = rand(mu), rand(mv), rand(mh)

    mg = converged_state(stress_balance)
    grid = mg.levels[0]
    fo, ao = grid.forward_operators, grid.adjoint_operators
    ao.diva_exact_coeff_adjoint = exact_coeff_adjoint
    grid.state.mask.data.fill(0.0)
    for a in (fo.f_u, fo.f_v, fo.f_H, ao.f_u, ao.f_v, ao.f_H):
        a.fill(0.0)
    u0 = grid.state.u.data.copy()
    v0 = grid.state.v.data.copy()
    # u_b must be restored too: the FD block below cold-starts it, and the closure's
    # warm start is part of the state both the JVP and the VJP are evaluated at.
    ub0 = grid.state.u_b.data.copy()

    # --- 1. JVP against FD of the residual (velocity direction only) ---
    fo.var_u[:, :] = cp.asarray(xu); fo.var_v[:, :] = cp.asarray(xv); fo.var_H.fill(0.0)
    fo.compute_jvp(dt, use_mask=False, freeze_phi=True)
    Ju = cp.asnumpy(fo.jvp_u).copy()

    def residual(eps):
        grid.state.u.data[:, :] = u0 + eps * cp.asarray(xu)
        grid.state.v.data[:, :] = v0 + eps * cp.asarray(xv)
        grid.state.u_b.data.fill(0.0)          # cold start -> pure function of the state
        fo.compute_residual(dt, use_mask=False, freeze_phi=True)
        return cp.asnumpy(fo.r_u).copy()

    eps = 1e-2
    fd = (residual(eps) - residual(-eps)) / (2 * eps)
    grid.state.u.data[:, :] = u0
    grid.state.v.data[:, :] = v0
    grid.state.u_b.data[:, :] = ub0
    if stress_balance > 0.5:
        # compute_residual refreshed eta_bar/F2/beta_eff at the perturbed state, so put
        # them back in step with the restored one -- otherwise the VJP reads coefficient
        # fields from a different state than the JVP differentiates.
        fo.compute_diva_coeffs()
    sel = mu > 0
    jvp_err = np.abs(Ju[sel] - fd[sel]).max() / max(np.abs(fd[sel]).max(), 1e-30)

    # --- 1b. JVP against FD in a THICKNESS direction ---
    # d_u = d_v = 0, d_H = xH.  Under DIVA this exercises the closure's response to
    # thickness; under SSA only the residual's explicit H terms, which is the control.
    H0 = grid.state.H.data.copy()
    fo.var_u.fill(0.0); fo.var_v.fill(0.0); fo.var_H[:, :] = cp.asarray(xH)
    fo.compute_jvp(dt, use_mask=False, freeze_phi=True)
    JuH = cp.asnumpy(fo.jvp_u).copy()

    def residual_H(eps):
        grid.state.H.data[:, :] = H0 + eps * cp.asarray(xH)
        grid.state.u_b.data.fill(0.0)          # cold start -> pure function of the state
        fo.compute_residual(dt, use_mask=False, freeze_phi=True)
        return cp.asnumpy(fo.r_u).copy()

    # H is O(1e3) and xH is O(1), so the step is swept in ABSOLUTE metres: too small and
    # the difference is float32 round-off on a large H, too large and the closure's
    # nonlinearity shows up as truncation error.
    jvp_err_H = np.inf
    for epsH in (1e-1, 3e-1, 1.0, 3.0):
        fdH = (residual_H(epsH) - residual_H(-epsH)) / (2 * epsH)
        err = np.abs(JuH[sel] - fdH[sel]).max() / max(np.abs(fdH[sel]).max(), 1e-30)
        jvp_err_H = min(jvp_err_H, err)
    grid.state.H.data[:, :] = H0
    grid.state.u.data[:, :] = u0
    grid.state.v.data[:, :] = v0
    grid.state.u_b.data[:, :] = ub0
    if stress_balance > 0.5:
        fo.compute_diva_coeffs()

    # --- 2. dot-product test over all three components ---
    fo.var_u[:, :] = cp.asarray(xu); fo.var_v[:, :] = cp.asarray(xv); fo.var_H[:, :] = cp.asarray(xH)
    fo.compute_jvp(dt, use_mask=False, freeze_phi=True)
    Jx = (cp.asnumpy(fo.jvp_u), cp.asnumpy(fo.jvp_v), cp.asnumpy(fo.jvp_H))

    grid.adjoint.lambda_u.data[:, :] = cp.asarray(lu)
    grid.adjoint.lambda_v.data[:, :] = cp.asarray(lv)
    grid.adjoint.lambda_H.data[:, :] = cp.asarray(lH)
    ao.compute_residual(dt, use_mask=False)
    JTl = (cp.asnumpy(ao.r_u), cp.asnumpy(ao.r_v), cp.asnumpy(ao.r_H))

    a = float((JTl[0] * xu * mu).sum() + (JTl[1] * xv * mv).sum() + (JTl[2] * xH * mh).sum())
    b = float((lu * Jx[0] * mu).sum() + (lv * Jx[1] * mv).sum() + (lH * Jx[2] * mh).sum())
    dot_err = abs(a - b) / max(abs(a), abs(b), 1e-30)

    print(f"{tag}: JVP vs FD (u) = {jvp_err:.3e}  (H) = {jvp_err_H:.3e} | "
          f"<J^T lam,x> = {a:+.6e}  <lam,Jx> = {b:+.6e}  rel diff = {dot_err:.3e}")
    return jvp_err, jvp_err_H, dot_err


def main():
    jvp_ssa, jvpH_ssa, dot_ssa = run(0.0, "SSA  (control)")
    assert jvp_ssa < JVP_FD_BOUND, f"SSA JVP disagrees with FD: {jvp_ssa:.3e}"
    assert dot_ssa < SSA_DOTPROD_BOUND, \
        f"SSA VJP is not the exact transpose of its JVP: {dot_ssa:.3e}"

    jvp_diva, jvpH_diva, dot_frozen = run(1.0, "DIVA frozen   ")
    assert jvp_diva < JVP_FD_BOUND, f"DIVA JVP disagrees with FD: {jvp_diva:.3e}"
    assert dot_frozen < DIVA_FROZEN_BOUND, \
        f"DIVA frozen adjoint far worse than expected: {dot_frozen:.3e}"

    # Coarse by construction -- see the module docstring for the measured sensitivity.
    # A gross error in the closure's H path lands here; a subtle one will not, which is
    # why diva_derivs_test.py checks d(eta_bar)/dH and friends directly.
    assert jvpH_diva < JVP_FD_H_BOUND, (
        f"the DIVA JVP disagrees with FD in a THICKNESS direction: {jvpH_diva:.3e} "
        f"(SSA control {jvpH_ssa:.3e}). Suspect the closure's H path -- H must be typed T "
        f"in diva_coeffs_cell and d_H must reach populate_diva_coeffs_dual.")

    _, _, dot_exact = run(1.0, "DIVA exact    ", exact_coeff_adjoint=True)
    assert dot_exact < DIVA_EXACT_BOUND, \
        f"the exact coefficient transpose is not exact: {dot_exact:.3e}"

    print(f"\nDIVA JVP validated: velocity {jvp_diva:.1e} vs FD (SSA {jvp_ssa:.1e}), "
          f"thickness {jvpH_diva:.1e} (SSA {jvpH_ssa:.1e}).")
    print(f"Frozen coefficient adjoint: {dot_frozen:.2e}  --  about 10% wrong.")
    print(f"Exact coefficient adjoint:  {dot_exact:.2e}  --  round-off, vs SSA control {dot_ssa:.1e}.")
    print("So the closure terms are exactly right, and they are ON by default.")
    print("The adjoint SMOOTHER still assembles the frozen block, which is fine -- it is")
    print("only a preconditioner, exactly as in SSA, where the VJP carries d(eta)/du and")
    print("the smoother does not; see notes/diva_numerics.md section 5.4.")


if __name__ == '__main__':
    main()
