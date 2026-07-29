"""Operator-consistency tests for DIVA: JVP against finite differences, then the
adjoint identity against the JVP.

Two checks, in the order that makes them interpretable:

  1. **JVP vs FD of the residual.** Validates compute_jvp_diva itself as a faithful
     Jacobian of compute_residual_diva. Without this, a failure in check 2 could equally
     be a JVP bug.

  2. **Dot-product test**  <J^T lam, x>  vs  <lam, J x>  for seeded random lam, x.
     This is the sharp instrument: it uses **no finite differences**, so it resolves to
     float32 round-off (~1e-7) rather than the ~1e-3 an FD gradient check tops out at.
     It is also not circular when the VJP takes the lambda-seeded shortcut: if the VJP
     returns J*lam rather than J^T*lam, the test compares lam^T J^T x against lam^T J x,
     which differ by exactly the asymmetry.

The SSA case is the control. Its VJP is an exact transpose of its JVP, so it should hit
round-off, which is what establishes that any DIVA discrepancy is signal.

Current status: the DIVA VJP is still the **frozen-coefficient** one -- eta_bar and
beta_eff held fixed with respect to velocity -- while the DIVA JVP carries the full
d(eta_bar)/du and d(beta_eff)/du through dual arithmetic. So check 2 measures precisely
what the frozen adjoint omits. It comes out near 1e-1, i.e. about 10%; a finite-difference
gradient check could not resolve this at all. DIVA_DOTPROD_BOUND should be tightened to
round-off once the missing terms are added to the VJP.

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
DIVA_DOTPROD_BOUND = 0.5     # loose: the frozen VJP is knowingly incomplete (~1e-1)


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


def run(stress_balance, tag):
    mu, mv, mh = masks()
    rng = np.random.RandomState(SEED)
    rand = lambda m: (rng.randn(*m.shape) * m).astype(np.float32)
    xu, xv, xH = rand(mu), rand(mv), rand(mh)
    lu, lv, lH = rand(mu), rand(mv), rand(mh)

    mg = converged_state(stress_balance)
    grid = mg.levels[0]
    fo, ao = grid.forward_operators, grid.adjoint_operators
    grid.state.mask.data.fill(0.0)
    for a in (fo.f_u, fo.f_v, fo.f_H, ao.f_u, ao.f_v, ao.f_H):
        a.fill(0.0)
    u0 = grid.state.u.data.copy()
    v0 = grid.state.v.data.copy()

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
    sel = mu > 0
    jvp_err = np.abs(Ju[sel] - fd[sel]).max() / max(np.abs(fd[sel]).max(), 1e-30)

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

    print(f"{tag}: JVP vs FD = {jvp_err:.3e} | "
          f"<J^T lam,x> = {a:+.6e}  <lam,Jx> = {b:+.6e}  rel diff = {dot_err:.3e}")
    return jvp_err, dot_err


def main():
    jvp_ssa, dot_ssa = run(0.0, "SSA  (control)")
    assert jvp_ssa < JVP_FD_BOUND, f"SSA JVP disagrees with FD: {jvp_ssa:.3e}"
    assert dot_ssa < SSA_DOTPROD_BOUND, \
        f"SSA VJP is not the exact transpose of its JVP: {dot_ssa:.3e}"

    jvp_diva, dot_diva = run(1.0, "DIVA          ")
    assert jvp_diva < JVP_FD_BOUND, f"DIVA JVP disagrees with FD: {jvp_diva:.3e}"
    assert dot_diva < DIVA_DOTPROD_BOUND, \
        f"DIVA adjoint identity far worse than the frozen approximation explains: {dot_diva:.3e}"

    print(f"\nDIVA JVP is validated ({jvp_diva:.1e} vs FD, same order as SSA's {jvp_ssa:.1e}),")
    print(f"so the adjoint-identity gap of {dot_diva:.2e} is the frozen-coefficient error,")
    print(f"measured against an SSA control at {dot_ssa:.1e}. Tighten DIVA_DOTPROD_BOUND")
    print("to round-off once the omitted terms are added to the VJP.")


if __name__ == '__main__':
    main()
