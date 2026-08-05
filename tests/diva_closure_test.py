"""Unit tests for the DIVA coefficient kernel (cuda/diva.cu).

compute_diva_coeffs diagnoses the depth-averaged viscosity eta_bar, the vertical shear
integral F2, and the basal speed u_b from the current state. Each is checked against a
value known independently rather than against a stored regression:

  1. zero drag   -- with tau_b = 0 there is no vertical shear, so eta_bar must collapse
                    onto the SSA membrane viscosity and F2 onto H*int(zeta^2)/eta.
                    This is the SSA limit of DIVA and the strongest single check.
  2. linear law  -- the closure U_b + f(U_b)*F2 = U_bar inverts in closed form,
                    U_b = U_bar/(1 + beta*F2), so the Newton iteration has an exact target.
  3. Coulomb law -- no closed form; require the closure residual itself to vanish.
  4. determinism -- repeated calls must agree bit-for-bit (the kernel reads u_b in place,
                    so only interior threads may write it).
  5. shear-dominated -- eta_bar, F2 and u_b against the TRUE root of the closure, found by
                    bisection in float64 with the per-level viscosity solved to convergence.
                    Run at n = 3 AND n = 4: the per-level contraction bound is (n-1)/n, so the
                    larger exponent is genuinely harder, and everything here was originally
                    validated at n = 3 only. Also asserts no cell hit an iteration cap, since
                    an adaptive loop that caps silently looks converged.
  6. idempotence -- repeated calls must agree, which an unconverged closure does not. This is the check the first four
                    miss: check 1 sets tau_b = 0, where the per-level solve is exact in one
                    step, and checks 2-3 validate the Newton against the COMPUTED F2, so they
                    pass whatever F2 happens to be. A truncated per-level solve was wrong by
                    16% in F2 at tau_b ~ 4 and 75% at tau_b ~ 30 while all four passed;
                    see notes/diva_numerics.md 5.2.1.

A uniform velocity field is used so every membrane strain rate vanishes exactly, which
makes eta_membrane analytic: eta = 0.5*B*eps_reg^((1-n)/(2n)).

    uv run python tests/diva_closure_test.py
"""
import cupy as cp
import numpy as np

from glide.multigrid import Multigrid

ny = nx = 32
dx = 2000.0
N_SIGMA = 8

U0 = 10.0        # uniform depth-averaged speed
H0 = 1000.0      # uniform thickness
B0 = 1.0         # rate factor, chosen so eta_membrane comes out a round number
EPS_REG = 1e-6
GLEN_N = 3.0
U_REG = 1e-12    # negligible, so f(U) is the clean law


def build(glen_n=GLEN_N):
    """Uniform-flow state: all membrane strain rates vanish identically."""
    mg = Multigrid(2, ny=ny, nx=nx, dx=dx)
    mg.state.u.set(cp.full((ny, nx + 1), U0, dtype=cp.float32))
    mg.state.v.set(cp.zeros((ny + 1, nx), dtype=cp.float32))
    mg.state.H.set(cp.full((ny, nx), H0, dtype=cp.float32))
    mg.state.phi.set(cp.ones((ny, nx), dtype=cp.float32))        # fully grounded
    mg.rheology.B.set(cp.full((ny, nx), B0, dtype=cp.float32))
    mg.rheology.n.set(glen_n)
    mg.rheology.eps_reg.set(EPS_REG)
    mg.rheology.n_sigma.set(float(N_SIGMA))
    mg.sliding.u_reg.set(U_REG)
    mg.sliding.water_drag.set(0.0)
    mg.sliding.m.set(1.0)
    mg.sliding.sliding_law.set(0.0)
    return mg


def coeffs(mg):
    """Run the kernel and read the outputs at an interior cell."""
    grid = mg.levels[0]
    grid.forward_operators.compute_diva_coeffs()
    at = lambda f: float(cp.asnumpy(f.data)[ny // 2, nx // 2])
    return (at(grid.rheology.eta_bar), at(grid.rheology.F2), at(grid.state.u_b),
            at(grid.sliding.beta_eff))


def exact_level_viscosity(A, k, glen_exp):
    """Solve eta = 0.5*B*(A + k/eta^2)^p to convergence in float64, by plain Picard run far
    past its (slow) contraction.  Deliberately a different algorithm from the kernel's Newton,
    so this is an independent reference rather than a re-implementation."""
    eta = 0.5 * B0 * A ** glen_exp
    for _ in range(4000):
        eta = 0.5 * B0 * (A + k / eta ** 2) ** glen_exp
    return eta


def reference_closure(beta, glen_exp):
    """The TRUE root of F(U_b) = U_b + f(U_b)*F2(f(U_b)) - U_bar = 0, found by BISECTION.

    Deliberately independent of the kernel's algorithm in both nesting levels: bisection rather
    than Newton for the closure, and plain Picard run to convergence rather than Newton for the
    per-level viscosity. Bisection is guaranteed here because F' >= 1, so F is strictly
    increasing and F(0) = -U_bar < 0 < F(U_bar).

    An earlier version of this reference replicated the kernel's own 3-sweep block structure,
    which made it a self-consistency check rather than a correctness one -- it would have
    passed the block iteration's 2-cycle. See notes/diva_numerics.md 5.2.0.
    """
    A = EPS_REG                      # membrane strain rates vanish in this configuration
    # Same Gauss-Legendre rule the kernel uses, built the same way (operators.py _quadrature).
    _x, _w = np.polynomial.legendre.leggauss(N_SIGMA)
    nodes, weights = 0.5 * (_x + 1.0), 0.5 * _w

    def quadrature(tau_b):
        eta_avg = 0.0
        F2 = 0.0
        for zeta, wq in zip(nodes, weights):
            eta = exact_level_viscosity(A, (tau_b * zeta / 2.0) ** 2, glen_exp)
            eta_avg += wq * eta
            F2 += wq * zeta * zeta / eta
        return eta_avg, F2 * H0

    # linear law here, so f(U_b) = beta*U_b
    F = lambda U_b: U_b + beta * U_b * quadrature(beta * U_b)[1] - U0
    lo, hi = 0.0, U0
    assert F(lo) < 0.0, "bracket lost: F(0) should be -U_bar"
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if F(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    U_b = 0.5 * (lo + hi)
    eta_avg, F2 = quadrature(beta * U_b)
    return eta_avg, F2, U_b


def check_shear_dominated(glen_exp):
    """Push tau_b until the vertical shear dominates the membrane term, and require the kernel
    to match the true root of the closure.

    This is the check the first four miss. Check 1 sets tau_b = 0, where both nested solves are
    exact in one step; checks 2-3 validate U_b against the COMPUTED F2, so they pass whatever
    F2 is. Two separate defects hid behind them -- a truncated per-level Picard (F2 wrong by
    16% at our own tau_b, 75% at ~250 kPa) and a block closure iteration that 2-cycled at high
    drag. See notes/diva_numerics.md 5.2.0 and 5.2.1.

    Run from a COLD start, so it also pins that one call converges rather than relying on the
    warm start from a previous one.
    """
    worst = 0.0
    # Both Glen exponents in use.  n matters here and not only cosmetically: the per-level
    # solve's contraction bound is (n-1)/n, so n = 4 is materially harder than n = 3 (0.75 vs
    # 0.67), and the shear-dominated asymptotic guess carries an n in its exponent.  Everything
    # was originally validated at n = 3 only.
    for glen_n in (3.0, 4.0):
      glen_exp = (1.0 - glen_n) / (2.0 * glen_n)
      for beta in (0.02, 0.1, 0.5, 2.0):
        mg = build(glen_n)
        mg.sliding.beta.set(cp.full((ny, nx), beta, dtype=cp.float32))
        eta_bar, F2, u_b, _ = coeffs(mg)
        eta_ref, F2_ref, u_b_ref = reference_closure(beta, glen_exp)
        d_eta = abs(eta_bar - eta_ref) / eta_ref
        d_F2 = abs(F2 - F2_ref) / F2_ref
        d_ub = abs(u_b - u_b_ref) / max(abs(u_b_ref), 1e-30)
        n_cl, n_eta = mg.levels[0].forward_operators.diva_cap_counts()
        print(f"[shear]     n = {glen_n:g}  beta = {beta:<5g} eta_bar err = {d_eta:.2e}   "
              f"F2 err = {d_F2:.2e}   u_b err = {d_ub:.2e}   capped = {n_cl}/{n_eta}")
        # An adaptive loop that hits its backstop looks converged.  Assert it did not.
        assert n_cl == 0 and n_eta == 0, (
            f"n={glen_n} beta={beta}: local solves hit their iteration cap "
            f"(closure {n_cl} cells, eta {n_eta} cells) -- the coefficients there are not "
            f"converged")
        worst = max(worst, d_eta, d_F2, d_ub)
    assert worst < 1e-4, (
        f"closure does not reach the true root: worst relative error {worst:.2e}. "
        f"see notes/diva_numerics.md 5.2.0")
    return worst


def check_idempotent():
    """Repeated calls must return the same answer.

    u_b is read in place as the warm start, so a call that has genuinely converged reproduces
    itself. The block iteration did not: it sat in a stable 2-cycle, u_b alternating between
    ~0.71 and ~5.75 at beta = 0.1, with F2 swinging by a factor of 18. That is invisible to
    every other check here, and it is the sharpest single symptom of an unconverged closure.
    """
    worst = 0.0
    for beta in (0.1, 2.0):
        mg = build()
        mg.sliding.beta.set(cp.full((ny, nx), beta, dtype=cp.float32))
        seq = [coeffs(mg)[2] for _ in range(6)]
        spread = (max(seq) - min(seq)) / max(abs(sum(seq) / len(seq)), 1e-30)
        print(f"[repeat x6] beta = {beta:<5g} u_b spread over 6 calls = {spread:.2e}")
        worst = max(worst, spread)
    assert worst < 1e-5, f"repeated calls disagree ({worst:.2e}): the closure has not converged"
    return worst


def main():
    glen_exp = (1.0 - GLEN_N) / (2.0 * GLEN_N)
    eta_mem = 0.5 * B0 * EPS_REG ** glen_exp
    # int_0^1 zeta^2 dzeta = 1/3 EXACTLY.  The vertical quadrature is Gauss-Legendre, which
    # integrates a quadratic exactly with 2 nodes and so certainly with N_SIGMA of them, and
    # this check pins that: under the previous midpoint rule the expected value carried the
    # quadrature error (0.332031 at N_SIGMA = 8) and had to be computed rather than known.
    quad = 1.0 / 3.0
    print(f"analytic: eta_membrane = {eta_mem:.4f}, int zeta^2 = {quad:.6f} (exact under "
          f"Gauss-Legendre)")

    # --- 1. zero drag: the SSA limit -------------------------------------------------
    mg = build()
    mg.sliding.beta.set(cp.zeros((ny, nx), dtype=cp.float32))
    eta_bar, F2, u_b, beta_eff = coeffs(mg)
    F2_expect = H0 * quad / eta_mem
    print(f"\n[zero drag] eta_bar = {eta_bar:.4f} (expect {eta_mem:.4f}), "
          f"F2 = {F2:.5f} (expect {F2_expect:.5f}), u_b = {u_b:.4f} (expect {U0})")
    assert abs(eta_bar - eta_mem) / eta_mem < 1e-3, \
        "with no shear eta_bar must reduce to the SSA membrane viscosity"
    assert abs(F2 - F2_expect) / F2_expect < 1e-5, (
        "F2 disagrees with the exact integral -- Gauss-Legendre should make this round-off")
    assert abs(u_b - U0) / U0 < 1e-5, "with no drag all motion is sliding (u_b = U_bar)"
    assert abs(beta_eff) < 1e-6, "no drag -> beta_eff = 0"

    # --- 2. linear law: closure has a closed form -------------------------------------
    beta = 0.01
    mg = build()
    mg.sliding.beta.set(cp.full((ny, nx), beta, dtype=cp.float32))
    eta_bar, F2, u_b, beta_eff = coeffs(mg)
    u_b_expect = U0 / (1.0 + beta * F2)      # uses the kernel's own F2: an exact identity
    print(f"[linear]    u_b = {u_b:.6f} (closed form {u_b_expect:.6f}), "
          f"F2 = {F2:.5f}, eta_bar = {eta_bar:.4f}")
    assert abs(u_b - u_b_expect) / u_b_expect < 1e-5, \
        "closure Newton disagrees with the linear closed form"
    assert u_b < U0, "with drag the basal speed must fall below the depth-averaged speed"
    assert eta_bar < eta_mem, "vertical shear must soften eta (shear thinning)"
    # Goldberg eq 41, and the identity the momentum solve relies on: tau_b = beta_eff*U_bar.
    beta_eff_expect = beta / (1.0 + beta * F2)
    tau_b = beta * u_b                                    # linear law: |tau_b| = beta*U_b
    print(f"            beta_eff = {beta_eff:.8f} (eq 41 {beta_eff_expect:.8f}), "
          f"beta_eff*U_bar = {beta_eff * U0:.6f} vs tau_b = {tau_b:.6f}")
    assert abs(beta_eff - beta_eff_expect) / beta_eff_expect < 1e-5, "beta_eff != c/(1+c*F2)"
    assert abs(beta_eff * U0 - tau_b) / tau_b < 1e-5, "tau_b = beta_eff*U_bar must hold"

    # --- 3. regularized Coulomb: closure residual must vanish -------------------------
    tau_max, u_c = 1.0, 100.0
    mg = build()
    mg.sliding.sliding_law.set(1.0)
    mg.sliding.beta.set(cp.full((ny, nx), tau_max, dtype=cp.float32))   # beta slot = tau_max
    mg.sliding.u_c.set(cp.full((ny, nx), u_c, dtype=cp.float32))
    _, F2, u_b, beta_eff = coeffs(mg)
    f_u_b = tau_max * u_b / (np.sqrt(u_b ** 2 + U_REG) + u_c)
    residual = u_b + f_u_b * F2 - U0
    print(f"[coulomb]   u_b = {u_b:.6f}, closure residual = {residual:.3e}")
    assert abs(residual) / U0 < 1e-5, "Coulomb closure residual should vanish"
    assert abs(beta_eff * U0 - f_u_b) / f_u_b < 1e-4, "tau_b = beta_eff*U_bar must hold (Coulomb)"

    # --- 4. determinism ---------------------------------------------------------------
    first, second = coeffs(mg), coeffs(mg)
    print(f"[repeat]    {first} vs {second}")
    assert first == second, "compute_diva_coeffs must be deterministic"

    print()
    worst = check_shear_dominated(glen_exp)
    print(f"            worst coefficient error vs the true root: {worst:.2e}")
    print()
    check_idempotent()

    print("\nOK: DIVA coefficients and sliding closure behave as derived")


if __name__ == '__main__':
    main()
