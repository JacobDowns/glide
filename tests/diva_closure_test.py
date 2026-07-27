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


def build():
    """Uniform-flow state: all membrane strain rates vanish identically."""
    mg = Multigrid(2, ny=ny, nx=nx, dx=dx)
    mg.state.u.set(cp.full((ny, nx + 1), U0, dtype=cp.float32))
    mg.state.v.set(cp.zeros((ny + 1, nx), dtype=cp.float32))
    mg.state.H.set(cp.full((ny, nx), H0, dtype=cp.float32))
    mg.state.phi.set(cp.ones((ny, nx), dtype=cp.float32))        # fully grounded
    mg.rheology.B.set(cp.full((ny, nx), B0, dtype=cp.float32))
    mg.rheology.n.set(GLEN_N)
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


def main():
    glen_exp = (1.0 - GLEN_N) / (2.0 * GLEN_N)
    eta_mem = 0.5 * B0 * EPS_REG ** glen_exp
    # Midpoint rule for int_0^1 zeta^2 dzeta (1/3 up to quadrature error).
    quad = sum((1.0 / N_SIGMA) * (((k + 0.5) / N_SIGMA) ** 2) for k in range(N_SIGMA))
    print(f"analytic: eta_membrane = {eta_mem:.4f}, quadrature int zeta^2 = {quad:.6f}")

    # --- 1. zero drag: the SSA limit -------------------------------------------------
    mg = build()
    mg.sliding.beta.set(cp.zeros((ny, nx), dtype=cp.float32))
    eta_bar, F2, u_b, beta_eff = coeffs(mg)
    F2_expect = H0 * quad / eta_mem
    print(f"\n[zero drag] eta_bar = {eta_bar:.4f} (expect {eta_mem:.4f}), "
          f"F2 = {F2:.5f} (expect {F2_expect:.5f}), u_b = {u_b:.4f} (expect {U0})")
    assert abs(eta_bar - eta_mem) / eta_mem < 1e-3, \
        "with no shear eta_bar must reduce to the SSA membrane viscosity"
    assert abs(F2 - F2_expect) / F2_expect < 1e-3, "F2 disagrees with the quadrature"
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

    print("\nOK: DIVA coefficients and sliding closure behave as derived")


if __name__ == '__main__':
    main()
