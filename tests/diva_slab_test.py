"""DIVA against an ANALYTIC solution: the uniform slab (SIA + sliding).

Every other DIVA test checks internal consistency -- that the code solves the equations it
claims to, that its transpose is the transpose of its Jacobian, that its gradient matches finite
differences of itself. None of them can catch an error in the FORMULATION. This one can, because
the slab has a closed-form answer.

Take a slab of uniform thickness on a uniform slope with uniform basal drag. There are no
horizontal gradients, so membrane stresses vanish identically, all driving stress reaches the
bed, and the shear profile integrates twice in closed form:

    tau_xz(zeta) = tau_b*zeta,   zeta = (s-z)/H
    Glen:  du/dz = 2*A*tau_xz^n
    =>  ubar - u_b = 2*A*H*tau_b^n/(n+2)

DIVA writes that same quantity as tau_b*F2, so it must produce

    F2 = 2*A*H*tau_b^(n-1)/(n+2),        A = B^(-n) in GLIDE's units

and this is an independent statement: DIVA defines F2 = H*int zeta^2/eta dzeta, from the
depth-varying viscosity, with no reference to the SIA result. Verified symbolically both ways.

WHAT IS AND IS NOT CHECKABLE HERE.  Only F2 (hence the deformational velocity). eta_bar is NOT:
in pure shear eta ~ zeta^(1-n) diverges at the stress-free surface, so its depth average is set
by the regularization rather than by the physics. F2's zeta^2 weight removes that singularity,
which is why F2 is the well-posed quantity and the one the momentum balance actually consumes.

    uv run python tests/diva_slab_test.py
"""
import cupy as cp
import numpy as np

from glide.multigrid import Multigrid

ny = nx = 32
dx = 2000.0
H0 = 1000.0
GLEN_N = 3.0
RHO_I, GRAV = 917.0, 9.81
B0 = (1e-16 ** -(1. / 3)) / (RHO_I * GRAV)      # B = A^(-1/n)/(rho g), GLIDE's units

# Round-off floor for this comparison in float32, measured by driving eps_reg down until the
# error stops moving.
FLOOR = 2e-4


def coeffs(U0, beta, n_sigma, eps_reg):
    """DIVA's coefficients for a uniform flow, which makes every membrane strain rate vanish
    exactly -- so the only strain rate left is the vertical shear the slab solution describes."""
    mg = Multigrid(2, ny=ny, nx=nx, dx=dx)
    mg.state.u.set(cp.full((ny, nx + 1), U0, dtype=cp.float32))
    mg.state.v.set(cp.zeros((ny + 1, nx), dtype=cp.float32))
    mg.state.H.set(cp.full((ny, nx), H0, dtype=cp.float32))
    mg.state.phi.set(cp.ones((ny, nx), dtype=cp.float32))
    mg.rheology.B.set(cp.full((ny, nx), B0, dtype=cp.float32))
    mg.rheology.n.set(GLEN_N)
    mg.rheology.eps_reg.set(eps_reg)
    mg.rheology.n_sigma.set(float(n_sigma))
    mg.sliding.beta.set(cp.full((ny, nx), beta, dtype=cp.float32))
    mg.sliding.m.set(1.0)
    mg.sliding.u_reg.set(1e-12)
    mg.sliding.water_drag.set(0.0)
    mg.sliding.sliding_law.set(0.0)
    grid = mg.levels[0]
    grid.forward_operators.compute_diva_coeffs()
    at = lambda f: float(cp.asnumpy(f.data)[ny // 2, nx // 2])
    return at(grid.rheology.F2), at(grid.state.u_b)


def F2_analytic(tau_b):
    return 2.0 * H0 * tau_b ** (GLEN_N - 1.0) / (B0 ** GLEN_N * (GLEN_N + 2.0))


def regularized_fraction(tau_b, eps_reg):
    """Fraction of the column, measured from the surface, where eps_reg exceeds the true shear
    term and therefore sets the viscosity: eps_xz^2 < eps_reg with eps_xz = B^-n (tau_b*zeta)^n."""
    amp = (B0 ** -GLEN_N) * tau_b ** GLEN_N
    return min((eps_reg / amp ** 2) ** (1.0 / (2 * GLEN_N)), 1.0)


def main():
    print(f"B = {B0:.5g} (code units), H = {H0:g} m, n = {GLEN_N:g}, linear sliding law\n")

    # ---- 1. the formulation is right: F2 -> analytic as the regularization is removed ----
    print("F2 vs the analytic SIA slab value, driving eps_reg down (n_sigma = 128):")
    worst_converged = 0.0
    for U0, beta in ((100.0, 0.05), (100.0, 0.5), (20.0, 0.5)):
        errs = []
        for eps_reg in (1e-6, 1e-8, 1e-10, 1e-12):
            F2, u_b = coeffs(U0, beta, 128, eps_reg)
            tau_b = beta * u_b
            errs.append(abs(F2 - F2_analytic(tau_b)) / F2_analytic(tau_b))
        print(f"  U0={U0:<6g} beta={beta:<6g} tau_b={tau_b:8.4g}   " +
              "  ".join(f"{e:.2e}" for e in errs))
        worst_converged = max(worst_converged, errs[-1])
    assert worst_converged < FLOOR, (
        f"DIVA's F2 does not converge to the analytic SIA slab value: {worst_converged:.2e}. "
        f"That is a formulation or quadrature error, not a solver one -- every other DIVA test "
        f"checks the code against itself and would pass regardless.")
    print(f"  -> converged to {worst_converged:.1e} at eps_reg = 1e-12, i.e. the float32 floor\n")

    # ---- 2. quadrature order, at a regularization small enough not to dominate ----
    print("n_sigma convergence at eps_reg = 1e-12 (U0 = 100, beta = 0.5):")
    prev = None
    for n_sigma in (4, 8, 16, 32):
        F2, u_b = coeffs(100.0, 0.5, n_sigma, 1e-12)
        err = abs(F2 - F2_analytic(beta * u_b)) / F2_analytic(beta * u_b)
        ratio = "" if prev is None else f"   ({prev / max(err, 1e-30):.1f}x)"
        print(f"  n_sigma = {n_sigma:<4d} rel err = {err:.2e}{ratio}")
        prev = err

    # ---- 3. what the DEFAULT regularization costs, which is the point of this test ----
    print(f"\nWhat eps_reg = 1e-6 (the default) costs at realistic driving stresses.")
    print(f"tau_b is in code units; multiply by rho*g = {RHO_I * GRAV:.0f} Pa/m for stress:")
    print(f"  {'tau_b':>7} {'~kPa':>7} {'F2 err':>9} {'column regularized':>20}")
    for U0, beta in ((100.0, 0.5), (20.0, 0.5), (100.0, 0.05), (20.0, 0.02)):
        F2, u_b = coeffs(U0, beta, 8, 1e-6)
        tau_b = beta * u_b
        err = abs(F2 - F2_analytic(tau_b)) / F2_analytic(tau_b)
        frac = regularized_fraction(tau_b, 1e-6)
        print(f"  {tau_b:7.3g} {tau_b * RHO_I * GRAV / 1e3:7.1f} {err:9.1%} "
              f"{'top ' + format(100 * frac, '.0f') + '%':>20}")
    print("""
  This is not a defect in the closure -- part 1 shows the formulation is exact. It is that
  eps_reg was chosen for SSA, whose only strain rates are membrane ones. DIVA's vertical shear
  strain rate is far smaller in slow ice, so the same eps_reg regularizes a large fraction of
  the column and suppresses the deformational velocity DIVA exists to capture. See
  notes/diva_numerics.md section 5.10.""")

    print("\nOK: DIVA reproduces the analytic SIA slab solution")


if __name__ == '__main__':
    main()
