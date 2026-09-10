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
depth-varying viscosity, with no reference to the SIA result.  Verified symbolically both ways.

The same integral with weight zeta rather than zeta^2 gives the SURFACE velocity,

    u_s = u_b + tau_b*F1,   F1 = 2*A*H*tau_b^(n-1)/(n+1)

(Arthern eq 10), which is what velocity observations measure and what an ISMIP-HOM comparison
needs.  Both are checked here.

WHAT IS AND IS NOT CHECKABLE HERE.  Only F2/F1 (hence the deformational and surface velocity).
eta_bar is NOT: in pure shear eta ~ zeta^(1-n) diverges at the stress-free surface, so its depth
average is set by the regularization rather than by the physics.  The zeta^2 (zeta) weight in
F2 (F1) removes that singularity, which is why the moments are the well-posed quantities and F2
is the one the momentum balance actually consumes.

The shear moments are regularized on the unified tree by `eps_reg_shear` -- distinct from the
membrane `eps_reg` that conditions eta_bar (notes/diva_numerics.md 3.2).  Part 1 drives
`eps_reg_shear` to zero (F2 converges); part 3 shows why the split matters: at the DEFAULT
eps_reg_shear the moments are already accurate at realistic driving stresses, where a single
SSA-sized eps_reg would have softened the column badly.

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
EPS_REG_SHEAR_DEFAULT = 1e-12

# Round-off floor for this comparison in float32, measured by driving eps_reg_shear down until
# the error stops moving.
FLOOR = 2e-4


def coeffs(U0, beta, n_sigma, eps_reg_shear):
    """DIVA's coefficients for a uniform flow, which makes every membrane strain rate vanish
    exactly -- so the only strain rate left is the vertical shear the slab solution describes.
    The membrane eps_reg is irrelevant here (no membrane strain); eps_reg_shear sets F1/F2."""
    mg = Multigrid(2, ny=ny, nx=nx, dx=dx, stress_scheme='diva')
    mg.state.u.set(cp.full((ny, nx + 1), U0, dtype=cp.float32))
    mg.state.v.set(cp.zeros((ny + 1, nx), dtype=cp.float32))
    mg.state.H.set(cp.full((ny, nx), H0, dtype=cp.float32))
    mg.state.phi.set(cp.ones((ny, nx), dtype=cp.float32))
    mg.rheology.B.set(cp.full((ny, nx), B0, dtype=cp.float32))
    mg.rheology.n.set(GLEN_N)
    mg.rheology.eps_reg.set(1e-6)
    mg.sliding.beta.set(cp.full((ny, nx), beta, dtype=cp.float32))
    mg.sliding.m.set(1.0)
    mg.sliding.u_reg.set(1e-12)
    mg.sliding.water_drag.set(0.0)
    mg.sliding.sliding_law.set(0.0)
    grid = mg.levels[0]
    grid.rheology.n_sigma.set(cp.float32(n_sigma))          # DIVA-only: set per level
    grid.rheology.eps_reg_shear.set(cp.float32(eps_reg_shear))
    grid.forward_operators.compute_diva_coeffs()
    at = lambda f: float(cp.asnumpy(f.data)[ny // 2, nx // 2])
    return at(grid.rheology.F2), at(grid.state.u_b), at(grid.rheology.F1)


def F2_analytic(tau_b):
    """ubar = u_b + tau_b*F2, the depth average."""
    return 2.0 * H0 * tau_b ** (GLEN_N - 1.0) / (B0 ** GLEN_N * (GLEN_N + 2.0))


def F1_analytic(tau_b):
    """u_surface = u_b + tau_b*F1.  Same integral with weight zeta instead of zeta^2, so the
    n+2 becomes n+1 -- verified symbolically alongside F2."""
    return 2.0 * H0 * tau_b ** (GLEN_N - 1.0) / (B0 ** GLEN_N * (GLEN_N + 1.0))


def regularized_fraction(tau_b, eps_reg_shear):
    """Fraction of the column, measured from the surface, where eps_reg_shear exceeds the true
    shear term and therefore sets the viscosity: eps_xz^2 < eps_reg_shear with
    eps_xz = B^-n (tau_b*zeta)^n."""
    amp = (B0 ** -GLEN_N) * tau_b ** GLEN_N
    return min((eps_reg_shear / amp ** 2) ** (1.0 / (2 * GLEN_N)), 1.0)


def main():
    print(f"B = {B0:.5g} (code units), H = {H0:g} m, n = {GLEN_N:g}, linear sliding law\n")

    # ---- 1. the formulation is right: F2 -> analytic as the shear regularization is removed ----
    print("F2 vs the analytic SIA slab value, driving eps_reg_shear down (n_sigma = 128):")
    worst_converged = 0.0
    for U0, beta in ((100.0, 0.05), (100.0, 0.5), (20.0, 0.5)):
        errs = []
        for eps in (1e-6, 1e-8, 1e-10, 1e-12):
            F2, u_b, F1 = coeffs(U0, beta, 128, eps)
            tau_b = beta * u_b
            errs.append(abs(F2 - F2_analytic(tau_b)) / F2_analytic(tau_b))
        print(f"  U0={U0:<6g} beta={beta:<6g} tau_b={tau_b:8.4g}   " +
              "  ".join(f"{e:.2e}" for e in errs))
        worst_converged = max(worst_converged, errs[-1])
    assert worst_converged < FLOOR, (
        f"DIVA's F2 does not converge to the analytic SIA slab value: {worst_converged:.2e}. "
        f"That is a formulation or quadrature error, not a solver one -- every other DIVA test "
        f"checks the code against itself and would pass regardless.")
    print(f"  -> converged to {worst_converged:.1e} at eps_reg_shear = 1e-12, i.e. the float32 floor\n")

    # ---- 2. quadrature order, at a regularization small enough not to dominate ----
    print("n_sigma convergence at eps_reg_shear = 1e-12 (U0 = 100, beta = 0.5).")
    print("The integrand is zeta^(n+1), a polynomial for integer n, so Gauss-Legendre is EXACT")
    print("from ceil((n+2)/2) = 3 nodes upward -- no convergence sequence to speak of:")
    for n_sigma in (2, 3, 4, 8, 32):
        F2, u_b, F1 = coeffs(100.0, 0.5, n_sigma, 1e-12)
        e2 = abs(F2 - F2_analytic(0.5 * u_b)) / F2_analytic(0.5 * u_b)
        e1 = abs(F1 - F1_analytic(0.5 * u_b)) / F1_analytic(0.5 * u_b)
        print(f"  n_sigma = {n_sigma:<4d} F2 err = {e2:.2e}   F1 err = {e1:.2e}")
    F2, u_b, F1 = coeffs(100.0, 0.5, 3, 1e-12)
    e3 = abs(F2 - F2_analytic(0.5 * u_b)) / F2_analytic(0.5 * u_b)
    assert e3 < FLOOR, (
        f"Gauss-Legendre with 3 nodes should integrate a degree-{int(GLEN_N)+1} polynomial "
        f"exactly, got {e3:.2e}")
    e1 = abs(F1 - F1_analytic(0.5 * u_b)) / F1_analytic(0.5 * u_b)
    assert e1 < FLOOR, f"F1 disagrees with the analytic surface moment: {e1:.2e}"

    # ---- 3. what the eps_reg / eps_reg_shear SPLIT buys, which is the point of this test ----
    # The fork carried a single eps_reg, sized for SSA's membrane strain; on DIVA's much smaller
    # vertical-shear strain that same value regularized a large fraction of the column and softened
    # F2 badly (up to ~60x in nearly stagnant ice).  The unified tree gives the shear moments their
    # own, far smaller eps_reg_shear (default 1e-12), so F2 stays accurate at the default.  Here we
    # show both: the default is accurate; an SSA-sized 1e-6 would not have been.
    print(f"\nWhat the eps_reg_shear split buys at realistic driving stresses (default = {EPS_REG_SHEAR_DEFAULT:g}).")
    print(f"tau_b is in code units; multiply by rho*g = {RHO_I * GRAV:.0f} Pa/m for stress:")
    print(f"  {'tau_b':>7} {'~kPa':>7} {'F2 err @default':>16} {'F2 err @1e-6':>14} {'col.reg @default':>16}")
    worst_active = 0.0
    for U0, beta in ((100.0, 0.5), (20.0, 0.5), (100.0, 0.05), (20.0, 0.02)):
        F2d, u_b, _ = coeffs(U0, beta, 8, EPS_REG_SHEAR_DEFAULT)
        tau_b = beta * u_b
        err_def = abs(F2d - F2_analytic(tau_b)) / F2_analytic(tau_b)
        F2o, u_bo, _ = coeffs(U0, beta, 8, 1e-6)
        err_old = abs(F2o - F2_analytic(beta * u_bo)) / F2_analytic(beta * u_bo)
        frac = regularized_fraction(tau_b, EPS_REG_SHEAR_DEFAULT)   # column fraction reg. even at default
        active = frac < 0.5                                          # deformation actually resolvable
        tag = "" if active else "  reg-limited (negligible deformation)"
        print(f"  {tau_b:7.3g} {tau_b * RHO_I * GRAV / 1e3:7.1f} {err_def:16.2e} {err_old:14.1%} "
              f"{'top ' + format(100 * frac, '.0f') + '%':>16}{tag}")
        # (a) the split must never be WORSE than a single SSA-sized eps_reg, at any stress:
        assert err_def <= err_old + FLOOR, (
            f"tau_b={tau_b:.3g}: default eps_reg_shear is worse than 1e-6 ({err_def:.2e} > {err_old:.2e})")
        # (b) where there IS deformation to capture, F2 must be accurate:
        if active:
            worst_active = max(worst_active, err_def)
    assert worst_active < FLOOR, (
        f"At the default eps_reg_shear the shear moment F2 should be accurate wherever the column "
        f"is not regularization-dominated, got {worst_active:.2e} -- the split is not doing its job.")
    print(f"""
  Where deformation is resolvable (>~40 kPa here) the @default column is at the float32 floor,
  while a single SSA-sized 1e-6 errs by up to 9%.  Nearly-stagnant ice (3.6 kPa) is regularization-
  limited even at 1e-12, but tau_b*F2 there is ~2e-3 m/a -- negligible, so it does not matter.  The
  split lets the shear moments use a regularization far below SSA's membrane one, so DIVA captures
  the deformational velocity it exists for without the operator-conditioning cost of a tiny eps_reg
  on eta_bar.  See notes/diva_numerics.md section 3.2 / 5.10.""")

    print("\nOK: DIVA reproduces the analytic SIA slab solution")


if __name__ == '__main__':
    main()
