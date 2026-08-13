"""The DIVA smoother must stay bounded for a SUBLINEAR sliding law.

This is a regression test for a real failure: the DIVA forward solve produced NaN on
Greenland geometry at m = 1/3, at every multigrid level but one, while SSA on the same
geometry was clean everywhere.

The cause was in the smoother's local block, which used to carry the cell's basal speed
U_b as an augmented unknown and eliminate it with a rank-1 update.  That update is the
exact Newton tangent for the drag, but

    d(beta_eff)/d(U_b) = c'(U_b)/(1 + c*F2)^2,      c' ~ (m - 1)*U

so for m < 1 it is negative, and since the momentum diagonals are already negative -- drag
resists -- it eroded the diagonal.  The failure was LOCAL rather than a step length: it
left the median correction unchanged (0.0029 -> 0.0030) while the maximum went 5.31 ->
4108, because the update nearly cancelled the block diagonal in a handful of cells and
left the 5x5 near-singular there.  A global under-relaxation cannot rescue that, and did
not: omega = 0.05 still diverged.

U_b is no longer the block's business.  It is diagnosed by compute_diva_coeffs from
(u,v,H) along with eta_bar, beta_eff, F1, F2 and u_s, and the smoother consumes exactly
two frozen coefficients -- eta_bar and beta_eff -- which is the same contract SSA's
smoother has with its frozen viscosity.  The measurements behind that decision, including
the convergence comparison that showed the exact tangent never won, are in
notes/diva_numerics.md 5.2.

What this test pins is the outcome: several sweeps at a step size and geometry that used
to blow up must stay bounded, for a strongly sublinear law as well as a linear one.

Two properties worth knowing if it ever fails:

  * m = 1 is a CONTROL, not merely a milder case.  There the sliding coefficient does not
    depend on speed at all, so a drag-linearization bug cannot show up in it.  Failing at
    m = 1 means something else is wrong.
  * The instability needed a finite speed to appear -- c' ~ U vanishes at rest -- which is
    why this runs several sweeps rather than one.

    uv run python tests/diva_smoother_stability_test.py
"""
import cupy as cp
import numpy as np

from glide.multigrid import Multigrid

ny = nx = 64
dx = 2000.0
dt = cp.float32(10.0)          # dt matters but does not by itself decide it: on Greenland
                               # dt = 0.1 was stable, while in THIS geometry the old block
                               # still diverged there. Geometry and dt both count.
n_levels = 3
N_SWEEPS = 8
RHO_I, GRAV = 917.0, 9.81
GROWTH_BOUND = 50.0            # max|u| may settle anywhere sane; it may not explode


def build(m):
    """A sloping slab with a bumpy bed, thick enough for a real basal drag."""
    mg = Multigrid(n_levels, ny=ny, nx=nx, dx=dx)
    x = cp.arange(nx, dtype=cp.float32) * dx
    X, Y = cp.meshgrid(x, cp.arange(ny, dtype=cp.float32) * dx)
    L = nx * dx

    srf = 2600.0 - cp.tan(cp.deg2rad(0.25)) * X
    bed = srf - (1200.0 + 300.0 * cp.sin(2 * cp.pi * X / L) * cp.sin(2 * cp.pi * Y / L))

    mg.geometry.bed.set(bed)
    mg.geometry.depth.set(cp.maximum(-bed, 0))
    mg.state.H.set(srf - bed)
    mg.state.H_prev.set(srf - bed)
    mg.rheology.B.set(cp.full((ny, nx), (1e-16 ** -(1. / 3)) / (RHO_I * GRAV), cp.float32))
    mg.rheology.n.set(3.0)
    mg.rheology.eps_reg.set(1e-6)
    mg.rheology.stress_balance.set(1.0)
    mg.rheology.n_sigma.set(8.0)
    mg.rheology.eps_reg_shear.set(1e-12)

    mg.sliding.beta.set(cp.full((ny, nx), 2.5, cp.float32))
    mg.sliding.m.set(m)
    mg.sliding.u_reg.set(1.0)
    mg.sliding.water_drag.set(1e-4)
    return mg


def sweep_trajectory(m):
    """max|u| after each single-level smoothing sweep, coefficients refreshed between."""
    mg = build(m)
    g = mg.levels[0]
    fo = g.forward_operators
    fo.compute_diva_coeffs()
    traj = []
    for _ in range(N_SWEEPS):
        fo.vanka_sweep(dt, 1)
        fo.compute_diva_coeffs()
        traj.append(float(cp.abs(cp.nan_to_num(g.state.u.data)).max()))
    del mg
    cp.get_default_memory_pool().free_all_blocks()
    return traj


def main():
    print(f'max|u| after each of {N_SWEEPS} smoothing sweeps, dt = {float(dt):g} a:')

    lin = sweep_trajectory(1.0)
    print('  m=1   (control)   ' + ' '.join(f'{t:9.3g}' for t in lin))
    assert np.all(np.isfinite(lin)), f'non-finite iterate for a LINEAR sliding law: {lin}'
    assert max(lin) < GROWTH_BOUND, (
        f'the smoother is unstable even for a linear sliding law (max|u| = {max(lin):.3g}). '
        f'A drag-linearization problem cannot show up at m = 1, where the sliding '
        f'coefficient does not depend on speed -- look elsewhere.')

    sub = sweep_trajectory(1. / 3.)
    print('  m=1/3 (regression)' + ' '.join(f'{t:9.3g}' for t in sub))
    assert np.all(np.isfinite(sub)), f'non-finite iterate at m = 1/3: {sub}'
    assert max(sub) < GROWTH_BOUND, (
        f'the DIVA smoother diverges for a sublinear sliding law: max|u| reached '
        f'{max(sub):.3g} over {N_SWEEPS} sweeps. Something has reintroduced a velocity '
        f'dependence of the drag coefficient into the local block -- beta_eff must enter '
        f'it as a FROZEN coefficient, like eta_bar. See notes/diva_numerics.md 5.2.')

    # The two laws must stay comparable: a sublinear law is not intrinsically harder for a
    # frozen-coefficient block, and a large gap would be the early warning.
    assert max(sub) < 5.0 * max(lin), (
        f'the sublinear case is far livelier than the linear control '
        f'({max(sub):.3g} vs {max(lin):.3g}), which is how the old instability began.')

    print(f'\nOK: bounded for both laws (m=1 peak {max(lin):.3g}, m=1/3 peak {max(sub):.3g})')


if __name__ == '__main__':
    main()
