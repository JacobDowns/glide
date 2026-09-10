"""End-to-end DIVA forward solve.

Runs the same slab problem under both stress balances and checks that DIVA behaves
the way the physics requires, rather than against a stored regression:

  * it converges -- the lagged-coefficient smoother (Goldberg's iteration on
    viscosity, his eqs 41-44) must not degrade the FAS convergence rate;
  * it is faster than SSA -- DIVA resolves internal deformation on top of the same
    sliding, so the depth-averaged velocity can only increase;
  * 0 <= u_b <= |ubar| everywhere -- the closure splits the depth-averaged motion
    into sliding plus deformation, so the basal speed cannot exceed the mean;
  * F2 > 0 wherever there is ice, and the solve is deterministic;
  * the answer does not depend on multigrid depth -- the coarse-grid correction has to
    be consistent with the DIVA operator, since each level diagnoses its own
    eta_bar/F2/beta_eff from its own restricted state;
  * it converges onto SSA as the ice stiffens -- the END-TO-END SSA limit. The vertical
    shear strain rate scales like B^-n, so stiffening the ice must drive the DIVA
    solution onto the SSA one. The coefficient-level version of this limit is
    diva_closure_test check 1 (eta_bar collapses onto the membrane viscosity at
    tau_b = 0) and the SIA limit is diva_slab_test; this is the only one that exercises
    the whole solve.

On the unified tree the balance is a COMPILE-time choice (`stress_scheme=`), so the SSA and
DIVA runs are separately-built grids rather than one grid with a runtime flag.

    uv run python tests/diva_solve_test.py
"""
import contextlib
import io

import cupy as cp
import numpy as np

from glide.multigrid import Multigrid, FASCDSolver

ny = nx = 128
dx = 2000.0
dt = cp.float32(1.0)
n_levels = 5
L = 20000.0                      # beta oscillation wavelength
RHO_I = cp.float32(917.0)
GRAV = cp.float32(9.81)


def solve(scheme, vcycles=15, levels=None, B_scale=1.0):
    """Cold-started, converged solve under the requested stress balance."""
    x = cp.arange(nx, dtype=cp.float32) * dx
    y = cp.arange(ny, dtype=cp.float32) * dx
    X, Y = cp.meshgrid(x, y)
    srf = 1000.0 * cp.ones((ny, nx), dtype=cp.float32) - cp.tan(cp.deg2rad(0.1)) * X + 10000
    bed = srf - 1000.0
    thk = srf - bed
    beta = (1000 * cp.sin(2 * cp.pi * X / L) * cp.sin(2 * cp.pi * Y / L) + 1000) / (RHO_I * GRAV)
    B = cp.ones((ny, nx), dtype=cp.float32)
    B.fill(B_scale * (1e-16 ** -(1. / 3)) / (RHO_I * GRAV))

    mg = Multigrid(levels or n_levels, ny=ny, nx=nx, dx=dx, stress_scheme=scheme)
    mg.geometry.bed.set(bed)
    mg.rheology.B.set(B)
    mg.state.H.set(thk)
    mg.state.H_prev.set(thk)
    mg.sliding.beta.set(beta)
    mg.sliding.m.set(1.0)
    mg.sliding.u_reg.set(1.0)
    for lvl in mg.levels:                    # DIVA-only, harmless under SSA; set on every level
        lvl.rheology.n_sigma.set(cp.float32(8.0))

    solver = FASCDSolver(mg)
    solver.vanka_options.omega.set(0.5)
    solver.vanka_options.newton_options.relaxation.set(0.5)
    solver.vanka_options.newton_options.steps.set(30)
    solver.fas_options.coarsest_steps.set(200)
    solver.fas_options.pre_steps.set(10)
    solver.fas_options.post_steps.set(50)
    solver.fas_options.finest_steps.set(150)
    solver.fas_options.maximum_vcycles.set(vcycles)
    solver.fas_options.relative_tolerance.set(1e-6)
    solver.fas_options.absolute_tolerance.set(1e-6)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        solver.solve(dt)
    cycles = [l for l in buf.getvalue().splitlines() if "V-cycle" in l]
    final = float(cycles[-1].split("|r|/|r0| =")[1].split(",")[0]) if cycles else float("nan")

    grid = mg.levels[0]
    # u_b / F2 are DIVA-only fields (None under SSA); the SSA result never reads them.
    u_b = cp.asnumpy(grid.state.u_b.data) if grid.state.u_b is not None else None
    F2 = cp.asnumpy(grid.rheology.F2.data) if grid.rheology.F2 is not None else None
    return dict(u=cp.asnumpy(grid.state.u.data), v=cp.asnumpy(grid.state.v.data),
                u_b=u_b, F2=F2, H=cp.asnumpy(grid.state.H.data), residual=final)


def cell_speed(r):
    """Cell-centred |ubar| from the staggered components, which is what u_b is comparable to."""
    u_ctr = 0.5 * (r['u'][:, :-1] + r['u'][:, 1:])
    v_ctr = 0.5 * (r['v'][:-1, :] + r['v'][1:, :])
    return np.hypot(u_ctr, v_ctr)


def main():
    ssa = solve('ssa')
    diva = solve('diva')

    ssa_speed = np.hypot(ssa['u'][:, :-1], ssa['v'][:-1, :]).max()
    diva_speed = np.hypot(diva['u'][:, :-1], diva['v'][:-1, :]).max()

    print(f"SSA : |r|/|r0| = {ssa['residual']:.2e}, max speed = {ssa_speed:.4f} m/a")
    print(f"DIVA: |r|/|r0| = {diva['residual']:.2e}, max speed = {diva_speed:.4f} m/a")
    print(f"      u_b in [{diva['u_b'].min():.4f}, {diva['u_b'].max():.4f}] m/a")
    print(f"      F2  in [{diva['F2'].min():.4e}, {diva['F2'].max():.4e}]")
    print(f"      DIVA/SSA speed ratio = {diva_speed / ssa_speed:.4f}")

    assert np.isfinite(diva['u']).all() and np.isfinite(diva['v']).all(), \
        "DIVA produced non-finite velocities"
    # Convergence must not be materially worse than SSA's.
    assert diva['residual'] < 10 * ssa['residual'], \
        f"DIVA converged far worse than SSA ({diva['residual']:.2e} vs {ssa['residual']:.2e})"
    # DIVA adds deformation on top of the same sliding law.
    assert diva_speed >= ssa_speed, "DIVA should not be slower than SSA"
    # The closure splits ubar into sliding + deformation, so u_b <= U_bar.  The closure's U_bar is
    # the SUM-OF-SQUARES cell speed sqrt(0.5*(u_l^2+u_r^2)+0.5*(v_t^2+v_b^2)) -- the sum-of-squares
    # form is the self-adjointness fix -- which is >= |mean velocity|, so u_b is compared against it,
    # not against the magnitude of the averaged velocity.
    ubar = np.sqrt(0.5 * (diva['u'][:, :-1] ** 2 + diva['u'][:, 1:] ** 2) +
                   0.5 * (diva['v'][:-1, :] ** 2 + diva['v'][1:, :] ** 2))
    assert diva['u_b'].min() >= 0.0, "basal speed must be non-negative"
    assert (diva['u_b'] <= ubar + 1e-3).all(), "basal speed cannot exceed the depth-averaged speed"
    assert (diva['F2'][diva['H'] > 1.0] > 0.0).all(), "F2 must be positive where there is ice"

    repeat = solve('diva')
    assert np.array_equal(diva['u'], repeat['u']), "DIVA solve must be deterministic"
    print("      repeat solve bit-identical: True")

    # Multigrid is only the solver: every depth must reach the same solution, which
    # is what shows the coarse-grid correction is consistent with the DIVA operator.
    print("\nDIVA across multigrid depths:")
    single = solve('diva', levels=1)
    for depth in (1, 3, 5):
        got = solve('diva', levels=depth)
        rel = np.abs(got['u'] - single['u']).max() / max(np.abs(single['u']).max(), 1e-30)
        print(f"  n_levels={depth}: |r|/|r0| = {got['residual']:.2e}, "
              f"max|u| = {np.abs(got['u']).max():.4f}, rel diff vs single grid = {rel:.2e}")
        assert got['residual'] < 1e-2, f"n_levels={depth} did not converge"
        assert rel < 2e-3, f"n_levels={depth} disagrees with the single-grid solution"

    # ---- end-to-end SSA limit: stiffen the ice and DIVA must collapse onto SSA ----
    print("\nSSA limit -- stiffening the ice removes the vertical shear:")
    prev = None
    for B_scale in (1.0, 4.0, 16.0):
        ssa = solve('ssa', B_scale=B_scale)
        div = solve('diva', B_scale=B_scale)
        # u_b is a cell-centred SPEED, so it has to be compared against the cell-centred speed,
        # not against the x-component of the facet velocity.
        sp_s, sp_d = cell_speed(ssa), cell_speed(div)
        rel = abs(sp_d.mean() - sp_s.mean()) / sp_s.mean()
        deform = 1.0 - div['u_b'].mean() / max(sp_d.mean(), 1e-30)
        print(f"  B x {B_scale:<5g} DIVA-SSA = {rel:8.2%}   deformation fraction = {deform:7.2%}")
        if prev is not None:
            assert rel < prev, (
                f"DIVA does not approach SSA as the ice stiffens: {rel:.2%} at B x {B_scale} "
                f"is not below {prev:.2%} at the previous scale")
        prev = rel
    assert prev < 5e-3, (
        f"DIVA has not collapsed onto SSA in the stiff limit: {prev:.2%}. The vertical shear "
        f"scales like B^-n, so it must.")

    print("\nOK: DIVA forward solve converges, is physically consistent, and recovers SSA")


if __name__ == '__main__':
    main()
