"""End-to-end DIVA forward solve.

Runs the same slab problem under both stress balances and checks that DIVA behaves
the way the physics requires, rather than against a stored regression:

  * it converges -- the lagged-coefficient smoother (Goldberg's iteration on
    viscosity, his eqs 41-44) must not degrade the FAS convergence rate;
  * it is faster than SSA -- DIVA resolves internal deformation on top of the same
    sliding, so the depth-averaged velocity can only increase;
  * 0 <= u_b <= |ubar| everywhere -- the closure splits the depth-averaged motion
    into sliding plus deformation, so the basal speed cannot exceed the mean;
  * F2 > 0 wherever there is ice, and the solve is deterministic.

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


def solve(stress_balance, vcycles=15):
    """Cold-started, converged solve under the requested stress balance."""
    x = cp.arange(nx, dtype=cp.float32) * dx
    y = cp.arange(ny, dtype=cp.float32) * dx
    X, Y = cp.meshgrid(x, y)
    srf = 1000.0 * cp.ones((ny, nx), dtype=cp.float32) - cp.tan(cp.deg2rad(0.1)) * X + 10000
    bed = srf - 1000.0
    thk = srf - bed
    beta = (1000 * cp.sin(2 * cp.pi * X / L) * cp.sin(2 * cp.pi * Y / L) + 1000) / (RHO_I * GRAV)
    B = cp.ones((ny, nx), dtype=cp.float32)
    B.fill((1e-16 ** -(1. / 3)) / (RHO_I * GRAV))

    mg = Multigrid(n_levels, ny=ny, nx=nx, dx=dx)
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
    solver.fas_options.maximum_vcycles.set(vcycles)
    solver.fas_options.relative_tolerance.set(1e-6)
    solver.fas_options.absolute_tolerance.set(1e-6)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        solver.solve(dt)
    cycles = [l for l in buf.getvalue().splitlines() if "V-cycle" in l]
    final = float(cycles[-1].split("|r|/|r0| =")[1].split(",")[0]) if cycles else float("nan")

    grid = mg.levels[0]
    return dict(u=cp.asnumpy(grid.state.u.data), v=cp.asnumpy(grid.state.v.data),
                u_b=cp.asnumpy(grid.state.u_b.data), F2=cp.asnumpy(grid.rheology.F2.data),
                H=cp.asnumpy(grid.state.H.data), residual=final)


def main():
    ssa = solve(0.0)
    diva = solve(1.0)

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
    # The closure splits ubar into sliding + deformation.
    ubar = np.hypot(0.5 * (diva['u'][:, :-1] + diva['u'][:, 1:]),
                    0.5 * (diva['v'][:-1, :] + diva['v'][1:, :]))
    assert diva['u_b'].min() >= 0.0, "basal speed must be non-negative"
    assert (diva['u_b'] <= ubar + 1e-3).all(), "basal speed cannot exceed the depth-averaged speed"
    assert (diva['F2'][diva['H'] > 1.0] > 0.0).all(), "F2 must be positive where there is ice"

    repeat = solve(1.0)
    assert np.array_equal(diva['u'], repeat['u']), "DIVA solve must be deterministic"
    print("      repeat solve bit-identical: True")

    print("\nOK: DIVA forward solve converges and is physically consistent")


if __name__ == '__main__':
    main()
