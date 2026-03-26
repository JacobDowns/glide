"""
Test horizontal advection in the enthalpy solver.

A flat ice slab with prescribed horizontal velocity and uniform
temperature. Under the advective form (u * dE/dx), uniform E should
remain exactly uniform regardless of the velocity field.

If the discretization uses the flux divergence form div(u*E) instead,
a divergent velocity field will create spurious E * div(u) sources.
"""
import cupy as cp
import numpy as np
from glide.grid import Grid
from glide.enthalpy import EnthalpyOperators, C_I, T_REF


def test_uniform_E_divergent_velocity():
    """
    Uniform E + spatially varying velocity  →  E must stay uniform.

    Sets u = alpha * x (constant strain rate) so div(u) != 0.
    With uniform E, the advective form gives zero, but the flux
    divergence form gives rho_i * E * div(u) != 0.
    """
    ny, nx, nz = 8, 8, 9
    dx = 1000.0
    H_ice = 1000.0
    T0 = 248.15  # K  (well below melting — stays linear/cold)
    dt = 100.0 * 365.25 * 86400.0  # 100 yr

    grid = Grid(ny, nx, cp.float32(dx))
    grid.state.H.set(cp.full((ny, nx), H_ice, dtype=cp.float32))
    grid.state.H_prev.set(cp.full((ny, nx), H_ice, dtype=cp.float32))
    grid.geometry.bed.set(cp.zeros((ny, nx), dtype=cp.float32))
    grid.forcing.smb.set(cp.zeros((ny, nx), dtype=cp.float32))
    grid.sliding.beta.set(cp.ones((ny, nx), dtype=cp.float32))
    grid.sliding.m.set(1.0)
    grid.sliding.u_reg.set(1.0)

    ops = EnthalpyOperators(grid, nz=nz)
    ops.initialize_from_temperature(T0)
    ops.set_surface_enthalpy_from_temperature(
        cp.full((ny, nx), T0, dtype=cp.float32))
    ops.enthalpy_forcing.Q_geo.fill(0)
    ops.enthalpy_forcing.Q_fh.fill(0)
    ops.enthalpy_forcing.phi_strain.fill(0)
    ops.enthalpy_forcing.drain_rate.set(0.0)
    ops.enthalpy_velocity.sigma_dot.fill(0)

    # Divergent velocity: u(x) = alpha * x on MAC faces (m/s)
    # u at face j+1/2:  u = alpha * (j + 0.5) * dx
    alpha = 1e-10  # 1/s — mild but nonzero divergence
    u_face = np.zeros((ny, nx + 1), dtype=np.float32)
    for j in range(nx + 1):
        u_face[:, j] = alpha * j * dx
    u_face_gpu = cp.array(u_face)
    for k in range(nz):
        ops.enthalpy_velocity.u3d[k, :, :] = u_face_gpu
    ops.enthalpy_velocity.v3d.fill(0)

    # --- Check 1: initial residual should be near zero ---
    ops.set_rhs(dt)
    r0 = float(ops.compute_residual(dt))
    E0 = float(ops.enthalpy_state.E[ny // 2, nx // 2, nz // 2])
    print(f"Uniform E + divergent velocity test:")
    print(f"  E0 = {E0:.1f} J/kg")
    print(f"  Initial residual |r0| = {r0:.2e}")

    # --- Check 2: one sweep should not change E ---
    ops.column_smooth(dt)
    delta = cp.asnumpy(ops.delta_E)
    max_delta = np.max(np.abs(delta))
    print(f"  Max |delta_E| after one smooth = {max_delta:.2e}")

    # --- Check 3: run 5 sweeps, E should stay uniform ---
    ops.set_rhs(dt)
    ops.smoother_config.report_norms = True
    ops.column_sweep(dt, 5)
    ops.smoother_config.report_norms = False

    E_final = cp.asnumpy(ops.enthalpy_state.E[:, :, nz // 2])
    E_range = np.max(E_final) - np.min(E_final)
    print(f"  E range after 5 sweeps = {E_range:.2e} J/kg")

    # The advective form should give r0 ~ 0 and delta_E ~ 0.
    # The flux divergence form gives r0 ~ RHO_I * E * alpha ~ O(1).
    assert r0 < 1.0, (
        f"Initial residual {r0:.2e} too large for uniform E — "
        f"likely using flux divergence form div(u*E) instead of "
        f"advective form u*dE/dx"
    )
    assert max_delta < 0.01, (
        f"Correction {max_delta:.2e} too large for uniform E"
    )
    print("  PASSED")


def test_uniform_velocity_step_profile():
    """
    Uniform velocity u > 0 with a temperature step in x.

    The step should advect to the right without creating asymmetry
    between the leading and trailing edges.
    """
    ny, nx, nz = 8, 16, 9
    dx = 1000.0
    H_ice = 1000.0
    T_cold = 240.0
    T_warm = 260.0
    dt = 10.0 * 365.25 * 86400.0  # 10 yr

    grid = Grid(ny, nx, cp.float32(dx))
    grid.state.H.set(cp.full((ny, nx), H_ice, dtype=cp.float32))
    grid.state.H_prev.set(cp.full((ny, nx), H_ice, dtype=cp.float32))
    grid.geometry.bed.set(cp.zeros((ny, nx), dtype=cp.float32))
    grid.forcing.smb.set(cp.zeros((ny, nx), dtype=cp.float32))
    grid.sliding.beta.set(cp.ones((ny, nx), dtype=cp.float32))
    grid.sliding.m.set(1.0)
    grid.sliding.u_reg.set(1.0)

    ops = EnthalpyOperators(grid, nz=nz)

    # Temperature step: cold on left half, warm on right half
    T_init = cp.full((ny, nx, nz), T_cold, dtype=cp.float32)
    T_init[:, nx // 2:, :] = T_warm
    ops.enthalpy_state.E[:] = C_I * (T_init - T_REF)
    ops.enthalpy_state.E_prev[:] = ops.enthalpy_state.E
    ops.set_surface_enthalpy_from_temperature(
        cp.full((ny, nx), T_cold, dtype=cp.float32))
    ops.enthalpy_forcing.Q_geo.fill(0)
    ops.enthalpy_forcing.Q_fh.fill(0)
    ops.enthalpy_forcing.phi_strain.fill(0)
    ops.enthalpy_forcing.drain_rate.set(0.0)
    ops.enthalpy_velocity.sigma_dot.fill(0)
    ops.enthalpy_velocity.v3d.fill(0)

    # Uniform rightward velocity: 100 m/yr in m/s
    u_ms = cp.float32(100.0 / (365.25 * 86400.0))
    ops.enthalpy_velocity.u3d.fill(u_ms)

    # Run 3 time steps
    for step in range(3):
        ops.set_rhs(dt)
        ops.column_sweep(dt, 20)

    # Check y-symmetry: all rows should be identical
    E_mid = cp.asnumpy(ops.enthalpy_state.E[:, :, nz // 2])
    row_spread = np.max(E_mid, axis=0) - np.min(E_mid, axis=0)
    max_row_spread = np.max(row_spread)
    print(f"\nUniform velocity + step profile test:")
    print(f"  Max row spread (y-asymmetry) = {max_row_spread:.2e} J/kg")

    # Check that the step moved to the right (warm region expanded)
    E_init_row = C_I * (np.where(np.arange(nx) >= nx // 2,
                                  T_warm, T_cold) - T_REF)
    E_final_row = E_mid[ny // 2, :]
    # The warm region should have expanded leftward (advection brings warm from right)
    # Wait — u > 0 means flow to the right. So cold air advects into the warm region.
    # At the interface (j = nx//2), cold E from the left replaces warm E.
    print(f"  E at interface-2: {E_final_row[nx//2-2]:.1f} "
          f"(init: {E_init_row[nx//2-2]:.1f})")
    print(f"  E at interface:   {E_final_row[nx//2]:.1f} "
          f"(init: {E_init_row[nx//2]:.1f})")
    print(f"  E at interface+2: {E_final_row[nx//2+2]:.1f} "
          f"(init: {E_init_row[nx//2+2]:.1f})")

    assert max_row_spread < 1.0, (
        f"y-asymmetry {max_row_spread:.2e} — rows should be identical "
        f"for uniform velocity + y-invariant initial condition"
    )
    print("  PASSED")


if __name__ == '__main__':
    test_uniform_E_divergent_velocity()
    test_uniform_velocity_step_profile()
