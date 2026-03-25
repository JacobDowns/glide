"""
Tests for high-flux enthalpy behavior near the bed.

The critical boundary-condition check is that geothermal heat flux is
still applied after the bed becomes temperate. The old implementation
incorrectly replaced the basal flux with a Dirichlet clamp E_bed=E_pmp.
"""
import cupy as cp
import numpy as np
from glide.grid import Grid
from glide.enthalpy import (
    EnthalpyOperators, C_I, T_REF, T_MELT, RHO_I, GRAVITY, BETA_CC
)


def _make_ops(T_surface=243.15, H_ice=1000.0, nx=4, ny=4, nz=21, dx=1000.0):
    grid = Grid(ny, nx, cp.float32(dx))
    grid.state.H.set(cp.full((ny, nx), H_ice, dtype=cp.float32))
    grid.state.H_prev.set(cp.full((ny, nx), H_ice, dtype=cp.float32))
    grid.geometry.bed.set(cp.zeros((ny, nx), dtype=cp.float32))
    grid.forcing.smb.set(cp.zeros((ny, nx), dtype=cp.float32))
    grid.sliding.beta.set(cp.ones((ny, nx), dtype=cp.float32))
    grid.sliding.m.set(1.0)
    grid.sliding.u_reg.set(1.0)

    ops = EnthalpyOperators(grid, nz=nz, sigma_q=1.0)
    ops.initialize_from_temperature(T_surface)
    ops.set_surface_enthalpy_from_temperature(
        cp.full((ny, nx), T_surface, dtype=cp.float32))
    ops.enthalpy_forcing.phi_strain.fill(0)
    ops.enthalpy_forcing.drain_rate.set(0.0)
    ops.enthalpy_velocity.u3d.fill(0)
    ops.enthalpy_velocity.v3d.fill(0)
    ops.enthalpy_velocity.sigma_dot.fill(0)
    ops.Q_fh.fill(0)
    return ops


def test_temperate_bed_still_applies_geothermal_flux():
    # ---- Parameters ----
    ny, nx = 4, 4
    H_ice = 1000.0
    T_surface = 243.15    # K (-30C)
    Q_geo = 0.50          # W/m^2
    dt = 1000.0 * 365.25 * 86400.0

    ops = _make_ops(T_surface=T_surface, H_ice=H_ice, nx=nx, ny=ny)
    ops.enthalpy_forcing.Q_geo[:, :] = Q_geo

    T_pmp_bed = T_MELT - BETA_CC * RHO_I * GRAVITY * H_ice
    E_pmp_bed = C_I * (T_pmp_bed - T_REF)

    # Put the basal node slightly above pressure melting and flatten the bed
    # gradient so the residual isolates the imposed geothermal-flux term.
    E_temperate = E_pmp_bed + 1000.0
    ops.enthalpy_state.E.fill(E_temperate)
    ops.enthalpy_state.E_prev.fill(E_temperate)

    ops.compute_residual(dt)
    bed_residual = float(cp.asnumpy(ops.r_E[ny // 2, nx // 2, 0]))
    dsig_half = 0.5 * float(ops.sigma[1] - ops.sigma[0])
    expected = -Q_geo / (H_ice * dsig_half)

    print("Temperate bed geothermal-flux test:")
    print(f"  Bed residual: {bed_residual:.6e}")
    print(f"  Expected:     {expected:.6e}")

    assert np.isclose(bed_residual, expected, rtol=1e-5, atol=1e-8), (
        "Temperate bed should still see the geothermal-flux residual term"
    )


def test_temperate_bed_is_not_clamped_to_pmp():
    H_ice = 1000.0
    T_surface = 243.15
    Q_geo = 0.50
    dt = 1000.0 * 365.25 * 86400.0

    ops = _make_ops(T_surface=T_surface, H_ice=H_ice)
    ops.enthalpy_forcing.Q_geo[:, :] = Q_geo

    T_pmp_bed = T_MELT - BETA_CC * RHO_I * GRAVITY * H_ice
    E_pmp_bed = C_I * (T_pmp_bed - T_REF)
    E_temperate = E_pmp_bed + 1000.0

    ops.enthalpy_state.E.fill(E_temperate)
    ops.enthalpy_state.E_prev.fill(E_temperate)

    ops.column_sweep(dt, 5)
    ops.enthalpy_state.E[:] += ops.smoother_config.omega * ops.delta_E

    E_bed = float(cp.asnumpy(ops.enthalpy_state.E[0, 0, 0]))
    print("Temperate bed unclamped test:")
    print(f"  E_bed after sweep: {E_bed:.2f}")
    print(f"  E_pmp at bed:      {E_pmp_bed:.2f}")

    assert E_bed > E_pmp_bed, "Basal enthalpy should not be hard-clamped to E_pmp"


if __name__ == '__main__':
    test_temperate_bed_still_applies_geothermal_flux()
    test_temperate_bed_is_not_clamped_to_pmp()
