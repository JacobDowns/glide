"""
Compare the effect of LF regularization constant on dome symmetry.

Tests lf_c = 1e-20 (near-upwind), 1e-12, 1e-8, 1e-4 on a coupled
dome with all terms enabled.
"""
import cupy as cp
import numpy as np
from glide.model import IceDynamics, ThermalModel
from glide.enthalpy import T_MELT, BETA_CC, RHO_I, GRAVITY

ny, nx = 128, 128
dx = 1000.0
n_levels = 3
DOME_RADIUS = 50000.0
DOME_HEIGHT = 2000.0
SMB_CENTER = 0.5
SMB_EDGE = -2.0
Q_GEO = 0.05
T_SEA_LEVEL = 270.15
LAPSE_RATE = -6.5e-3
T_INIT = 253.15
NZ = 11
N_SMOOTH = 10
DT_YR = 5.0
N_STEPS = 100
SEC_PER_YR = 365.25 * 86400.0

x = (cp.arange(nx, dtype=cp.float32) + 0.5) * dx - 0.5 * nx * dx
y = (cp.arange(ny, dtype=cp.float32) + 0.5) * dx - 0.5 * ny * dx
xx, yy = cp.meshgrid(x, y)
radius = cp.sqrt(xx**2 + yy**2)
radial_fraction = cp.clip(radius / DOME_RADIUS, 0.0, 1.0)
hemisphere = cp.sqrt(cp.clip(1.0 - radial_fraction**2, 0.0, 1.0))

thickness_init = DOME_HEIGHT * hemisphere
bed = cp.zeros((ny, nx), dtype=cp.float32)
smb = cp.float32(SMB_EDGE) + cp.float32(SMB_CENTER - SMB_EDGE) * hemisphere


def surface_temperature(H, bed):
    return cp.minimum(
        cp.float32(T_SEA_LEVEL) + cp.float32(LAPSE_RATE) * (bed + H),
        cp.float32(T_MELT))


lf_c_values = [1e-20, 1e-12, 1e-8, 1e-4]
results = {}

for lf_c_val in lf_c_values:
    print(f"\n{'='*60}")
    print(f"  lf_c = {lf_c_val:.0e}")
    print(f"{'='*60}")

    model = IceDynamics(n_levels=n_levels, ny=ny, nx=nx, dx=cp.float32(dx))
    mg = model.mg
    mg.geometry.bed.set(bed)
    mg.state.H.set(thickness_init.copy())
    mg.state.H_prev.set(thickness_init.copy())
    mg.forcing.smb.set(smb)
    mg.rheology.B.set(cp.ones((ny, nx), dtype=cp.float32))
    mg.rheology.n.set(3.0)
    mg.rheology.eps_reg.set(1e-6)
    mg.sliding.beta.set(cp.full((ny, nx), 10.0, dtype=cp.float32))
    mg.sliding.m.set(1.0 / 3.0)
    mg.sliding.u_reg.set(1.0)
    mg.calving.calving_rate.set(0.0)

    model.forward_solver.fas_options.set(
        coarsest_steps=200, pre_steps=5, post_steps=20, finest_steps=0,
        relative_tolerance=5e-5, absolute_tolerance=1e-3, report_norms=False)

    grid = mg.levels[0]
    thermal = ThermalModel(grid, nz=NZ, n_smooth=N_SMOOTH,
                           update_rheology=True, frictional_heating=False)

    thermal.ops.smoother_config.n_newton = 5
    thermal.ops.smoother_config.lf_c = cp.float32(lf_c_val)
    thermal.ops.enthalpy_forcing.h_thin.set(50.0)

    T_surf = surface_temperature(grid.state.H.data, grid.geometry.bed.data)
    thermal.initialize(T_surface=T_surf, T_field=T_INIT, Q_geo=Q_GEO)
    B_init = thermal.ops.get_arrhenius_factor() / thermal.B_scale
    mg.rheology.B.set(B_init)

    # Spin up
    t_yr = 0.0
    dt_yr = cp.float32(DT_YR)
    for _ in range(3):
        model.forward(cp.float32(t_yr), dt_yr)
        t_yr += DT_YR

    T_surf = surface_temperature(grid.state.H.data, grid.geometry.bed.data)
    thermal.initialize(T_surface=T_surf, T_field=T_INIT, Q_geo=Q_GEO)

    dt_sec = DT_YR * SEC_PER_YR

    H_asym_hist = []
    T_asym_hist = []
    times = []

    for step in range(N_STEPS):
        model.forward(cp.float32(t_yr), dt_yr)

        T_surf = surface_temperature(grid.state.H.data, grid.geometry.bed.data)
        thermal.ops.set_surface_enthalpy_from_temperature(T_surf)

        ops = thermal.ops
        ops.broadcast_velocity()
        sec = cp.float32(SEC_PER_YR)
        ops.enthalpy_velocity.u3d /= sec
        ops.enthalpy_velocity.v3d /= sec
        smb_si = ops.grid.forcing.smb.data / sec
        ops.compute_omega(smb=smb_si)
        ops.set_rhs(dt_sec)
        ops.column_sweep(dt_sec, N_SMOOTH)
        if thermal.update_rheology:
            n = float(ops.grid.rheology.n.value)
            scale = cp.float32(thermal.rho_i * thermal.g * SEC_PER_YR ** (1.0 / n))
            ops.grid.rheology.B.data[:] = ops.get_arrhenius_factor() / scale

        t_yr += DT_YR

        # Symmetry check
        H_np = cp.asnumpy(grid.state.H.data)
        H_asym = np.max(np.abs(H_np - H_np[::-1, ::-1]))
        T_3d = cp.asnumpy(thermal.temperature)
        T_mid = T_3d[:, :, NZ // 2]
        T_asym = np.max(np.abs(T_mid - T_mid[::-1, ::-1]))

        H_asym_hist.append(H_asym)
        T_asym_hist.append(T_asym)
        times.append(t_yr)

        if (step + 1) % 25 == 0:
            print(f"  step {step+1:4d}  t={t_yr:6.0f} yr  "
                  f"H_asym={H_asym:.3e}  T_asym={T_asym:.3e}")

    results[lf_c_val] = {
        'times': np.array(times),
        'H_asym': np.array(H_asym_hist),
        'T_asym': np.array(T_asym_hist),
    }

# Summary plot
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for lf_c_val, res in results.items():
    label = f'lf_c = {lf_c_val:.0e}'
    ax1.semilogy(res['times'], res['H_asym'], label=label)
    ax2.semilogy(res['times'], res['T_asym'], label=label)

ax1.set_xlabel('Time (yr)')
ax1.set_ylabel('Max |H - H_rot|')
ax1.set_title('Thickness asymmetry (180-deg rotation)')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('Time (yr)')
ax2.set_ylabel('Max |T_mid - T_mid_rot|')
ax2.set_title('Temperature asymmetry (180-deg rotation)')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.suptitle('Effect of Lax-Friedrichs regularization on dome symmetry',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('lf_c_comparison.png', dpi=150)
print(f"\nSaved lf_c_comparison.png")
plt.show()
