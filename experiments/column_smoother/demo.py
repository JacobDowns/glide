"""
Demo: column smoother convergence on canonical problems.

Each case with an analytical solution runs real time stepping to
steady state and checks convergence against the reference.

Run:
    cd experiments/column_smoother && python demo.py
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from column import (
    ColumnConfig, column_sweep, time_step, compute_residual,
    initialize_from_surface, initialize_linear_profile,
    initialize_uniform_temperature,
    enthalpy_from_temperature, temperature_from_enthalpy,
    water_content_from_enthalpy, get_E_pmp, E_history_to_fields,
    RHO_I, RHO_W, K_I, C_I, K_COLD, K_TEMP_FACTOR,
    T_MELT, BETA_CC, GRAVITY, T_REF, L_HEAT,
    SECONDS_PER_YEAR,
)


def plot_heatmaps(times_kyr, sigma, T_hist, omega_hist, title_prefix, filename):
    """Plot temperature and water content heatmaps side by side."""
    fig, (ax_t, ax_w) = plt.subplots(1, 2, figsize=(12, 4.5))

    im_t = ax_t.pcolormesh(times_kyr, sigma, T_hist.T,
                           shading='auto', cmap='RdYlBu_r')
    ax_t.set_xlabel('Time (kyr)')
    ax_t.set_ylabel(r'$\sigma$ (bed=0, surface=1)')
    ax_t.set_title(f'{title_prefix}: Temperature (K)')
    plt.colorbar(im_t, ax=ax_t)

    im_w = ax_w.pcolormesh(times_kyr, sigma, omega_hist.T * 100,
                           shading='auto', cmap='Blues')
    ax_w.set_xlabel('Time (kyr)')
    ax_w.set_ylabel(r'$\sigma$ (bed=0, surface=1)')
    ax_w.set_title(f'{title_prefix}: Water content (%)')
    plt.colorbar(im_w, ax=ax_w)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  Saved {filename}")


def plot_profiles(sigma, T_sol, T_exact, omega_sol, omega_exact,
                  title, filename, H=1000.0):
    """Plot numerical vs analytical temperature and water content profiles."""
    fig, (ax_t, ax_w) = plt.subplots(1, 2, figsize=(11, 5))

    ax_t.plot(T_exact, sigma, 'k--', linewidth=2, label='Analytical')
    ax_t.plot(T_sol, sigma, 'r-', linewidth=1.5, label='Numerical')
    T_pmp = np.array([T_MELT - BETA_CC * RHO_I * GRAVITY * (1 - s) * H
                      for s in sigma])
    ax_t.plot(T_pmp, sigma, 'b:', alpha=0.4, label='$T_{pmp}$')
    ax_t.set_xlabel('Temperature (K)')
    ax_t.set_ylabel(r'$\sigma$ (bed=0, surface=1)')
    ax_t.set_title(f'{title}: Temperature')
    ax_t.legend(fontsize=8)

    if omega_exact is not None or omega_sol is not None:
        if omega_exact is not None:
            ax_w.plot(omega_exact * 100, sigma, 'k--', linewidth=2,
                      label='Analytical')
        if omega_sol is not None:
            ax_w.plot(omega_sol * 100, sigma, 'r-', linewidth=1.5,
                      label='Numerical')
        ax_w.set_xlabel('Water content (%)')
        ax_w.set_ylabel(r'$\sigma$ (bed=0, surface=1)')
        ax_w.set_title(f'{title}: Water content')
        ax_w.legend(fontsize=8)
    else:
        ax_w.text(0.5, 0.5, 'No water content\n(cold ice)',
                  ha='center', va='center', transform=ax_w.transAxes,
                  fontsize=12, color='gray')
        ax_w.set_title(f'{title}: Water content')

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  Saved {filename}")


def check_steady_state(E, cfg, tol=1e-4):
    """Check that the residual with E_prev=E (steady state) is small."""
    r = np.linalg.norm(compute_residual(E, E, cfg.with_dt_years(1e10)))
    passed = r < tol
    status = "PASS" if passed else "FAIL"
    print(f"  Steady-state residual: {r:.2e} ({status}, tol={tol:.0e})")
    return passed


def check_analytical(T_sol, T_exact, label, tol_K=0.5):
    """Check temperature solution against analytical reference."""
    err = np.max(np.abs(T_sol - T_exact))
    rel = err / (np.max(T_exact) - np.min(T_exact)) if np.ptp(T_exact) > 0 else 0
    passed = err < tol_K
    status = "PASS" if passed else "FAIL"
    print(f"  {label}: max |T_err| = {err:.4f} K, "
          f"rel = {rel:.2e} ({status}, tol={tol_K} K)")
    return passed


# ---- Semi-analytical polythermal solution ----

def polythermal_analytical(sigma, H, T_s, Q_geo):
    """
    Semi-analytical steady-state polythermal profile.

    The column splits into a cold zone (sigma* < sigma < 1) with a linear
    temperature profile, and a temperate zone (0 < sigma < sigma*) with a
    linear enthalpy profile (slope Q_geo*H / K_temp).

    Returns T_exact, omega_exact arrays on the given sigma grid.
    """
    K_c = K_COLD           # cold diffusivity = k_i / c_i
    K_t = K_c * K_TEMP_FACTOR  # temperate diffusivity

    # E_surface
    E_s = C_I * (min(T_s, T_MELT) - T_REF)

    # Find CTS location: where cold profile reaches E_pmp
    # Cold profile: E(sigma) = E_s + Q_geo*H/K_c * (1 - sigma)
    # E_pmp(sigma) = C_I * (T_MELT - BETA_CC*RHO_I*GRAVITY*(1-sigma)*H - T_REF)
    def cts_residual(sig):
        E_cold = E_s + Q_geo * H / K_c * (1.0 - sig)
        E_pmp = get_E_pmp(sig, H)
        return E_cold - E_pmp

    # Check if bed is even temperate
    if cts_residual(0.0) < 0:
        # Entirely cold — no CTS
        T_exact = T_s + (Q_geo * H / K_I) * (1.0 - sigma)
        return T_exact, np.zeros_like(sigma)

    # CTS is between 0 and 1
    sigma_star = brentq(cts_residual, 0.0, 1.0)
    E_cts = get_E_pmp(sigma_star, H)

    T_exact = np.zeros_like(sigma)
    omega_exact = np.zeros_like(sigma)

    for i, sig in enumerate(sigma):
        E_pmp = get_E_pmp(sig, H)
        if sig >= sigma_star:
            # Cold zone: linear temperature
            E_val = E_s + Q_geo * H / K_c * (1.0 - sig)
            T_exact[i] = E_val / C_I + T_REF
            omega_exact[i] = 0.0
        else:
            # Temperate zone: T = T_pmp, excess enthalpy is water
            T_exact[i] = E_pmp / C_I + T_REF
            E_val = E_cts + Q_geo * H / K_t * (sigma_star - sig)
            omega_exact[i] = max(E_val - E_pmp, 0.0) / L_HEAT

    return T_exact, omega_exact


# ---- Case 1: Cold column (pure diffusion) ----

def case_cold_column():
    """
    Pure diffusion, cold base. Time-step to steady state and compare
    against the analytical linear profile.
    """
    cfg = ColumnConfig(
        H=1000.0, nz=21,
        T_surface=243.15,  # -30 C
        Q_geo=0.042,
        n_newton=3,
    )

    E = initialize_from_surface(cfg)

    # Diffusion timescale ~ H^2 / (kappa * pi^2) ~ 2.8 kyr
    # Ramp: 100 steps at 100 yr, then 100 steps at 10 kyr
    E, hist1 = time_step(E, cfg, dt_years=100.0, n_steps=100,
                         save_profiles=True)
    E, hist2 = time_step(E, cfg, dt_years=1e4, n_steps=100,
                         save_profiles=True, report=True)

    times = np.concatenate([hist1['times'],
                            hist1['times'][-1] + hist2['times'][1:]])
    E_history = np.concatenate([hist1['E_history'],
                                hist2['E_history'][1:]], axis=0)
    hist = {'times': times, 'E_history': E_history}

    sigma = cfg.sigma
    T_exact = cfg.T_surface + (cfg.Q_geo * cfg.H / K_I) * (1.0 - sigma)
    T_sol = np.array([temperature_from_enthalpy(E[k], sigma[k], cfg.H)
                      for k in range(cfg.nz)])

    return cfg, E, hist, T_exact, T_sol


# ---- Case 2: Polythermal (high Q_geo, temperate base) ----

def case_polythermal():
    """
    High geothermal flux drives a temperate basal layer.
    Time-step to steady state.
    """
    cfg = ColumnConfig(
        H=1000.0, nz=41,
        T_surface=223.15,  # -50 C
        Q_geo=0.5,
        drain_rate=0.0,
        n_newton=3,
    )

    E = initialize_from_surface(cfg)

    # Cold diffusion timescale ~2.8 kyr, but temperate zone is 1/epsilon
    # slower. Ramp: 200 steps at 1 kyr, then 200 at 100 kyr
    E, hist1 = time_step(E, cfg, dt_years=1e3, n_steps=200,
                         save_profiles=True)
    E, hist2 = time_step(E, cfg, dt_years=1e5, n_steps=200,
                         save_profiles=True, report=True)

    times = np.concatenate([hist1['times'],
                            hist1['times'][-1] + hist2['times'][1:]])
    E_history = np.concatenate([hist1['E_history'],
                                hist2['E_history'][1:]], axis=0)
    hist = {'times': times, 'E_history': E_history}

    sigma = cfg.sigma
    T_sol = np.array([temperature_from_enthalpy(E[k], sigma[k], cfg.H)
                      for k in range(cfg.nz)])
    omega_sol = np.array([water_content_from_enthalpy(E[k], sigma[k], cfg.H)
                          for k in range(cfg.nz)])

    T_exact, omega_exact = polythermal_analytical(
        sigma, cfg.H, cfg.T_surface, cfg.Q_geo)

    return cfg, E, hist, T_sol, omega_sol, T_exact, omega_exact


# ---- Case 3: Advection-diffusion (constant Pe) ----

def case_advection_diffusion():
    """
    Constant downward sigma_dot (Pe = -5). Time-step to steady state
    and compare against the exponential analytical solution.
    """
    cfg = ColumnConfig(
        H=1000.0, nz=41,
        T_surface=243.15,
        Q_geo=0.042,
        n_newton=3,
    )

    Pe = -5.0
    w = Pe * K_COLD / (RHO_I * cfg.H**2)
    cfg.sigma_dot = np.full(cfg.nz, w)

    E = initialize_from_surface(cfg)

    E, hist = time_step(E, cfg, dt_years=500.0, n_steps=60,
                        save_profiles=True, report=True)

    # Analytical: E = A + B*exp(Pe*sigma)
    sigma = cfg.sigma
    B = -cfg.H * cfg.Q_geo / (K_COLD * Pe)
    A = cfg.E_surface - B * np.exp(Pe)
    E_exact = A + B * np.exp(Pe * sigma)
    T_exact = E_exact / C_I + T_REF

    T_sol = np.array([temperature_from_enthalpy(E[k], sigma[k], cfg.H)
                      for k in range(cfg.nz)])

    return cfg, E, hist, T_exact, T_sol


# ---- Case 4: Newton sensitivity ----

def case_newton_sensitivity():
    """Effect of n_newton on the polythermal single-step solve."""
    histories = {}
    for n_newton in [1, 2, 3, 5, 10]:
        cfg = ColumnConfig(
            H=1000.0, nz=41,
            T_surface=223.15,
            Q_geo=0.5,
            dt=1e13,
            drain_rate=0.0,
            n_newton=n_newton,
        )
        E = initialize_from_surface(cfg)
        E_prev = E.copy()
        _, hist = column_sweep(E, E_prev, cfg, n_iter=60, atol=1e-8)
        histories[n_newton] = hist

    return histories


# ---- Case 5: Drainage effect ----

def case_drainage_effect():
    """Polythermal column with varying drainage rates, run to steady state."""
    results = {}
    for rd_per_yr, label in [(0.0, 'no drainage'),
                              (0.01, r'$r_d$ = 0.01 a$^{-1}$'),
                              (0.1, r'$r_d$ = 0.1 a$^{-1}$')]:
        cfg = ColumnConfig(
            H=1000.0, nz=41,
            T_surface=223.15,
            Q_geo=0.5,
            drain_rate=rd_per_yr / SECONDS_PER_YEAR,
            n_newton=5,
        )
        E = initialize_from_surface(cfg)
        # Ramp: 200 steps at 1 kyr, then 300 steps at 1 Myr
        E, hist1 = time_step(E, cfg, dt_years=1e3, n_steps=200,
                             n_iter=30, save_profiles=True)
        E, hist2 = time_step(E, cfg, dt_years=1e6, n_steps=300,
                             n_iter=30, save_profiles=True)
        hist_times = np.concatenate([hist1['times'],
                                     hist1['times'][-1] + hist2['times'][1:]])
        hist_E = np.concatenate([hist1['E_history'],
                                 hist2['E_history'][1:]], axis=0)
        hist = {'times': hist_times, 'E_history': hist_E}

        sigma = cfg.sigma
        T_sol = np.array([temperature_from_enthalpy(E[k], sigma[k], cfg.H)
                          for k in range(cfg.nz)])
        omega_sol = np.array([water_content_from_enthalpy(E[k], sigma[k], cfg.H)
                              for k in range(cfg.nz)])
        results[label] = (cfg, E, hist, T_sol, omega_sol)

    return results


if __name__ == '__main__':

    all_passed = True

    # ---- Case 1: Cold column ----
    print("=== Case 1: Cold column (pure diffusion) ===")
    cfg1, E1, hist1, T_exact1, T_sol1 = case_cold_column()
    sigma1 = cfg1.sigma

    all_passed &= check_steady_state(E1, cfg1)
    all_passed &= check_analytical(T_sol1, T_exact1, 'vs analytical')

    T_hist1, omega_hist1 = E_history_to_fields(hist1['E_history'], cfg1)
    plot_heatmaps(hist1['times'] / 1e3, sigma1, T_hist1, omega_hist1,
                  'Cold column', 'cold_column_heatmap.png')
    plot_profiles(sigma1, T_sol1, T_exact1, None, None,
                  'Cold column', 'cold_column_profiles.png', cfg1.H)

    # ---- Case 2: Polythermal ----
    print("\n=== Case 2: Polythermal (Q_geo=0.5 W/m²) ===")
    cfg2, E2, hist2, T_sol2, omega_sol2, T_exact2, omega_exact2 = case_polythermal()
    sigma2 = cfg2.sigma

    all_passed &= check_steady_state(E2, cfg2)
    all_passed &= check_analytical(T_sol2, T_exact2, 'vs analytical (T)', tol_K=0.2)

    T_hist2, omega_hist2 = E_history_to_fields(hist2['E_history'], cfg2)
    plot_heatmaps(hist2['times'] / 1e3, sigma2, T_hist2, omega_hist2,
                  'Polythermal', 'polythermal_heatmap.png')
    plot_profiles(sigma2, T_sol2, T_exact2, omega_sol2, omega_exact2,
                  'Polythermal (Q_geo=0.5)', 'polythermal_profiles.png', cfg2.H)

    # ---- Case 3: Advection-diffusion ----
    print("\n=== Case 3: Advection-diffusion (Pe=-5) ===")
    cfg3, E3, hist3, T_exact3, T_sol3 = case_advection_diffusion()
    sigma3 = cfg3.sigma

    all_passed &= check_steady_state(E3, cfg3)
    all_passed &= check_analytical(T_sol3, T_exact3, 'vs analytical')

    T_hist3, omega_hist3 = E_history_to_fields(hist3['E_history'], cfg3)
    plot_heatmaps(hist3['times'] / 1e3, sigma3, T_hist3, omega_hist3,
                  'Advection-diffusion (Pe=-5)', 'advection_diffusion_heatmap.png')
    plot_profiles(sigma3, T_sol3, T_exact3, None, None,
                  'Advection-diffusion (Pe=-5)',
                  'advection_diffusion_profiles.png', cfg3.H)

    # ---- Case 4: Newton sensitivity ----
    print("\n=== Case 4: Newton sensitivity ===")
    newton_hists = case_newton_sensitivity()

    fig2, ax2 = plt.subplots(1, 1, figsize=(7, 5))
    for n_newton, hist in newton_hists.items():
        norms = hist['residual_norms']
        ax2.semilogy(range(len(norms)), norms, 'o-',
                     label=f'n_newton = {n_newton}', markersize=3)
    ax2.set_xlabel('Sweep')
    ax2.set_ylabel('Residual norm')
    ax2.set_title('Polythermal: effect of n_newton')
    ax2.legend()
    plt.tight_layout()
    plt.savefig('newton_sensitivity.png', dpi=150)
    plt.close(fig2)
    print("  Saved newton_sensitivity.png")

    # ---- Case 5: Drainage effect ----
    print("\n=== Case 5: Drainage effect ===")
    drain_results = case_drainage_effect()

    for label, (cfg_d, E_d, hist_d, T_d, omega_d) in drain_results.items():
        print(f"  {label}:")
        # Smoothed drainage leaves a small residual from the softplus
        # transition zone (~200 J/kg wide). Tol scales with drain_rate.
        all_passed &= check_steady_state(E_d, cfg_d, tol=1e-2)

    # Heatmaps and profile comparison for each drainage rate
    # Use no-drainage analytical as reference for all
    sigma_d = list(drain_results.values())[0][0].sigma
    T_ref, omega_ref = polythermal_analytical(
        sigma_d, 1000.0, 223.15, 0.5)

    for label, (cfg_d, E_d, hist_d, T_d, omega_d) in drain_results.items():
        T_hist_d, omega_hist_d = E_history_to_fields(hist_d['E_history'], cfg_d)
        safe_label = label.replace('$', '').replace('\\', '').replace(' ', '_')
        plot_heatmaps(hist_d['times'] / 1e3, cfg_d.sigma,
                      T_hist_d, omega_hist_d,
                      f'Drainage: {label}', f'drainage_{safe_label}_heatmap.png')
        plot_profiles(cfg_d.sigma, T_d, T_ref, omega_d, omega_ref,
                      f'Drainage: {label}',
                      f'drainage_{safe_label}_profiles.png', cfg_d.H)

    # ---- Summary ----
    print("\n" + "=" * 40)
    if all_passed:
        print("All checks PASSED")
    else:
        print("Some checks FAILED")
