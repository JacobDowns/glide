"""
Demo: warm surface (at the pressure melting point).

The surface is at T_pmp, so the entire column starts temperate or
near-temperate. This stresses the K(E) nonlinearity because the
diffusivity transitions across the whole column rather than being
confined to a thin basal layer.

Cases:
  1. Warm surface, moderate Q_geo — steady state time evolution
  2. Warm surface, high Q_geo — large temperate zone with water content
  3. Warm surface + downward advection — competes with diffusion
  4. Effect of drainage on the warm-surface column

Run:
    cd experiments/column_smoother && python demo_warm_surface.py
"""
import numpy as np
import matplotlib.pyplot as plt
from column import (
    ColumnConfig, column_sweep, time_step,
    initialize_from_surface, initialize_uniform_temperature,
    enthalpy_from_temperature, temperature_from_enthalpy,
    water_content_from_enthalpy, get_E_pmp, E_history_to_fields,
    RHO_I, K_I, C_I, K_COLD, T_MELT, BETA_CC, GRAVITY, T_REF, L_HEAT,
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
    print(f"Saved {filename}")


# ---- Case 1: warm surface, time evolution to steady state ----

def case_warm_surface_evolution():
    """
    Surface at PMP, moderate geothermal flux. Start from uniform
    cold ice and watch the warm surface diffuse downward over time.
    """
    cfg = ColumnConfig(
        H=1000.0, nz=21,
        T_surface=T_MELT,
        Q_geo=0.042,
        n_newton=3,
    )

    E = initialize_uniform_temperature(243.15, cfg)  # -30 C initial

    dt_yr = 50.0
    n_steps = 1000  # 50 kyr total

    E, hist = time_step(E, cfg, dt_years=dt_yr, n_steps=n_steps,
                        save_profiles=True, report=True)

    return cfg, hist


# ---- Case 2: warm surface + high Q_geo ----

def case_warm_surface_high_qgeo():
    """
    Surface at PMP with high geothermal flux. The entire column
    may become temperate with significant water content.
    """
    results = {}
    for Q_geo, label in [(0.042, 'Q_geo=42mW'),
                          (0.15, 'Q_geo=150mW'),
                          (0.5, 'Q_geo=500mW')]:
        cfg = ColumnConfig(
            H=1000.0, nz=41,
            T_surface=T_MELT,
            Q_geo=Q_geo,
            drain_rate=0.0,
            n_newton=5,
        )

        E = initialize_from_surface(cfg)
        dt_yr = 5000.0
        n_steps = 200
        E, hist = time_step(E, cfg, dt_years=dt_yr, n_steps=n_steps,
                            n_iter=30, save_profiles=True)

        results[label] = (cfg, hist)

    return results


# ---- Case 3: warm surface + downward advection ----

def case_warm_surface_advection():
    """
    Surface at PMP with downward advection (accumulation-driven).
    The cold advection from the surface competes with the warm BC.
    """
    results = {}
    for Pe, label in [(0.0, 'Pe=0'),
                       (-2.0, 'Pe=-2'),
                       (-10.0, 'Pe=-10'),
                       (-50.0, 'Pe=-50')]:
        cfg = ColumnConfig(
            H=1000.0, nz=41,
            T_surface=T_MELT,
            Q_geo=0.042,
            n_newton=5,
        )

        if Pe != 0.0:
            w = Pe * K_COLD / (RHO_I * cfg.H**2)
            cfg.sigma_dot = np.full(cfg.nz, w)

        E = initialize_from_surface(cfg)
        dt_yr = 5000.0
        n_steps = 200
        E, hist = time_step(E, cfg, dt_years=dt_yr, n_steps=n_steps,
                            n_iter=30, save_profiles=True)

        results[label] = (cfg, hist)

    return results


# ---- Case 4: warm surface + drainage ----

def case_warm_surface_drainage():
    """
    Surface at PMP, high Q_geo, varying drainage rates.
    """
    results = {}
    for rd_per_yr, label in [(0.0, 'drain=0'),
                              (0.01, 'drain=0.01'),
                              (0.1, 'drain=0.1'),
                              (1.0, 'drain=1.0')]:
        cfg = ColumnConfig(
            H=1000.0, nz=41,
            T_surface=T_MELT,
            Q_geo=0.5,
            drain_rate=rd_per_yr / SECONDS_PER_YEAR,
            n_newton=5,
        )

        E = initialize_from_surface(cfg)
        dt_yr = 5000.0
        n_steps = 200
        E, hist = time_step(E, cfg, dt_years=dt_yr, n_steps=n_steps,
                            n_iter=30, save_profiles=True)

        results[label] = (cfg, hist)

    return results


if __name__ == '__main__':

    # ---- Case 1: time evolution ----
    print("=== Case 1: Warm surface evolution ===")
    cfg1, hist1 = case_warm_surface_evolution()
    sigma1 = cfg1.sigma
    times1_kyr = hist1['times'] / 1e3
    T_hist1, omega_hist1 = E_history_to_fields(hist1['E_history'], cfg1)

    plot_heatmaps(times1_kyr, sigma1, T_hist1, omega_hist1,
                  'Warm surface (Q_geo=42 mW/m²)', 'case1_evolution.png')

    # ---- Case 2: varying Q_geo ----
    print("\n=== Case 2: Varying Q_geo ===")
    qgeo_results = case_warm_surface_high_qgeo()
    for label, (cfg, hist) in qgeo_results.items():
        T_hist, omega_hist = E_history_to_fields(hist['E_history'], cfg)
        times_kyr = hist['times'] / 1e3
        plot_heatmaps(times_kyr, cfg.sigma, T_hist, omega_hist,
                      label, f'case2_{label}.png')

    # ---- Case 3: advection ----
    print("\n=== Case 3: Advection ===")
    adv_results = case_warm_surface_advection()
    for label, (cfg, hist) in adv_results.items():
        T_hist, omega_hist = E_history_to_fields(hist['E_history'], cfg)
        times_kyr = hist['times'] / 1e3
        plot_heatmaps(times_kyr, cfg.sigma, T_hist, omega_hist,
                      label, f'case3_{label}.png')

    # ---- Case 4: drainage ----
    print("\n=== Case 4: Drainage ===")
    drain_results = case_warm_surface_drainage()
    for label, (cfg, hist) in drain_results.items():
        T_hist, omega_hist = E_history_to_fields(hist['E_history'], cfg)
        times_kyr = hist['times'] / 1e3
        plot_heatmaps(times_kyr, cfg.sigma, T_hist, omega_hist,
                      f'Q_geo=500 mW/m², {label} a⁻¹',
                      f'case4_{label}.png')
