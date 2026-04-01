"""
Demo: column smoother convergence on a few canonical problems.

Run from repo root:
    python experiments/column_smoother/demo.py
"""
import numpy as np
import matplotlib.pyplot as plt
from column import (
    ColumnConfig, column_sweep, compute_residual,
    initialize_from_surface, initialize_linear_profile,
    initialize_uniform_temperature,
    enthalpy_from_temperature, temperature_from_enthalpy,
    water_content_from_enthalpy, get_E_pmp,
    RHO_I, K_I, C_I, K_COLD, T_MELT, BETA_CC, GRAVITY, T_REF, L_HEAT,
)


def plot_convergence(ax, history, label):
    norms = history['residual_norms']
    ax.semilogy(range(len(norms)), norms, 'o-', label=label, markersize=3)


def case_cold_column():
    """
    Pure diffusion, cold base. Should converge in ~1 sweep since
    the linear profile is exact on a uniform grid with constant K.
    """
    cfg = ColumnConfig(
        H=1000.0, nz=21,
        T_surface=280.15,  
        Q_geo=0.0,
        dt=1e11,           # pseudo-steady-state
        n_newton=3,
    )

    E = initialize_from_surface(cfg)
    E_prev = E.copy()

    E_sol, history = column_sweep(E, E_prev, cfg, n_iter=20, report=True)

    # Analytical: T(sigma) = T_s + Q_geo*H/k_i * (1 - sigma)
    sigma = cfg.sigma
    T_exact = cfg.T_surface + (cfg.Q_geo * cfg.H / K_I) * (1.0 - sigma)
    T_sol = np.array([temperature_from_enthalpy(E_sol[k], sigma[k], cfg.H)
                      for k in range(cfg.nz)])

    return cfg, E_sol, history, T_exact, T_sol



if __name__ == '__main__':

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # --- Case 1: Cold column ---
    print("=== Cold column ===")
    cfg1, E1, hist1, T_exact1, T_sol1 = case_cold_column()
    sigma1 = cfg1.sigma

    axes[0, 0].plot(T_exact1, sigma1, 'k--', label='Analytical')
    axes[0, 0].plot(T_sol1, sigma1, 'r-', label='Column solver')
    axes[0, 0].set_xlabel('Temperature (K)')
    axes[0, 0].set_ylabel(r'$\sigma$')
    axes[0, 0].set_title('Cold column: profile')
    axes[0, 0].legend()

    plot_convergence(axes[1, 0], hist1, 'Cold column')
    axes[1, 0].set_xlabel('Sweep')
    axes[1, 0].set_ylabel('Residual norm')
    axes[1, 0].set_title('Cold column: convergence')