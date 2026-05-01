"""
Generate the column discretization diagram for the discretization note.

Run:
    cd notes/figures && python column_grid.py
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(14, 8),
                         gridspec_kw={'width_ratios': [1, 1.3]})

# ================================================================
# LEFT PANEL: Vertical sigma grid (single column)
# ================================================================
ax = axes[0]
ax.set_xlim(-0.5, 4.5)
ax.set_ylim(-0.25, 1.25)
ax.set_aspect('equal')
ax.set_title('Vertical sigma grid (single column)', fontsize=12, fontweight='bold')

nz = 6
sigma = np.linspace(0, 1, nz)
dsig = sigma[1] - sigma[0]

# Control volume shading
for k in range(nz):
    if k == 0:
        # Half-cell at bed
        y_lo = 0.0
        y_hi = dsig / 2
        color = '#FFE0B2'
        label = 'Bed half-cell CV'
    elif k == nz - 1:
        # Surface: Dirichlet, no CV
        continue
    else:
        y_lo = sigma[k] - dsig / 2
        y_hi = sigma[k] + dsig / 2
        color = '#E3F2FD'
        label = 'Interior CV' if k == 1 else None
    rect = mpatches.FancyBboxPatch((0.3, y_lo), 3.4, y_hi - y_lo,
                                    boxstyle="round,pad=0.02",
                                    facecolor=color, edgecolor='gray',
                                    linewidth=0.5, alpha=0.6)
    ax.add_patch(rect)

# Sigma level lines
for k in range(nz):
    style = '-' if k in (0, nz - 1) else '--'
    alpha = 1.0 if k in (0, nz - 1) else 0.3
    ax.axhline(sigma[k], color='gray', linestyle=style, alpha=alpha, linewidth=0.5)

# Half-node lines (where K is evaluated)
for k in range(nz - 1):
    y_half = 0.5 * (sigma[k] + sigma[k + 1])
    ax.axhline(y_half, color='green', linestyle=':', alpha=0.4, linewidth=0.8)

# E nodes (cell centers)
x_E = 1.0
for k in range(nz):
    marker = 's' if k == nz - 1 else 'o'
    color = '#D32F2F' if k == nz - 1 else '#1565C0'
    ax.plot(x_E, sigma[k], marker, color=color, markersize=10, zorder=5)
    if k == 0:
        ax.annotate(f'$E_0$ (bed)', (x_E + 0.15, sigma[k]),
                    fontsize=9, va='center')
    elif k == nz - 1:
        ax.annotate(f'$E_s$ (Dirichlet)', (x_E + 0.15, sigma[k]),
                    fontsize=9, va='center', color='#D32F2F')
    elif k == 1:
        ax.annotate(f'$E_{k}$', (x_E + 0.15, sigma[k]),
                    fontsize=9, va='center')
    elif k == nz - 2:
        ax.annotate(f'$E_{{N_z-2}}$', (x_E + 0.15, sigma[k]),
                    fontsize=9, va='center')

# sigma_dot nodes
x_sd = 2.2
for k in range(nz - 1):
    ax.plot(x_sd, sigma[k], '^', color='#6A1B9A', markersize=7, zorder=5)
    if k == 0:
        ax.annotate(r'$\dot\sigma_0$', (x_sd + 0.12, sigma[k]),
                    fontsize=8, va='center', color='#6A1B9A')

# K at half-nodes
x_K = 3.2
for k in range(nz - 1):
    y_half = 0.5 * (sigma[k] + sigma[k + 1])
    ax.plot(x_K, y_half, 'D', color='#2E7D32', markersize=6, zorder=5)
    if k == 0:
        ax.annotate(r'$K_{1/2}$', (x_K + 0.12, y_half),
                    fontsize=8, va='center', color='#2E7D32')
    elif k == nz - 2:
        ax.annotate(r'$K_{N_z-3/2}$', (x_K + 0.12, y_half),
                    fontsize=8, va='center', color='#2E7D32')

# Neumann flux arrow at bed
ax.annotate('', xy=(0.6, 0.03), xytext=(0.6, -0.15),
            arrowprops=dict(arrowstyle='->', color='#E65100', lw=2))
ax.text(0.6, -0.18, r'$Q_{\mathrm{geo}} + Q_{\mathrm{fh}}$',
        ha='center', fontsize=9, color='#E65100')

# Half-cell bracket
bx = 0.15
ax.annotate('', xy=(bx, 0), xytext=(bx, dsig / 2),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1))
ax.text(bx - 0.1, dsig / 4, r'$\frac{\Delta\sigma}{2}$',
        ha='right', va='center', fontsize=9)

# Full cell bracket
ax.annotate('', xy=(bx, sigma[1] - dsig / 2), xytext=(bx, sigma[1] + dsig / 2),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1))
ax.text(bx - 0.1, sigma[1], r'$\Delta\sigma$',
        ha='right', va='center', fontsize=9)

# Sigma labels on right
for k in range(nz):
    ax.text(4.3, sigma[k], f'$\\sigma={k}/(N_z-1)$' if k in (0, nz-1)
            else f'$k={k}$', fontsize=7, va='center', ha='left', color='gray')

ax.set_ylabel(r'$\sigma$ (bed=0, surface=1)', fontsize=10)
ax.set_xticks([])

# Legend
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1565C0',
               markersize=8, label='$E_k$ (enthalpy node)'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#D32F2F',
               markersize=8, label='$E_s$ (surface Dirichlet)'),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#6A1B9A',
               markersize=7, label=r'$\dot\sigma_k$ (sigma velocity)'),
    plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#2E7D32',
               markersize=6, label='$K_{k+1/2}$ (half-node diffusivity)'),
    mpatches.Patch(facecolor='#E3F2FD', edgecolor='gray', alpha=0.6,
                   label='Interior control volume'),
    mpatches.Patch(facecolor='#FFE0B2', edgecolor='gray', alpha=0.6,
                   label='Bed half-cell control volume'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=7,
          framealpha=0.9)


# ================================================================
# RIGHT PANEL: Horizontal MAC grid (one sigma layer)
# ================================================================
ax2 = axes[1]
ax2.set_xlim(-0.8, 3.8)
ax2.set_ylim(-0.8, 3.8)
ax2.set_aspect('equal')
ax2.set_title('Horizontal MAC grid (one $\\sigma$ layer)', fontsize=12,
              fontweight='bold')

# Draw cell boundaries
for i in range(4):
    ax2.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.5)
    ax2.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.5)

# Highlight center cell
rect = mpatches.FancyBboxPatch((0.5, 0.5), 1.0, 1.0,
                                boxstyle="round,pad=0.02",
                                facecolor='#E3F2FD', edgecolor='#1565C0',
                                linewidth=1.5, alpha=0.5)
ax2.add_patch(rect)

# Cell-center E values
for i in range(3):
    for j in range(3):
        if i == 1 and j == 1:
            ax2.plot(j, i, 'o', color='#1565C0', markersize=12, zorder=5)
            ax2.text(j, i + 0.2, '$E_{i,j,k}$', ha='center', va='bottom',
                     fontsize=10, fontweight='bold', color='#1565C0')
        else:
            ax2.plot(j, i, 'o', color='#90CAF9', markersize=8, zorder=5)

# Neighbor labels
ax2.text(0, 1.2, '$E_{i,j-1}$', ha='center', va='bottom', fontsize=8, color='gray')
ax2.text(2, 1.2, '$E_{i,j+1}$', ha='center', va='bottom', fontsize=8, color='gray')
ax2.text(1, 2.2, '$E_{i+1,j}$', ha='center', va='bottom', fontsize=8, color='gray')
ax2.text(1, 0.2, '$E_{i-1,j}$', ha='center', va='bottom', fontsize=8, color='gray')

# u-velocities on vertical facets
for i in range(3):
    for j in range(4):
        x = j - 0.5
        y = i
        if 0 <= x <= 2.5 and 0 <= y <= 2:
            color = '#E65100' if (j in (1, 2) and i == 1) else '#FFCC80'
            size = 9 if (j in (1, 2) and i == 1) else 5
            ax2.plot(x, y, '>', color=color, markersize=size, zorder=4)

# Label the highlighted u facets
ax2.text(0.5, 1 - 0.25, '$u_{j-1/2}$', ha='center', fontsize=8, color='#E65100')
ax2.text(1.5, 1 - 0.25, '$u_{j+1/2}$', ha='center', fontsize=8, color='#E65100')

# v-velocities on horizontal facets
for i in range(4):
    for j in range(3):
        x = j
        y = i - 0.5
        if 0 <= x <= 2 and 0 <= y <= 2.5:
            color = '#6A1B9A' if (i in (1, 2) and j == 1) else '#CE93D8'
            size = 9 if (i in (1, 2) and j == 1) else 5
            ax2.plot(x, y, '^', color=color, markersize=size, zorder=4)

# Label the highlighted v facets
ax2.text(1 + 0.2, 0.5, '$v_{i-1/2}$', ha='left', fontsize=8, color='#6A1B9A')
ax2.text(1 + 0.2, 1.5, '$v_{i+1/2}$', ha='left', fontsize=8, color='#6A1B9A')

# Flux arrows for the center cell
arrow_kw = dict(arrowstyle='->', color='#D32F2F', lw=1.5)
# Right face (outflow example)
ax2.annotate('', xy=(1.7, 1), xytext=(1.3, 1), arrowprops=arrow_kw)
ax2.text(1.5, 1.35, '$F^x_{j+1/2}$', ha='center', fontsize=8, color='#D32F2F')
# Left face
ax2.annotate('', xy=(0.3, 1), xytext=(0.7, 1),
             arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=1.5, alpha=0.5))

# Axis labels
ax2.set_xlabel('$j$ (x-direction)', fontsize=10)
ax2.set_ylabel('$i$ (y-direction)', fontsize=10)
ax2.set_xticks([0, 1, 2])
ax2.set_xticklabels(['$j{-}1$', '$j$', '$j{+}1$'])
ax2.set_yticks([0, 1, 2])
ax2.set_yticklabels(['$i{-}1$', '$i$', '$i{+}1$'])

# Legend
legend2 = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1565C0',
               markersize=9, label='$E_{i,j,k}$ (cell center)'),
    plt.Line2D([0], [0], marker='>', color='w', markerfacecolor='#E65100',
               markersize=8, label='$u$ (x-facet velocity)'),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#6A1B9A',
               markersize=8, label='$v$ (y-facet velocity)'),
    plt.Line2D([0], [0], marker='', color='#D32F2F', lw=1.5,
               label='Upwind enthalpy flux'),
    mpatches.Patch(facecolor='#E3F2FD', edgecolor='#1565C0',
                   alpha=0.5, label='Cell $(i,j)$ control volume'),
]
ax2.legend(handles=legend2, loc='upper left', fontsize=7, framealpha=0.9)

plt.tight_layout()
plt.savefig('column_grid.png', dpi=200, bbox_inches='tight')
print("Saved column_grid.png")
plt.savefig('column_grid.svg', bbox_inches='tight')
print("Saved column_grid.svg")
