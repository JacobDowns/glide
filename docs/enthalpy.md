# Enthalpy Model for GLIDE

This document describes the enthalpy formulation, its transformation to terrain-following coordinates, the numerical discretization, and the implementation within GLIDE's multigrid framework.

## 1. The Enthalpy Equation

Following Aschwanden et al. (2012), ice enthalpy $E$ is defined as a function of temperature $T$, water content $\omega$, and pressure $p$:

$$
E(T, \omega, p) = \begin{cases}
c_i(T - T_{\text{ref}}) & \text{if } T < T_{\text{pmp}} \quad (\text{cold ice}) \\
E_{\text{pmp}} + L\omega & \text{if } T = T_{\text{pmp}}, \; \omega \geq 0 \quad (\text{temperate ice})
\end{cases}
$$

where:
- $c_i = 2009 \; \text{J/(kg·K)}$ is the heat capacity of ice
- $T_{\text{ref}} = 223.15 \; \text{K}$ is a reference temperature
- $L = 3.34 \times 10^5 \; \text{J/kg}$ is the latent heat of fusion

The pressure melting point and its associated enthalpy are:

$$
T_{\text{pmp}} = T_0 - \beta \rho_i g d, \qquad E_{\text{pmp}} = c_i(T_{\text{pmp}} - T_{\text{ref}})
$$

where $T_0 = 273.15 \; \text{K}$, $\beta = 7.9 \times 10^{-8} \; \text{K/Pa}$, and $d$ is depth below the ice surface.

### Governing equation

The enthalpy evolution follows the advection-diffusion equation (neglecting horizontal diffusion):

$$
\rho_i \left(\frac{\partial E}{\partial t} + u_x \frac{\partial E}{\partial x} + u_y \frac{\partial E}{\partial y} + u_z \frac{\partial E}{\partial z}\right) - \frac{\partial}{\partial z}\left(K_{c,t} \frac{\partial E}{\partial z}\right) = \phi - \rho_w L \, D_w(\omega)
$$

where:
- $K_{c,t} = k_i/c_i$ for cold ice ($E < E_{\text{pmp}}$), $K_{c,t} = \epsilon \, k_i/c_i$ for temperate ice ($\epsilon \approx 10^{-5}$)
- $\phi = A^{-1/n} |D(\mathbf{u})|^{1+1/n}$ is strain heating
- $D_w(\omega)$ is a drainage function that removes water from temperate ice

### Boundary conditions

**Surface** ($z = s$): Dirichlet condition from the mean annual surface temperature:

$$E|_{\text{surface}} = E_s = c_i(\min(T_s, T_0) - T_{\text{ref}})$$

**Bed** ($z = b$): Four cases depending on thermal state:

| Condition | BC Type | Equation |
|---|---|---|
| Cold base, dry ($E_b < E_{\text{pmp}}$, $W_{\text{till}} = 0$) | Neumann | $K_c \frac{\partial E}{\partial z}\big|_b = Q_{\text{geo}} + Q_{\text{fh}}$ |
| Cold base, wet ($E_b < E_{\text{pmp}}$, $W_{\text{till}} > 0$) | Dirichlet | $E_b = E_{\text{pmp}}$ |
| Temperate base, cold ice | Dirichlet | $E_b = E_{\text{pmp}}$ |
| Temperate base, temperate ice | Neumann | $K_t \frac{\partial E}{\partial z}\big|_b = 0$ |

where $Q_{\text{geo}}$ is geothermal heat flux and $Q_{\text{fh}} = \beta |\mathbf{u}_b|^{m+1}$ is frictional heating.

### Arrhenius factor coupling

The enthalpy feeds back into the velocity solver through the Glen-Paterson-Budd-Lliboutry-Duval law:

$$
A(T, \omega) = A_c(T)(1 + 181.25\omega)
$$

where $A_c(T)$ is the Paterson-Budd law with activation energies $Q = 60 \; \text{kJ/mol}$ ($T < 263.15 \; \text{K}$) or $Q = 139 \; \text{kJ/mol}$ ($T \geq 263.15 \; \text{K}$).

---

## 2. Terrain-Following (Sigma) Coordinate Transformation

### Coordinate definition

Define the sigma coordinate:

$$
\sigma = \frac{z - b(x,y)}{h(x,y,t)}, \quad \sigma \in [0, 1]
$$

where $\sigma = 0$ is the bed and $\sigma = 1$ is the ice surface.

### Transformation rules

The vertical derivative transforms as:

$$
\frac{\partial}{\partial z} = \frac{1}{h} \frac{\partial}{\partial \sigma}
$$

Horizontal and time derivatives at constant $z$ relate to those at constant $\sigma$:

$$
\frac{\partial E}{\partial x}\bigg|_z = \frac{\partial E}{\partial x}\bigg|_\sigma - \frac{1}{h}\left(\frac{\partial b}{\partial x} + \sigma \frac{\partial h}{\partial x}\right)\frac{\partial E}{\partial \sigma}
$$

$$
\frac{\partial E}{\partial y}\bigg|_z = \frac{\partial E}{\partial y}\bigg|_\sigma - \frac{1}{h}\left(\frac{\partial b}{\partial y} + \sigma \frac{\partial h}{\partial y}\right)\frac{\partial E}{\partial \sigma}
$$

$$
\frac{\partial E}{\partial t}\bigg|_z = \frac{\partial E}{\partial t}\bigg|_\sigma - \frac{\sigma}{h}\frac{\partial h}{\partial t}\frac{\partial E}{\partial \sigma}
$$

### The sigma velocity

The extra terms define the sigma pseudo-velocity $\dot{\sigma}$:

$$
h\dot{\sigma} = u_z - u_x\left(\frac{\partial b}{\partial x} + \sigma\frac{\partial h}{\partial x}\right) - u_y\left(\frac{\partial b}{\partial y} + \sigma\frac{\partial h}{\partial y}\right) - \sigma\frac{\partial h}{\partial t}
$$

For SSA (depth-averaged, no vertical shear), this simplifies to:

$$
\dot{\sigma} = \frac{m_b - \sigma \cdot \text{SMB}}{h}
$$

which is linear in $\sigma$. For general velocity profiles, $\dot{\sigma}$ is computed from the 3D velocity field.

### Transformed enthalpy equation

$$
\boxed{
\rho_i\left(\frac{\partial E}{\partial t}\bigg|_\sigma + u_x\frac{\partial E}{\partial x}\bigg|_\sigma + u_y\frac{\partial E}{\partial y}\bigg|_\sigma + \dot{\sigma}\frac{\partial E}{\partial \sigma}\right) - \frac{1}{h^2}\frac{\partial}{\partial \sigma}\left(K_{c,t}\frac{\partial E}{\partial \sigma}\right) = \phi - \rho_w L\,D_w(\omega)
}
$$

All spatial derivatives are now at constant $\sigma$. The key changes from the Cartesian form:
- $\dot{\sigma}\,\partial E/\partial\sigma$ replaces $u_z\,\partial E/\partial z$
- The diffusion term acquires a factor $1/h^2$, meaning diffusion is stronger in thin ice

### Transformed boundary conditions

**Surface** ($\sigma = 1$): $E|_{\sigma=1} = E_s$ (Dirichlet)

**Bed** ($\sigma = 0$, cold/dry Neumann):

$$
-\frac{K_c}{h}\frac{\partial E}{\partial \sigma}\bigg|_{\sigma=0} = Q_{\text{geo}} + Q_{\text{fh}}
$$

Note the $1/h$ factor from the coordinate transformation.

---

## 3. Numerical Approach

### Multigrid with column-wise Newton smoother

The enthalpy equation is solved implicitly using GLIDE's existing multigrid framework. The column-wise Newton solve acts as a smoother, analogous to the Vanka smoother for the SSA momentum balance:

| | SSA (Vanka) | Enthalpy (Column) |
|---|---|---|
| **Local unknowns** | $(u, v, H)$ on a 3×3 cell patch | $E_k$ for $k = 0, \ldots, N_z{-}1$ |
| **Frozen neighbors** | State on surrounding cells | $E$ in adjacent columns |
| **Local solve** | 5×5 dense system (LU) | $N_z \times N_z$ tridiagonal (Thomas) |
| **Nonlinearity** | Viscosity depends on strain rate | $K_{c,t}$ switches cold/temperate |
| **Parallelism** | All patches independent | All columns independent |

The column-wise solve is effective because the dominant coupling is vertical (diffusion with $1/h^2$ coefficient). Horizontal coupling through advection is weaker and handled by the multigrid coarse grid corrections.

### Vertical sigma grid

Non-uniform sigma levels bunched toward the bed:

$$
\sigma_k = \left(\frac{k}{N_z - 1}\right)^q, \quad k = 0, \ldots, N_z{-}1
$$

with $q > 1$ (default $q = 2$). This gives finer resolution near $\sigma = 0$ where the steepest enthalpy gradients occur.

### Horizontal discretization: finite volume with upwind fluxes

Enthalpy $E$ lives at cell centers, same as $H$. Horizontal advection uses the existing MAC-grid facet velocities at each sigma layer:

$$
F^x_{i,j+\frac{1}{2},k} = \begin{cases}
u_{i,j+\frac{1}{2},k} \cdot E_{i,j,k} & \text{if } u_{i,j+\frac{1}{2},k} > 0 \\
u_{i,j+\frac{1}{2},k} \cdot E_{i,j+1,k} & \text{if } u_{i,j+\frac{1}{2},k} < 0
\end{cases}
$$

The advective contribution to the residual:

$$
[\nabla_H \cdot (\mathbf{u}E)]_{i,j,k} = \frac{F^x_{i,j+\frac{1}{2},k} - F^x_{i,j-\frac{1}{2},k}}{\Delta x} + \frac{F^y_{i+\frac{1}{2},j,k} - F^y_{i-\frac{1}{2},j,k}}{\Delta x}
$$

### Vertical discretization: finite differences

**Vertical diffusion** (centered, non-uniform spacing):

$$
\frac{1}{h^2}\frac{\partial}{\partial\sigma}\!\left(K\frac{\partial E}{\partial\sigma}\right)_k \approx \frac{1}{h^2} \cdot \frac{2}{\Delta\sigma_k^- + \Delta\sigma_k^+}\left[\frac{K_{k+\frac{1}{2}}(E_{k+1} - E_k)}{\Delta\sigma_k^+} - \frac{K_{k-\frac{1}{2}}(E_k - E_{k-1})}{\Delta\sigma_k^-}\right]
$$

where $\Delta\sigma_k^+ = \sigma_{k+1} - \sigma_k$, $\Delta\sigma_k^- = \sigma_k - \sigma_{k-1}$, and $K_{k+\frac{1}{2}}$ is evaluated at the midpoint between nodes $k$ and $k+1$.

**Vertical advection** (upwind):

$$
\dot{\sigma}_k \frac{\partial E}{\partial\sigma}\bigg|_k \approx \dot{\sigma}_k^+ \frac{E_k - E_{k-1}}{\Delta\sigma_k^-} + \dot{\sigma}_k^- \frac{E_{k+1} - E_k}{\Delta\sigma_k^+}
$$

where $\dot{\sigma}^+ = \max(\dot{\sigma}, 0)$ and $\dot{\sigma}^- = \min(\dot{\sigma}, 0)$.

### Tridiagonal system assembly

For a single column $(i,j)$, the linearized system is $a_k E_{k-1} + b_k E_k + c_k E_{k+1} = d_k$:

$$
a_k = -\frac{2K_{k-\frac{1}{2}}}{h^2\,\Delta\sigma_k^-(\Delta\sigma_k^- + \Delta\sigma_k^+)} - \frac{\rho_i\,\dot{\sigma}_k^+}{\Delta\sigma_k^-}
$$

$$
c_k = -\frac{2K_{k+\frac{1}{2}}}{h^2\,\Delta\sigma_k^+(\Delta\sigma_k^- + \Delta\sigma_k^+)} + \frac{\rho_i\,\dot{\sigma}_k^-}{\Delta\sigma_k^+}
$$

$$
b_k = \frac{\rho_i}{\Delta t} - a_k - c_k
$$

$$
d_k = \frac{\rho_i\,E_k^n}{\Delta t} - \rho_i [\nabla_H \cdot (\mathbf{u}E)]_{i,j,k} + \phi_k - \rho_w L\,D_w(\omega_k)
$$

Horizontal advection enters only in $d_k$ (frozen neighbors), preserving the tridiagonal structure.

### Boundary rows

**Surface** ($k = N_z{-}1$): $b_{N_z-1} = 1$, $a_{N_z-1} = c_{N_z-1} = 0$, $d_{N_z-1} = E_s$

**Bed** ($k = 0$, cold/dry Neumann):

$$
b_0 = \frac{\rho_i}{\Delta t} + \frac{K_c}{h\,\Delta\sigma_0^+}, \quad c_0 = -\frac{K_c}{h\,\Delta\sigma_0^+}, \quad d_0 = \frac{\rho_i E_0^n}{\Delta t} - \rho_i[\nabla_H\cdot(\mathbf{u}E)]_0 + \phi_0 + \frac{Q_{\text{geo}} + Q_{\text{fh}}}{h}
$$

**Bed** ($k = 0$, temperate/Dirichlet): $b_0 = 1$, $c_0 = 0$, $d_0 = E_{\text{pmp}}$

---

## 4. Implementation Details

### Data layout

| Field | Grid location | Shape | Notes |
|---|---|---|---|
| `E` | cell center | `(ny, nx, nz)` | Contiguous columns for Thomas |
| `E_prev` | cell center | `(ny, nx, nz)` | Previous time step |
| `u3d` | vertical facet | `(nz, ny, nx+1)` | Layer-wise x-velocity |
| `v3d` | horizontal facet | `(nz, ny+1, nx)` | Layer-wise y-velocity |
| `sigma_dot` | cell center | `(ny, nx, nz)` | Sigma pseudo-velocity |
| `sigma` | — | `(nz,)` | Sigma node positions |
| `phi_strain` | cell center | `(ny, nx, nz)` | Strain heating |
| `E_surface` | cell center | `(ny, nx)` | Surface Dirichlet BC |
| `Q_geo` | cell center | `(ny, nx)` | Geothermal heat flux |
| `Q_fh` | cell center | `(ny, nx)` | Frictional heating |

`E` uses `(ny, nx, nz)` so that a column `E[i, j, :]` is contiguous in memory, optimizing the Thomas algorithm. The 3D velocities use `(nz, ny, ...)` so that horizontal slices are contiguous for the flux computation.

### CUDA kernel structure

**`enthalpy.cu`** contains five kernels:
- `enthalpy_compute_residual`: Full residual at all $(i,j,k)$ — one thread per column
- `enthalpy_column_smooth`: Newton/Thomas smoother — one thread per column
- `compute_sigma_dot`: Sigma velocity from 3D velocity field
- `restrict_enthalpy`: Full-weighting restriction (horizontal only, per sigma layer)
- `prolongate_enthalpy`: Bilinear prolongation (horizontal only, per sigma layer)

Each column-solve thread executes the Thomas algorithm sequentially in $O(N_z)$ — about 60 FLOPs for $N_z = 20$ — while all $N_y \times N_x$ columns run in parallel.

### Python interface

**`enthalpy.py`** provides:
- `EnthalpyState`, `EnthalpyForcing`, `EnthalpyVelocity`: Dataclasses for state/forcing
- `EnthalpyOperators`: Main class paralleling `ForwardOperators` for the SSA
- `ColumnSmootherConfig`: Configuration dataclass (analogous to `VankaConfig`)

### Operator splitting with velocity solver

Within each time step:
1. Solve SSA (momentum + mass conservation) via existing multigrid
2. Broadcast SSA velocities to all sigma layers (or use full 3D velocity)
3. Compute $\dot{\sigma}$ from the 3D velocity field
4. Compute frictional heating $Q_{\text{fh}}$
5. Solve enthalpy via multigrid with column smoother
6. Update Arrhenius factor $B = A^{-1/n}$ from new enthalpy
7. (Optionally) update basal sliding conditions via till water model

### Multigrid integration

- Restriction and prolongation operate **horizontally only** — $N_z$ is fixed at all levels
- For each sigma layer, the standard `restrict_cell_avg` / `prolongate_cell_bilinear` patterns apply, adapted for the `(ny, nx, nz)` memory layout
- The FAS V-cycle structure mirrors the existing momentum solver

---

## 5. Physical Constants

| Symbol | Value | Units | Description |
|---|---|---|---|
| $\rho_i$ | 910 | kg/m³ | Ice density |
| $\rho_w$ | 1000 | kg/m³ | Water density |
| $c_i$ | 2009 | J/(kg·K) | Heat capacity of ice |
| $k_i$ | 2.1 | W/(m·K) | Thermal conductivity of ice |
| $L$ | $3.34 \times 10^5$ | J/kg | Latent heat of fusion |
| $T_0$ | 273.15 | K | Melting point at standard pressure |
| $T_{\text{ref}}$ | 223.15 | K | Reference temperature |
| $\beta$ | $7.9 \times 10^{-8}$ | K/Pa | Clausius-Clapeyron constant |
| $g$ | 9.81 | m/s² | Gravitational acceleration |
| $\epsilon$ | $10^{-5}$ | — | Temperate conductivity reduction |

---

## 6. References

- Aschwanden, A., Bueler, E., Khroulev, C., and Blatter, H. (2012). An enthalpy formulation for glaciers and ice sheets. *Journal of Glaciology*, 58(209):441–457.
- Kleiner, T., Rückamp, M., Bondzio, J.H., and Humbert, A. (2015). Enthalpy benchmark experiments for numerical ice sheet models. *The Cryosphere*, 9(1):217–228.
- Jouvet, G. et al. (2024). Concepts and capabilities of the Instructed Glacier Model (IGM v2.2.1). Preprint, EarthArXiv.
