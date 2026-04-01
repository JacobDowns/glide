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

where: - $c_i = 2009 \; \text{J/(kg·K)}$ is the heat capacity of ice - $T_{\text{ref}} = 223.15 \; \text{K}$ is a reference temperature - $L = 3.34 \times 10^5 \; \text{J/kg}$ is the latent heat of fusion

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

where: - $K_{c,t} = k_i/c_i$ for cold ice ($E < E_{\text{pmp}}$), $K_{c,t} = \epsilon \, k_i/c_i$ for temperate ice ($\epsilon \approx 10^{-5}$) - $\phi = A^{-1/n} |D(\mathbf{u})|^{1+1/n}$ is strain heating - $D_w(\omega)$ is a drainage function that removes water from temperate ice

### Drainage function

The drainage term $D_w(\omega)$ removes liquid water from temperate ice. The current implementation uses a simple linear model:

$$D_w(\omega) = r_d \cdot \omega, \qquad \omega = \frac{E - E_{\text{pmp}}}{L} \;\text{for}\; E > E_{\text{pmp}}, \quad 0 \;\text{otherwise}$$

where $r_d$ is a constant drainage rate (units: s$^{-1}$) passed as the `drain_rate` kernel parameter. The full source contribution to the residual is $\rho_w L \, D_w(\omega)$, and its Jacobian $\partial/\partial E(\rho_w L \, r_d \, \omega) = \rho_w \, r_d$ is nonzero only in temperate ice.

This is a placeholder model. Physically motivated alternatives include: - Threshold drainage: $D_w = r_d \cdot \max(\omega - \omega_{\max}, 0)$, draining only excess water above a maximum porosity - Coupling to a basal hydrology model where drained water feeds till water storage $W_{\text{till}}$, which in turn affects basal sliding and the bed boundary condition switch

### Boundary conditions

**Surface** ($z = s$): Dirichlet condition from the mean annual surface temperature:

$$E|_{\text{surface}} = E_s = c_i(\min(T_s, T_0) - T_{\text{ref}})$$

**Bed** ($z = b$): Four cases depending on thermal state (Aschwanden et al., 2012):

| Condition | BC Type | Equation |
|------------------------|------------------------|------------------------|
| Cold base, dry ($E_b < E_{\text{pmp}}$, $W_{\text{till}} = 0$) | Neumann | $K_c \frac{\partial E}{\partial z}\big|_b = Q_{\text{geo}} + Q_{\text{fh}}$ |
| Cold base, wet ($E_b < E_{\text{pmp}}$, $W_{\text{till}} > 0$) | Dirichlet | $E_b = E_{\text{pmp}}$ |
| Temperate base, cold ice | Dirichlet | $E_b = E_{\text{pmp}}$ |
| Temperate base, temperate ice | Neumann | $K_t \frac{\partial E}{\partial z}\big|_b = 0$ |

where $Q_{\text{geo}}$ is geothermal heat flux and $Q_{\text{fh}} = \beta |\mathbf{u}_b|^{m+1}$ is frictional heating.

**Current implementation** (`enthalpy.cu`): The basal geothermal and frictional heat fluxes are always applied through the Neumann term

$$
-\frac{K}{h}\frac{\partial E}{\partial \sigma}\bigg|_{\sigma=0} = Q_{\text{geo}} + Q_{\text{fh}},
$$

with $K$ evaluated from the local enthalpy state. When $E_b > E_{\text{pmp}}$, the diagnosed temperature is still clipped to $T_{\text{pmp}}$ and the excess energy is stored as latent enthalpy (and may be removed by the drainage term if enabled).

This preserves the imposed basal energy flux after the bed becomes temperate, but it is still not a full polythermal basal model: the cold-wet and temperate-temperate cases require till water content ($W_{\text{till}}$), basal melt / freeze-on, or hydrology coupling, none of which are yet evolved explicitly.

### Arrhenius factor coupling

The enthalpy feeds back into the velocity solver through the Glen-Paterson-Budd-Lliboutry-Duval law:

$$
A(T, \omega) = A_c(T)(1 + 181.25\omega)
$$

where $A_c(T)$ is the Paterson-Budd law with activation energies $Q = 60 \; \text{kJ/mol}$ ($T < 263.15 \; \text{K}$) or $Q = 139 \; \text{kJ/mol}$ ($T \geq 263.15 \; \text{K}$).

------------------------------------------------------------------------

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

All spatial derivatives are now at constant $\sigma$. The key changes from the Cartesian form: - $\dot{\sigma}\,\partial E/\partial\sigma$ replaces $u_z\,\partial E/\partial z$ - The diffusion term acquires a factor $1/h^2$, meaning diffusion is stronger in thin ice

### Transformed boundary conditions

**Surface** ($\sigma = 1$): $E|_{\sigma=1} = E_s$ (Dirichlet)

**Bed** ($\sigma = 0$, cold/dry Neumann):

$$
-\frac{K_c}{h}\frac{\partial E}{\partial \sigma}\bigg|_{\sigma=0} = Q_{\text{geo}} + Q_{\text{fh}}
$$

Note the $1/h$ factor from the coordinate transformation.

------------------------------------------------------------------------

## 3. Numerical Approach

### Multigrid with column-wise Newton smoother

The enthalpy equation is solved implicitly using GLIDE's existing multigrid framework. The column-wise Newton solve acts as a smoother, analogous to the Vanka smoother for the SSA momentum balance:

|   | SSA (Vanka) | Enthalpy (Column) |
|------------------------|------------------------|------------------------|
| **Local unknowns** | $(u, v, H)$ on a 3×3 cell patch | $E_k$ for $k = 0, \ldots, N_z{-}1$ |
| **Frozen neighbors** | State on surrounding cells | $E$ in adjacent columns |
| **Local solve** | 5×5 dense system (LU) | $N_z \times N_z$ tridiagonal (Thomas) |
| **Nonlinearity** | Viscosity depends on strain rate | $K_{c,t}$ switches cold/temperate |
| **Parallelism** | All patches independent | All columns independent |

The column-wise solve is effective because the dominant coupling is vertical (diffusion with $1/h^2$ coefficient). Horizontal coupling through advection is weaker and handled by the multigrid coarse grid corrections.

### Vertical sigma grid

Uniform sigma levels from bed to surface:

$$
\sigma_k = \frac{k}{N_z - 1}, \quad k = 0, \ldots, N_z{-}1
$$

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

**Vertical diffusion** (centered, uniform spacing $\Delta\sigma = 1/(N_z - 1)$):

$$
\frac{1}{h^2}\frac{\partial}{\partial\sigma}\!\left(K\frac{\partial E}{\partial\sigma}\right)_k \approx \frac{1}{h^2 \Delta\sigma^2}\left[K_{k+\frac{1}{2}}(E_{k+1} - E_k) - K_{k-\frac{1}{2}}(E_k - E_{k-1})\right]
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

**Bed** ($k = 0$, Neumann):

The bed node sits at $\sigma = 0$; its control volume extends to the midpoint with the first interior node, giving a half-cell width $\Delta\sigma_{\text{half}} = \Delta\sigma_0^+/2$. The flux divergence is divided by this width to stay consistent with interior cells (which divide by $\Delta\sigma_{\text{avg}}$):

$$
b_0 = \frac{\rho_i}{\Delta t} + \frac{K_{1/2}}{h^2\,\Delta\sigma_0^+\,\Delta\sigma_{\text{half}}}, \quad c_0 = -\frac{K_{1/2}}{h^2\,\Delta\sigma_0^+\,\Delta\sigma_{\text{half}}}
$$

$$
d_0 = \frac{\rho_i E_0^n}{\Delta t} - \rho_i[\nabla_H\cdot(\mathbf{u}E)]_0 + \phi_0 + \frac{Q_{\text{geo}} + Q_{\text{fh}}}{h\,\Delta\sigma_{\text{half}}}
$$

------------------------------------------------------------------------

## 4. Implementation Details

### Data layout

| Field | Grid location | Shape | Notes |
|----|----|----|----|
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

**`enthalpy.cu`** contains five kernels: - `enthalpy_compute_residual`: Full residual at all $(i,j,k)$ — one thread per column - `enthalpy_column_smooth`: Newton/Thomas smoother — one thread per column - `compute_sigma_dot`: Sigma velocity from 3D velocity field - `restrict_enthalpy`: Full-weighting restriction (horizontal only, per sigma layer) - `prolongate_enthalpy`: Bilinear prolongation (horizontal only, per sigma layer)

Each column-solve thread executes the Thomas algorithm sequentially in $O(N_z)$ — about 60 FLOPs for $N_z = 20$ — while all $N_y \times N_x$ columns run in parallel.

### Jacobian architecture: consistency with the stress balance

The enthalpy solver follows the same Stencil/Jacobian decomposition used by the momentum balance (`flux.cu`, `stress.cu`, `viscosity.cu`). Each physical term is factored into:

1.  **Stencil struct** — the local inputs needed to evaluate the term
2.  **Jacobian struct** — holds the residual value and partial derivatives with respect to column unknowns, plus an `apply_jvp()` method for Jacobian-vector products
3.  **`get_*_jac()` function** — the single source of truth that computes both residual and derivatives from the stencil

Both `enthalpy_compute_residual` and `enthalpy_column_smooth` call the same `get_*_jac()` functions. The residual kernel reads only the `.res` field; the smoother reads both `.res` and `.d_E_*` fields to assemble the tridiagonal system. This guarantees that the Jacobian is always the exact linearization of the residual.

#### Enthalpy Jacobian structs

| Struct | Term | Derivatives | Momentum analogue |
|------------------|------------------|------------------|:-----------------:|
| `ColumnDiffusionJacobian` | $-(1/h^2)\,\partial_\sigma(K\,\partial_\sigma E)$ (interior) | `d_E_km1`, `d_E_k`, `d_E_kp1` | `SigmaNormalJacobian` |
| `BedDiffusionJacobian` | Same, Neumann BC at $k=0$ | `d_E_k`, `d_E_kp1` | — |
| `SigmaAdvectionJacobian` | $\rho_i\,\dot\sigma\,\partial_\sigma E$ (upwind, interior) | `d_E_km1`, `d_E_k`, `d_E_kp1` | `HorizontalFluxJacobian` |
| `BedSigmaAdvectionJacobian` | Same, one-sided at $k=0$ | `d_E_k`, `d_E_kp1` | — |
| `HorizEnthalpyFluxJacobian` | Upwind horizontal flux per facet | `d_E_here` | `HorizontalFluxJacobian` |
| `DrainageJacobian` | $\rho_w L\,D_w(\omega)$ | `d_E_k` | `TauBxJacobian` (sliding) |

#### Assembly pattern

The column smoother assembles the tridiagonal system by summing Jacobian struct fields, exactly as the Vanka smoother assembles its 5\$\times\$5 system:

```         
// Enthalpy column smoother (tridiagonal)
a[k] = diff_jac.d_E_km1 + adv_jac.d_E_km1;
c[k] = diff_jac.d_E_kp1 + adv_jac.d_E_kp1;
b[k] = RHO_I/dt + diff_jac.d_E_k + adv_jac.d_E_k
     + drain_jac.d_E_k + horiz_adv_diag[k];
rhs[k] = -r;

// Compare: Vanka smoother (5x5 dense)
J[20] -= j_l.d_u * dx_inv;     // off-diagonal
J[24] -= j_l.d_H_r * dx_inv;   // diagonal
r[4]  -= j_l.res * dx_inv;     // residual
```

#### Horizontal advection and the column diagonal

When the column cell $(i,j,k)$ is the upwind donor for an outflow face, the flux depends on $E_{i,j,k}$. The `HorizEnthalpyFluxJacobian.d_E_here` captures this derivative, which is accumulated into the diagonal $b_k$ of the tridiagonal system. In the momentum balance, the analogous terms appear as `d_H_l` / `d_H_r` in `HorizontalFluxJacobian`. Previously, the enthalpy smoother omitted this diagonal contribution; it is now included, improving convergence when horizontal advection is significant.

#### DualFloat support for automatic differentiation

Every Jacobian struct has a corresponding `*StencilDual` and `get_*_dual()` wrapper, following the same forward-mode AD pattern used by the momentum balance (`get_horizontal_flux_dual`, `get_sigma_xx_dual`, etc.).

**Pattern** (identical to `flux.cu` / `stress.cu`):

```         
struct ColumnDiffusionStencilDual {
    DualFloat E_km1, E_k, E_kp1;         // differentiated variables
    float E_pmp_km1, E_pmp_k, E_pmp_kp1; // frozen parameters
    float dsig, h2_inv;

    ColumnDiffusionStencil get_primals() const;  // extract .v fields
    ColumnDiffusionStencil get_diffs() const;    // extract .d fields
};

DualFloat get_column_diffusion_dual(ColumnDiffusionStencilDual s) {
    auto jac = get_column_diffusion_jac(s.get_primals());
    return {jac.res, jac.apply_jvp(s.get_diffs())};
}
```

`DualFloat` (defined in `common.cu`) carries a primal value `.v` and a perturbation `.d` through arithmetic via operator overloading. The `get_*_dual()` wrappers compose: to compute a full-column Jacobian-vector product, seed the `DualFloat` inputs with the perturbation vector and sum the `.d` outputs.

**Available dual functions:**

| Function                       | Term                             |
|--------------------------------|----------------------------------|
| `get_column_diffusion_dual`    | Interior vertical diffusion      |
| `get_bed_diffusion_dual`       | Bed Neumann BC diffusion         |
| `get_sigma_advection_dual`     | Interior upwind sigma advection  |
| `get_bed_sigma_advection_dual` | One-sided bed sigma advection    |
| `get_horiz_enthalpy_flux_dual` | Per-facet horizontal upwind flux |
| `get_drainage_dual`            | Water drainage source            |

**Use cases:** - Matrix-free Jacobian-vector products for Krylov-accelerated smoothing or defect correction - Sensitivity analysis (e.g., $\partial E / \partial Q_{\text{geo}}$ by seeding through parameters) - Verification: compare `get_*_dual()` output against finite differences to validate the hand-coded Jacobian entries

### Python interface

**`enthalpy.py`** provides: - `EnthalpyState`, `EnthalpyForcing`, `EnthalpyVelocity`: Dataclasses for state/forcing - `EnthalpyOperators`: Main class paralleling `ForwardOperators` for the SSA - `ColumnSmootherConfig`: Configuration dataclass (analogous to `VankaConfig`)

### Operator splitting with velocity solver

Within each time step: 1. Solve SSA (momentum + mass conservation) via existing multigrid 2. Broadcast SSA velocities to all sigma layers (or use full 3D velocity) 3. Compute $\dot{\sigma}$ from the 3D velocity field 4. Compute frictional heating $Q_{\text{fh}}$ 5. Solve enthalpy via multigrid with column smoother 6. Update Arrhenius factor $B = A^{-1/n}$ from new enthalpy 7. (Optionally) update basal sliding conditions via till water model

### Multigrid integration

-   Restriction and prolongation operate **horizontally only** — $N_z$ is fixed at all levels
-   For each sigma layer, the standard `restrict_cell_avg` / `prolongate_cell_bilinear` patterns apply, adapted for the `(ny, nx, nz)` memory layout
-   The FAS V-cycle structure mirrors the existing momentum solver

------------------------------------------------------------------------

## 5. Physical Constants

| Symbol | Value | Units | Description |
|----|----|----|----|
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

------------------------------------------------------------------------

## 6. Verification Tests

Tests live in `tests/` and can be run with `python tests/test_enthalpy_*.py`. Corresponding plotting examples are in `examples/thermal/`.

### Cold column (`test_enthalpy_cold_column.py`, `examples/thermal/cold_column.py`)

| Parameter        | Value                 |
|------------------|-----------------------|
| $H$              | 1000 m                |
| $T_s$            | 243.15 K ($-30°$C)    |
| $Q_{\text{geo}}$ | 0.04 W/m$^2$          |
| $N_z$            | 21 (uniform $\sigma$) |

A static ice slab, no velocity, geothermal flux at the base. The base stays cold ($T_{\text{bed}} < T_{\text{pmp}}$).

**Analytical steady state:** With constant diffusivity $K_c = k_i/c_i$ and no advection, the enthalpy equation reduces to $\partial_\sigma(K_c\,\partial_\sigma E) = 0$. The solution is a linear profile:

$$T(\sigma) = T_s + \frac{Q_{\text{geo}} H}{k_i}(1 - \sigma)$$

**What it tests:** - Neumann basal BC via `get_bed_diffusion_jac` (including half-cell control volume correction) - Interior vertical diffusion via `get_column_diffusion_jac` - Column smoother convergence (residual drops $\sim 10^4\times$ in one sweep) - Time-stepping to steady state (200 kyr, well past the $\sim 2.8$ kyr cold-zone diffusion timescale $\tau = H^2/(\kappa\pi^2)$ where $\kappa = k_i/(\rho_i c_i)$)

**Tolerance:** Relative error $< 0.5\%$. In practice achieves $\sim 10^{-7}$ because the linear profile is exact on any grid.

### Polythermal Stefan problem (`test_enthalpy_stefan.py`, `examples/thermal/stefan.py`)

| Parameter        | Value                                      |
|------------------|--------------------------------------------|
| $H$              | 1000 m                                     |
| $T_s$            | 223.15 K ($-50°$C)                         |
| $Q_{\text{geo}}$ | 0.5 W/m$^2$                                |
| $N_z$            | 41 (test) / 64 (example), uniform $\sigma$ |
| Drainage         | $r_d = 0$ (disabled)                       |

Same static slab geometry, but with $Q_{\text{geo}} \gg Q_{\text{crit}} = k_i(T_{\text{pmp,bed}} - T_s)/H \approx 0.104$ W/m$^2$. The excess geothermal flux drives a temperate basal layer where enthalpy exceeds the pressure melting point.

**Semi-analytical steady state:** The column splits into two zones separated by a cold-temperate surface (CTS) at $\sigma^*$:

-   **Cold zone** ($\sigma^* < \sigma < 1$): Heat flux $Q_{\text{geo}}$ is carried upward by cold diffusivity $K_c = k_i/c_i$. The enthalpy profile is linear:

$$E(\sigma) = c_i(T_s - T_{\text{ref}}) + \frac{Q_{\text{geo}} H}{K_c}(1 - \sigma)$$

-   **Temperate zone** ($0 \leq \sigma < \sigma^*$): Temperature is locked to $T_{\text{pmp}}(\sigma)$; excess energy is stored as water content. The enthalpy profile is linear with the much steeper temperate slope:

$$E(\sigma) = E_{\text{cts}} + \frac{Q_{\text{geo}} H}{K_t}(\sigma^* - \sigma)$$

where $K_t = \epsilon\, K_c$ ($\epsilon = 10^{-5}$). The CTS location $\sigma^*$ is where the cold profile first reaches $E_{\text{pmp}}(\sigma)$:

$$\frac{Q_{\text{geo}} H (1 - \sigma^*)}{k_i} = T_{\text{pmp}}(\sigma^*) - T_s$$

which is solved by Brent's method.

**Three sub-tests:**

1.  **`test_temperate_bed_still_applies_geothermal_flux`** — Sets $E$ uniformly above $E_{\text{pmp}}$ (no gradients), computes one residual, and checks that the bed residual equals $-Q_{\text{geo}}/(h \cdot \Delta\sigma_{\text{half}})$. Confirms the Neumann heat flux is not disabled for temperate ice.

2.  **`test_temperate_bed_is_not_clamped_to_pmp`** — From a uniform temperate initial state, runs a few column sweeps and asserts $E_{\text{bed}} > E_{\text{pmp}}$. Confirms enthalpy is not hard-clamped to the pressure melting point.

3.  **`test_steady_state_polythermal_profile`** — Runs the solver to steady state with adaptive time stepping (dt ramps from 100 kyr to $10^6$ kyr when converging) and verifies the final profile against the semi-analytical reference.

**Tolerances:**

| Metric                        | Tolerance | Typical result            |
|-------------------------------|-----------|---------------------------|
| Temperature error             | $< 0.1$ K | $\sim 4 \times 10^{-4}$ K |
| Enthalpy relative error       | $< 2\%$   | $\sim 1.2\%$              |
| Basal $\omega$ relative error | $< 2\%$   | $\sim 1.2\%$              |

**Convergence note:** The temperate zone has a diffusion timescale $\tau_{\text{temp}} = (\sigma^* H)^2 / (\kappa_{\text{temp}} \pi^2)$ that is $1/\epsilon = 10^5\times$ longer than the cold zone. With fixed time stepping this would require $\sim 10^5$ steps; adaptive dt collapses this to $\sim 300$ steps by ramping toward pseudo-steady-state ($\Delta t \to \infty$).

### Advection-diffusion (`test_enthalpy_advection.py`)

| Parameter        | Value                 |
|------------------|-----------------------|
| $H$              | 1000 m                |
| $T_s$            | 243.15 K ($-30°$C)    |
| $Q_{\text{geo}}$ | 0.04 W/m$^2$          |
| $N_z$            | 41 (uniform $\sigma$) |

Same cold slab as the cold column test, but with prescribed vertical advection ($\dot\sigma \neq 0$). No horizontal velocity. The base stays cold so $K = K_c$ throughout.

#### Constant $\dot\sigma$ (exponential solution)

A spatially uniform $\dot\sigma = w < 0$ (downward) gives the steady-state ODE:

$$\rho_i w \frac{dE}{d\sigma} = \frac{K_c}{h^2}\frac{d^2 E}{d\sigma^2}$$

with Peclet number $\text{Pe} = \rho_i w h^2 / K_c$. The general solution is $E = A + B\exp(\text{Pe}\,\sigma)$ with constants fixed by the surface Dirichlet and bed Neumann BCs:

$$B = \frac{-h\,Q_{\text{geo}}}{K_c\,\text{Pe}}, \quad A = E_s - B\exp(\text{Pe})$$

The test uses $\text{Pe} = -5$. Downward advection compresses the thermal boundary layer toward the bed, reducing $T_{\text{bed}}$ from 262 K (pure diffusion) to $\sim 247$ K.

#### Linear $\dot\sigma$ — Robin problem (error-function solution)

Accumulation-driven flow gives $\dot\sigma(\sigma) = -(\text{SMB}/h)\,\sigma$ (zero at the bed, maximum downward at the surface). The Robin Peclet number is $\text{Pe}_R = \rho_i\,\text{SMB}\,h / K_c$. The solution involves the error function:

$$E(\sigma) = E_s + \frac{h\,Q_{\text{geo}}}{K_c}\sqrt{\frac{\pi}{2\,\text{Pe}_R}} \left[\text{erf}\!\left(\sqrt{\tfrac{\text{Pe}_R}{2}}\right) - \text{erf}\!\left(\sigma\sqrt{\tfrac{\text{Pe}_R}{2}}\right)\right]$$

The test uses $\text{Pe}_R = 10$ ($\text{SMB} \approx 0.36$ m/yr). This case also tests the `BedSigmaAdvection` stencil at $k=0$ where $\dot\sigma = 0$: the one-sided upwind correctly contributes nothing when there is no flow through the bed.

**What they test:** - Sigma advection stencils (`get_sigma_advection_jac`, `get_bed_sigma_advection_jac`) - Advection-diffusion coupling in the column tridiagonal system - Correct upwind direction: downward $\dot\sigma$ upwinds from above - Convergence to analytical solution for both constant and depth-varying $\dot\sigma$

**Tolerance:** Enthalpy relative error $< 2\%$ ($\sim 0.9\%$ constant, $\sim 0.6\%$ Robin). The discretization error is $O(\Delta\sigma)$ from the upwind scheme.

### Planned benchmarks

-   **Kleiner et al. (2015) Column A:** Pure diffusion in a 200 m ice column with prescribed surface temperature and geothermal flux. Tests cold/temperate transition with known analytical solution.
-   **Kleiner et al. (2015) Column B:** Same geometry with Robin-type surface BC and an imposed vertical velocity profile ($\dot\sigma$ from accumulation). Tests advection-diffusion steady state.
-   **IGM enthalpy benchmark (Jouvet et al. 2024):** 3D dome geometry with coupled velocity-enthalpy evolution.

------------------------------------------------------------------------

## 7. References

-   Aschwanden, A., Bueler, E., Khroulev, C., and Blatter, H. (2012). An enthalpy formulation for glaciers and ice sheets. *Journal of Glaciology*, 58(209):441–457.
-   Kleiner, T., Rückamp, M., Bondzio, J.H., and Humbert, A. (2015). Enthalpy benchmark experiments for numerical ice sheet models. *The Cryosphere*, 9(1):217–228.
-   Jouvet, G. et al. (2024). Concepts and capabilities of the Instructed Glacier Model (IGM v2.2.1). Preprint, EarthArXiv.