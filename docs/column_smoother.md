# Column Smoother: Problem Statement and Algorithm

This document describes the precise mathematical problem solved by the column-wise Newton/Thomas smoother in each sweep, how it relates to the full enthalpy residual, and how the Jacobian is constructed.

## 1. The Full Enthalpy Residual

The enthalpy equation, after implicit Euler time discretization in sigma coordinates, defines a nonlinear residual $R_{i,j,k}(E) = 0$ at every grid point $(i,j,k)$:

$$
R_{i,j,k} = \underbrace{\frac{\rho_i}{\Delta t}(E_k - E_k^n)}_{\text{time derivative}}
+ \underbrace{\rho_i\left(u\frac{\partial E}{\partial x} + v\frac{\partial E}{\partial y}\right)_k}_{\text{horizontal advection}}
+ \underbrace{\rho_i\,\dot\sigma_k\frac{\partial E}{\partial\sigma}\bigg|_k}_{\text{vertical advection}}
- \underbrace{\frac{1}{h^2}\frac{\partial}{\partial\sigma}\!\left(K\frac{\partial E}{\partial\sigma}\right)_k}_{\text{vertical diffusion}}
- \underbrace{\phi_k}_{\text{strain heating}}
+ \underbrace{\rho_w L\,r_d\,\omega_k}_{\text{drainage}}
$$

where (dropping the $(i,j)$ index for clarity):

-   $E_k^n$ is the enthalpy at the previous time step
-   $\phi_k = A^{-1/n}|D(\mathbf{u})|^{1+1/n}$ is the strain heating (viscous dissipation), passed in as an externally computed field `phi_strain` — it does not depend on $E$
-   $\omega_k = \max(E_k - E_{\text{pmp},k},\; 0) / L$ is the water content, nonzero only in temperate ice ($E_k > E_{\text{pmp},k}$)
-   $r_d$ is the drainage rate (units s$^{-1}$), so the drainage source is $\rho_w L\,r_d\,\omega_k = \rho_w\,r_d\,\max(E_k - E_{\text{pmp},k},\; 0)$
-   $K(E)$ is the enthalpy diffusivity: $K = k_i/c_i$ in cold ice, smoothly transitioning to $\epsilon\,k_i/c_i$ ($\epsilon = 0.1$) in temperate ice via a sigmoid on $E - E_{\text{pmp}}$
-   $\dot\sigma$ is the sigma pseudo-velocity, which accounts for the moving coordinate system. It absorbs the vertical velocity $u_z$ and the geometric terms from the coordinate transformation:

$$h\dot{\sigma} = u_z - u_x\!\left(\frac{\partial b}{\partial x} + \sigma\frac{\partial h}{\partial x}\right) - u_y\!\left(\frac{\partial b}{\partial y} + \sigma\frac{\partial h}{\partial y}\right) - \sigma\frac{\partial h}{\partial t}$$

For SSA (depth-averaged velocity, no vertical shear), this simplifies to $\dot\sigma = (m_b - \sigma\cdot\text{SMB})/h$, which is linear in $\sigma$. In the column smoother $\dot\sigma_k$ is frozen — it is precomputed from the velocity field and does not depend on $E$

### Notation

Throughout this document we use uniform sigma spacing $\Delta\sigma = 1/(N_z - 1)$, with:

-   $\dot\sigma_k^+ = \max(\dot\sigma_k, 0)$, $\dot\sigma_k^- = \min(\dot\sigma_k, 0)$ (upwind splitting)
-   $K_{k+1/2} = K\!\bigl(\tfrac{1}{2}(E_k + E_{k+1}),\; \tfrac{1}{2}(E_{\text{pmp},k} + E_{\text{pmp},k+1})\bigr)$ (half-node diffusivity)
-   Horizontal advection in the advective form: $\rho_i(u\,\partial_x E + v\,\partial_y E)$, computed as the upwind flux divergence minus the $E\,\nabla\cdot\mathbf{u}$ correction

## 2. Discrete Residual by Cell Type

![Grid discretization](figures/column_grid.png)

**Left:** Vertical sigma grid for a single column. Enthalpy $E_k$ lives at sigma nodes (blue circles), with the surface node (red square) prescribed by the Dirichlet BC. Diffusivity $K$ is evaluated at half-nodes (green diamonds) between sigma levels. The bed node has a half-cell control volume (orange); interior nodes have full-cell control volumes (light blue). The geothermal + frictional heat flux enters through the bed face.

**Right:** Horizontal MAC grid at one sigma layer. Enthalpy is cell-centered; $u$-velocities live on vertical facets, $v$-velocities on horizontal facets. Upwind enthalpy fluxes are evaluated at each face using the donor cell's $E$ value.

The residual takes a different form at each of the four cell types: interior nodes, the surface (Dirichlet), the bed (Neumann), and thin-ice margins.

### Interior nodes ($k = 1, \ldots, N_z - 2$)

$$
\boxed{
R_k = \frac{\rho_i}{\Delta t}(E_k - E_k^n)
+ \rho_i\frac{\dot\sigma_k^+(E_k - E_{k-1}) + \dot\sigma_k^-(E_{k+1} - E_k)}{\Delta\sigma}
- \frac{1}{h^2\Delta\sigma^2}\bigl[K_{k+1/2}(E_{k+1} - E_k) - K_{k-1/2}(E_k - E_{k-1})\bigr]
+ H_k - \phi_k + \rho_w L\,r_d\,\omega_k
}
$$

where $H_k$ denotes the horizontal advection contribution (see below). The residual depends on $(E_{k-1}, E_k, E_{k+1})$ through vertical advection and diffusion, plus the diagonal-only contributions from horizontal advection and drainage.

**Horizontal advection $H_k$:** The advective form is computed per-face using upwind fluxes and the divergence correction:

$$
H_k = \rho_i\frac{F^x_{j+1/2,k} - F^x_{j-1/2,k} + F^y_{i+1/2,k} - F^y_{i-1/2,k}}{\Delta x} - \rho_i\,E_k\,\nabla_h \cdot \mathbf{u}_k
$$

where $F^x_{j+1/2,k} = u_{j+1/2,k}\,E_{\text{upwind}}$ is the upwind enthalpy flux on the x-face, and $\nabla_h \cdot \mathbf{u}_k = (u_{j+1/2} - u_{j-1/2} + v_{i+1/2} - v_{i-1/2})/\Delta x$ is the horizontal velocity divergence. When $E_{i,j,k}$ is the upwind donor for an outflow face, the flux depends on $E_k$ and contributes a Jacobian diagonal entry.

### Surface node ($k = N_z - 1$): Dirichlet

The surface is not a free unknown. The residual is set to zero:

$$
\boxed{R_{N_z-1} = 0}
$$

and in the column smoother the corresponding row enforces $E_{N_z-1} = E_s$ directly:

$$
b_{N_z-1} = 1, \quad a_{N_z-1} = c_{N_z-1} = 0, \quad \text{rhs}_{N_z-1} = -(E_{N_z-1} - E_s)
$$

### Bed node ($k = 0$): Neumann flux

The bed node sits at $\sigma = 0$. Its control volume extends to the midpoint with the first interior node, giving a half-cell width $\Delta\sigma_{\text{half}} = \Delta\sigma/2$. The diffusive flux is divided by this half-cell width to stay consistent with interior cells:

$$
\boxed{
R_0 = \frac{\rho_i}{\Delta t}(E_0 - E_0^n)
+ r_0^{\text{v-adv}}
+ \frac{1}{\Delta\sigma_{\text{half}}}\left(-\frac{K_{1/2}}{h^2}\frac{E_1 - E_0}{\Delta\sigma} - \frac{Q_{\text{geo}} + Q_{\text{fh}}}{h}\right)
+ H_0 - \phi_0 + \rho_w L\,r_d\,\omega_0
}
$$

where the bed sigma advection is one-sided (can only upwind from above):

$$
r_0^{\text{v-adv}} = \begin{cases}
\rho_i\,\dot\sigma_0\,(E_1 - E_0)/\Delta\sigma & \text{if } \dot\sigma_0 < 0 \\
0 & \text{otherwise}
\end{cases}
$$

The Neumann flux $(Q_{\text{geo}} + Q_{\text{fh}})/h$ is always applied regardless of thermal state.

### Thin-ice margins ($H < h_{\text{thin}}$)

When ice thickness is below `h_thin` (default 100 m), the entire column is clamped to the surface enthalpy:

$$
\boxed{R_k = 0 \quad \forall\, k, \qquad \delta E_k = E_s - E_k \quad \forall\, k}
$$

The residual kernel returns zero for these columns so they do not contribute to the convergence norm. The smoother directly drives the column toward $E_s$.

## 3. The Full Jacobian

### Derivation

The full Jacobian of the enthalpy system is the sparse matrix $\mathbf{J}$ with entries:

$$
J_{(i,j,k),\,(i',j',k')} = \frac{\partial R_{i,j,k}}{\partial E_{i',j',k'}}
$$

Each residual $R_{i,j,k}$ depends on:

1.  **Vertical neighbors** in the same column: $E_{i,j,k-1}$ and $E_{i,j,k+1}$ (through diffusion and advection)
2.  **The node itself**: $E_{i,j,k}$ (through all terms)
3.  **Horizontal neighbors** in the same sigma layer: $E_{i\pm1,j,k}$ and $E_{i,j\pm1,k}$ (through horizontal advection fluxes, only when the neighbor is the upwind donor)

So row $(i,j,k)$ of $\mathbf{J}$ has at most 7 nonzero entries: one vertical triad $(k-1, k, k+1)$ in the same column, and up to 4 horizontal neighbors in the same layer. For a single column $(i,j)$, the $N_z \times N_z$ diagonal block of $\mathbf{J}$ is tridiagonal.

### Term-by-term Jacobian entries

Each `get_*_jac()` function computes both the residual and its partial derivatives. The derivatives for each term are:

**Time derivative** ($\rho_i(E_k - E_k^n)/\Delta t$):

$$
\frac{\partial}{\partial E_k} = \frac{\rho_i}{\Delta t}
$$

Diagonal only. This is the dominant diagonal entry for small $\Delta t$.

**Interior vertical diffusion** ($-h^{-2}\,\partial_\sigma(K\,\partial_\sigma E)$):

$$
\frac{\partial}{\partial E_{k-1}} = -\frac{K_{k-1/2}}{h^2\Delta\sigma^2}, \quad
\frac{\partial}{\partial E_{k+1}} = -\frac{K_{k+1/2}}{h^2\Delta\sigma^2}, \quad
\frac{\partial}{\partial E_k} = \frac{K_{k-1/2} + K_{k+1/2}}{h^2\Delta\sigma^2}
$$

These are computed with $K$ frozen at the current state (the $dK/dE$ chain-rule terms are omitted for stability — see Section 5).

**Interior sigma advection** ($\rho_i\,\dot\sigma\,\partial_\sigma E$, upwind):

$$
\frac{\partial}{\partial E_{k-1}} = -\frac{\rho_i\,\dot\sigma_k^+}{\Delta\sigma}, \quad
\frac{\partial}{\partial E_{k+1}} = \frac{\rho_i\,\dot\sigma_k^-}{\Delta\sigma}, \quad
\frac{\partial}{\partial E_k} = \frac{\rho_i(\dot\sigma_k^+ - \dot\sigma_k^-)}{\Delta\sigma}
$$

**Bed diffusion** ($k = 0$, half-cell):

$$
\frac{\partial}{\partial E_0} = \frac{K_{1/2}}{h^2\,\Delta\sigma\,\Delta\sigma_{\text{half}}}, \quad
\frac{\partial}{\partial E_1} = -\frac{K_{1/2}}{h^2\,\Delta\sigma\,\Delta\sigma_{\text{half}}}
$$

The Neumann flux $(Q_{\text{geo}} + Q_{\text{fh}})/h$ has no $E$-dependence.

**Bed sigma advection** ($k = 0$, one-sided):

$$
\frac{\partial}{\partial E_0} = \begin{cases} \rho_i\,|\dot\sigma_0|/\Delta\sigma & \dot\sigma_0 < 0 \\ 0 & \text{otherwise} \end{cases}, \quad
\frac{\partial}{\partial E_1} = \begin{cases} -\rho_i\,|\dot\sigma_0|/\Delta\sigma & \dot\sigma_0 < 0 \\ 0 & \text{otherwise} \end{cases}
$$

**Drainage** ($\rho_w L\,r_d\,\omega_k$):

$$
\frac{\partial}{\partial E_k} = \begin{cases} \rho_w\,r_d & \text{if } E_k > E_{\text{pmp},k} \\ 0 & \text{otherwise} \end{cases}
$$

Diagonal only.

**Horizontal advection** (upwind flux divergence + advective correction):

$$
\frac{\partial H_k}{\partial E_k} = \rho_i\!\left(\frac{\partial F^x_R}{\partial E_k} - \frac{\partial F^x_L}{\partial E_k} + \frac{\partial F^y_B}{\partial E_k} - \frac{\partial F^y_T}{\partial E_k}\right)\frac{1}{\Delta x} - \rho_i\,\nabla_h \cdot \mathbf{u}_k
$$

Each face derivative $\partial F / \partial E_k$ is nonzero only when $(i,j,k)$ is the upwind donor (outflow face), in which case $\partial F / \partial E_k = u_{\text{face}}$. The horizontal neighbor derivatives $\partial H_k / \partial E_{\text{neighbor}}$ are nonzero when the neighbor is the donor, but these are **frozen** in the column smoother and only appear in the full Jacobian.

## 4. The Column Smoother Jacobian as a Restriction

### The full Jacobian structure

Ordering unknowns column-by-column, the full Jacobian $\mathbf{J}$ has a block structure:

$$
\mathbf{J} = \begin{pmatrix}
\mathbf{T}_{1,1} & \mathbf{C}_{1,2} & \cdots \\
\mathbf{C}_{2,1} & \mathbf{T}_{2,2} & \cdots \\
\vdots & & \ddots
\end{pmatrix}
$$

where each diagonal block $\mathbf{T}_{(i,j),(i,j)}$ is $N_z \times N_z$ tridiagonal (vertical coupling within column $(i,j)$), and each off-diagonal block $\mathbf{C}_{(i,j),(i',j')}$ is diagonal (horizontal coupling between columns at the same sigma level).

### The column smoother uses only the diagonal blocks

The column smoother solves column $(i,j)$ using only the diagonal block $\mathbf{T}_{(i,j),(i,j)}$, plus the diagonal entries from horizontal advection where the current cell is the upwind donor. That is, it uses the **block-diagonal restriction** of the full Jacobian:

$$
\mathbf{J}_{\text{column}} = \text{blkdiag}\!\left(\mathbf{T}_{1,1} + \mathbf{D}_1^{\text{h-adv}},\; \mathbf{T}_{2,2} + \mathbf{D}_2^{\text{h-adv}},\; \ldots\right)
$$

where $\mathbf{D}_{(i,j)}^{\text{h-adv}}$ is the diagonal matrix of horizontal advection self-derivatives (`horiz_adv_diag[k]`). The off-diagonal blocks $\mathbf{C}_{(i,j),(i',j')}$ — which represent horizontal neighbor coupling — are dropped entirely.

This is a standard block-Jacobi preconditioner. Each column solve is exact for the vertical coupling and approximate for horizontal coupling. The quality of this approximation depends on the relative strength of vertical vs. horizontal terms:

-   **Vertical diffusion** scales as $K/(h^2\Delta\sigma^2)$ — dominant in thin ice or fine vertical grids
-   **Horizontal advection** scales as $\rho_i\,u/\Delta x$ — dominant when velocities are large relative to $\Delta x$

When vertical coupling dominates (the typical case for thermal problems in ice sheets), the column smoother captures most of the Jacobian and converges rapidly. When horizontal advection dominates, the off-diagonal blocks $\mathbf{C}$ become significant and the block-Jacobi iteration converges slowly, requiring $O(N_x)$ sweeps to propagate information across the domain.

## 5. The Newton Iteration

Because the residual is nonlinear (the diffusivity $K(E)$ switches between cold and temperate regimes, and the drainage term activates at $E_{\text{pmp}}$), the column solve uses Newton's method. Each Newton step:

1.  **Evaluate** the column residual $r_k = R_{i,j,k}(\mathbf{E}^{(\nu)})$ at the current Newton iterate $\mathbf{E}^{(\nu)}$
2.  **Assemble** the tridiagonal Jacobian $J_{k,\ell} = \partial R_{i,j,k} / \partial E_{i,j,\ell}$ (only $\ell \in \{k-1, k, k+1\}$ are nonzero)
3.  **Solve** $J\,\delta\mathbf{E} = -\mathbf{r}$ via the Thomas algorithm
4.  **Update** $\mathbf{E}^{(\nu+1)} = \mathbf{E}^{(\nu)} + \alpha\,\delta\mathbf{E}$ (with relaxation $\alpha$)

The number of Newton steps per smoother application is controlled by `n_newton` (default 3).

### Local copy and horizontal advection linearization

The smoother keeps a local copy `E_local[k]` of the column that is updated across Newton iterations. Horizontal advection is precomputed once at the start (using the global `E` array for neighbor values), then linearized around the initial state during Newton iterations:

$$
r_k^{\text{h-adv}} \approx r_k^{\text{h-adv,0}} + \frac{\partial r_k^{\text{h-adv}}}{\partial E_k}\,(E_k^{(\nu)} - E_k^{(0)})
$$

where $r_k^{\text{h-adv,0}}$ is the horizontal advection residual at the start of the sweep and $\partial r_k^{\text{h-adv}} / \partial E_k$ is the diagonal Jacobian contribution from outflow faces.

### Tridiagonal assembly

For an interior node $k$, the tridiagonal entries are assembled by summing the Jacobian contributions from each physical term:

$$
a_k = \underbrace{\left(-\frac{K_{k-1/2}}{h^2\Delta\sigma^2}\right)}_{\text{diffusion}} + \underbrace{\left(-\frac{\rho_i\,\dot\sigma_k^+}{\Delta\sigma}\right)}_{\text{vert advection}}
$$

$$
c_k = \left(-\frac{K_{k+1/2}}{h^2\Delta\sigma^2}\right) + \left(\frac{\rho_i\,\dot\sigma_k^-}{\Delta\sigma}\right)
$$

$$
b_k = \frac{\rho_i}{\Delta t} + \frac{K_{k-1/2} + K_{k+1/2}}{h^2\Delta\sigma^2} + \frac{\rho_i(\dot\sigma_k^+ - \dot\sigma_k^-)}{\Delta\sigma} + (\text{drain\_jac.d\_E\_k}) + (\text{horiz\_adv\_diag}_k)
$$

$$
\text{rhs}_k = -R_k
$$

### What is approximate in the Jacobian

One deliberate approximation: the **diffusivity Jacobian freezes $K$ at the current state**. The exact linearization of the diffusion term includes chain-rule terms $dK/dE$ (the derivative of the cold-to-temperate sigmoid transition). These are omitted because:

-   When $|E_{k+1} - E_k|$ is large (e.g., warm surface BC on cold ice), the $dK/dE$ chain-rule term overwhelms the diagonal and destroys positive-definiteness
-   The frozen-$K$ Jacobian is always SPD (symmetric positive-definite for the diffusion block), ensuring the Thomas algorithm is stable
-   The Newton iteration compensates: even with an approximate Jacobian, the residual is exact, so Newton still converges to the correct solution

All other Jacobian entries (advection, drainage, time derivative, horizontal advection diagonal) are exact.

### Key design: shared `get_*_jac()` functions

Both the full residual kernel (`enthalpy_compute_residual`) and the column smoother kernel (`enthalpy_column_smooth`) call the **same** `get_*_jac()` functions. Each function returns a struct containing both the residual and the partial derivatives:

| Function | Returns | Residual field | Jacobian fields |
|---|---|---|---|
| `get_column_diffusion_jac` | `ColumnDiffusionJacobian` | `.res` | `.d_E_km1`, `.d_E_k`, `.d_E_kp1` |
| `get_bed_diffusion_jac` | `BedDiffusionJacobian` | `.res` | `.d_E_k`, `.d_E_kp1` |
| `get_sigma_advection_jac` | `SigmaAdvectionJacobian` | `.res` | `.d_E_km1`, `.d_E_k`, `.d_E_kp1` |
| `get_bed_sigma_advection_jac` | `BedSigmaAdvectionJacobian` | `.res` | `.d_E_k`, `.d_E_kp1` |
| `get_horiz_enthalpy_flux_jac` | `HorizEnthalpyFluxJacobian` | `.res` | `.d_E_here` |
| `get_drainage_jac` | `DrainageJacobian` | `.res` | `.d_E_k` |

The residual kernel reads only `.res`. The smoother reads both `.res` and `.d_E_*` to assemble the tridiagonal system. Because both paths call the same function, **the Jacobian is guaranteed to be the exact linearization of the residual** (modulo the frozen-$K$ approximation in the diffusion term).

## 6. Outer Iteration and Convergence

One call to `column_smooth` (one sweep) applies the above Newton solve independently to every column in parallel. The outer loop in `column_sweep` then:

1.  Updates the global solution: $E \leftarrow E + \omega\,\delta E$ (with relaxation $\omega$, default 1.0)
2.  Recomputes the **full** residual $R$ (including horizontal coupling)
3.  Checks convergence: $\|R\| / \|R_0\| < \text{rtol}$ or $\|R\| < \text{atol}$

Because horizontal neighbors are frozen in each sweep, the smoother is a block-Jacobi iteration. Each sweep propagates information one cell horizontally, so convergence of horizontal coupling requires $O(N_x)$ sweeps in the worst case. The `n_iter` cap and `absolute_tolerance` act as safety nets — the solver does its best and moves on.

## 7. Summary

| Aspect | Detail |
|---|---|
| **Problem per column** | Find $E_k$, $k=0,\ldots,N_z-1$ such that $R_{i,j,k} = 0$ with horizontal neighbors frozen |
| **Structure** | Tridiagonal (vertical diffusion + vertical advection + diagonal from horizontal advection and drainage) |
| **Solver** | Newton iteration with Thomas algorithm |
| **Boundary: surface** | Dirichlet $E_{N_z-1} = E_s$, residual = 0 |
| **Boundary: bed** | Neumann flux $(Q_{\text{geo}} + Q_{\text{fh}})/h$ with half-cell control volume |
| **Boundary: margins** | Thin ice ($H < h_{\text{thin}}$): clamp to $E_s$, residual = 0 |
| **Fixed sources** | $\rho_i E_k^n / \Delta t$ (previous time step) and $\phi_k$ (strain heating) |
| **Full Jacobian** | Block-tridiagonal columns + diagonal horizontal coupling |
| **Column Jacobian** | Block-diagonal restriction: drops off-diagonal (horizontal) blocks |
| **Approximation** | $K$ is frozen in the diffusion Jacobian (not in the residual) for stability |
| **Convergence** | Checked on the full (non-frozen) residual after each sweep |
