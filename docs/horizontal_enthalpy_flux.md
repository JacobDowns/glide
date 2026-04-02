# Horizontal Enthalpy Flux: Lax-Friedrichs in Advective Form

This document explains how horizontal advection in the enthalpy equation is discretized using a Lax-Friedrichs numerical flux while maintaining the advective form required by the sigma-coordinate formulation.

## 1. Why the Advective Form?

The enthalpy PDE in sigma coordinates uses the **advective form** of horizontal transport:

$$\rho_i \left(u \frac{\partial E}{\partial x} + v \frac{\partial E}{\partial y}\right)$$

not the **flux divergence form**:

$$\rho_i \, \nabla_h \cdot (\mathbf{u} E) = \rho_i \left(\frac{\partial (uE)}{\partial x} + \frac{\partial (vE)}{\partial y}\right)$$

The two differ by $\rho_i \, E \, \nabla_h \cdot \mathbf{u}$. In Cartesian coordinates this vanishes for incompressible flow, but in sigma coordinates the horizontal velocity divergence $\nabla_h \cdot \mathbf{u}$ is generally **nonzero** — ice thins and thickens, and the coordinate system absorbs this through the $\dot\sigma$ vertical transport term. The $E \, \nabla_h \cdot \mathbf{u}$ contribution is already accounted for via $\dot\sigma \, \partial E / \partial \sigma$, so the horizontal term must use the advective form to avoid double-counting.

## 2. The Finite Volume Challenge

Finite volume methods naturally produce fluxes in **divergence form**. Given enthalpy $E$ at cell centers and velocities $u$ at cell faces, a numerical flux $\hat{F}_{j+1/2}$ approximates the face flux $u \, E$. The flux divergence is:

$$\nabla_h \cdot (\mathbf{u} E) \approx \frac{\hat{F}_{j+1/2} - \hat{F}_{j-1/2}}{\Delta x} + \frac{\hat{G}_{i+1/2} - \hat{G}_{i-1/2}}{\Delta x}$$

To recover the advective form, we subtract $E \, \nabla_h \cdot \mathbf{u}$:

$$u \frac{\partial E}{\partial x} + v \frac{\partial E}{\partial y} \approx \underbrace{\frac{\hat{F}_{j+1/2} - \hat{F}_{j-1/2}}{\Delta x} + \frac{\hat{G}_{i+1/2} - \hat{G}_{i-1/2}}{\Delta x}}_{\text{flux divergence}} - \underbrace{E_{i,j} \frac{u_{j+1/2} - u_{j-1/2} + v_{i+1/2} - v_{i-1/2}}{\Delta x}}_{E \, \nabla_h \cdot \mathbf{u}}$$

This correction is computed from the raw face velocities and is exact for the physical (central) part of the flux.

## 3. The Lax-Friedrichs Flux

The Lax-Friedrichs numerical flux at a face with velocity $u$ and left/right enthalpy values $E_l$, $E_r$ is:

$$\hat{F}^{LF} = \underbrace{\tfrac{1}{2} u (E_l + E_r)}_{\text{central flux}} - \underbrace{\tfrac{1}{2} \alpha (E_r - E_l)}_{\text{numerical dissipation}}$$

The central flux is the physical transport. The dissipation term is a numerical artifact that stabilizes the scheme by adding diffusion proportional to the wave speed $\alpha$.

### Choice of $\alpha$

The wave speed $\alpha$ controls the amount of numerical dissipation. The implementation uses:

$$\alpha = |u| \sqrt{1 + c}$$

where $c$ is a configurable regularization constant (`lf_c`, default $10^{-12}$). This gives:

| $c$     | Behavior                                                       |
|---------|----------------------------------------------------------------|
| $c = 0$ | Exact upwind ($\alpha = |u|$)                                  |
| $c > 0$ | Slightly more dissipative than upwind by a factor $\sqrt{1+c}$ |

This formulation has two key properties:

1.  **Zero dissipation at zero velocity**: When $u = 0$, $\alpha = 0$ regardless of $c$. This prevents spurious cross-diffusion on faces with no flow.

2.  **Smooth Jacobian**: The derivatives $\partial \hat{F} / \partial E_l = \tfrac{1}{2}(u + \alpha)$ and $\partial \hat{F} / \partial E_r = \tfrac{1}{2}(u - \alpha)$ are both always nonzero when $u \neq 0$, avoiding the discontinuous upwind switch where the donor cell flips with the sign of $u$.

### Why not $\alpha = \sqrt{u^2 + c}$?

The momentum solver uses $\alpha = \sqrt{u^2 + c}$ with $c = 10$ (in m$^2$/yr$^2$ units). This gives a baseline dissipation of $\sqrt{c}$ even when $u = 0$, which is acceptable for the mass flux where $H$ and $u$ are coupled unknowns.

For the enthalpy equation, this baseline dissipation is problematic:

-   **Spurious cross-diffusion**: On faces with zero velocity (e.g., y-faces when flow is purely in x), $\alpha = \sqrt{c} \neq 0$ creates artificial diffusion perpendicular to the flow. This breaks symmetry and adds non-physical heat transport.
-   **Diagonal contamination**: The dissipation contributes to the column smoother's diagonal via $\partial \hat{F} / \partial E_{\text{here}}$. For large $\Delta t$ (pseudo-steady-state), this can dominate the physical diagonal $\rho_i / \Delta t$, degrading convergence.

The $\alpha = |u|\sqrt{1+c}$ formulation avoids both issues: dissipation is strictly proportional to the flow speed.

## 4. Interaction with the Advective Form Correction

The divergence-to-advective correction must be consistent with the flux. The LF flux has two parts:

-   **Central flux** $\tfrac{1}{2} u (E_l + E_r)$: This is the discretization of $u E$. Its divergence includes $E \, \nabla \cdot \mathbf{u}$, which the correction removes.

-   **Dissipation** $-\tfrac{1}{2} \alpha (E_r - E_l)$: This is a purely diffusive term with no velocity divergence content. It should **not** be corrected.

With $\alpha = |u|\sqrt{1+c}$, the dissipation term is proportional to $|u|$, and the correction $E \, \text{div}(\mathbf{u})$ uses the same face velocities. The correction naturally applies to the central flux part (which is proportional to $u$) and is consistent with the dissipation part (which is proportional to $|u|$). When $u = 0$ on a face, both the central flux and the dissipation are zero, so the correction is trivially consistent.

If we had used $\alpha = \sqrt{u^2 + c}$ with constant $c > 0$, the dissipation would be nonzero on faces where $u = 0$, but the correction $E \, \text{div}(\mathbf{u})$ would not account for this — creating an inconsistency that manifests as symmetry breaking.

## 5. Jacobian Structure

The per-face Jacobian has derivatives with respect to both cells:

$$\frac{\partial \hat{F}}{\partial E_l} = \tfrac{1}{2}(u + \alpha), \qquad \frac{\partial \hat{F}}{\partial E_r} = \tfrac{1}{2}(u - \alpha)$$

In the column smoother, only the derivative with respect to the **center cell** enters the tridiagonal diagonal. The center cell is $E_l$ for the right/bottom face and $E_r$ for the left/top face. After assembling the flux divergence and the advective correction:

$$\frac{\partial}{\partial E_{\text{here}}} = \rho_i \frac{(\partial \hat{F}_R / \partial E_l) - (\partial \hat{F}_L / \partial E_r)}{\Delta x} - \rho_i \, \nabla_h \cdot \mathbf{u}$$

The neighbor derivatives $\partial \hat{F} / \partial E_{\text{neighbor}}$ are dropped (frozen in the column smoother), corresponding to the off-diagonal blocks of the full Jacobian.

## 6. Comparison with the Momentum Solver

| Aspect | Momentum (flux.cu) | Enthalpy (enthalpy.cu) |
|------------------------|------------------------|------------------------|
| **Transported quantity** | $H$ (thickness) | $E$ (enthalpy) |
| **Flux** | $q = Hu$ (nonlinear: both unknowns) | $F = uE$ (linear: $u$ frozen) |
| **Form** | Flux divergence (mass conservation) | Advective (sigma coordinates) |
| $\alpha$ | $\sqrt{u^2 + 10}$ (m/yr units, constant baseline) | $|u|\sqrt{1+c}$ (m/s units, no baseline) |
| **Baseline diffusion** | $\sqrt{10} \approx 3.16$ m/yr always | Zero when $u = 0$ |
| **Jacobian** | $d_{H_l}$, $d_{H_r}$, $d_u$ (all three unknowns) | $d_{E_l}$, $d_{E_r}$ ($u$ frozen) |
| **Divergence correction** | Not needed (conservative form) | Subtract $E \, \nabla_h \cdot \mathbf{u}$ |

Both use the same Lax-Friedrichs structure; they differ in the $\alpha$ formulation because the enthalpy equation requires the advective form and the velocities are frozen (not co-solved).

## 7. Configuration

The regularization constant is configurable from Python:

``` python
thermal.ops.smoother_config.lf_c = cp.float32(1e-12)  # default
```

Setting `lf_c = 0` recovers exact upwind (non-differentiable Jacobian at $u = 0$). Increasing `lf_c` adds more dissipation proportional to $|u|$, which can help stability for strongly advection-dominated problems but is unnecessary for typical ice sheet configurations.