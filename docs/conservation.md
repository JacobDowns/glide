# Deriving the Conservative Enthalpy Equation in Sigma Coordinates

## 1. The Rationale: Why Conservative Form Matters for FVM

The Finite Volume Method (FVM) is built on the physical principle that the rate of change of a conserved quantity within a control volume is exactly balanced by the fluxes crossing its boundaries. Mathematically, this relies on the Divergence Theorem, which requires the governing partial differential equation (PDE) to be written in strong conservation (divergence) form:

$$\frac{\partial U}{\partial t} + \nabla \cdot \mathbf{F} = Q$$

In a terrain-following sigma coordinate system ($\sigma = \frac{z - b}{H}$), the computational grid stretches and shrinks as the ice thickness ($H$) evolves. If the enthalpy equation is solved in its standard advective form, the solver tracks the transport of specific enthalpy ($E$) but ignores the changing physical volume of the cell. This results in the numerical creation or destruction of total energy whenever the ice thickens or thins.

To restore strict energy conservation in FVM, the conserved quantity must shift from specific enthalpy ($E$) to **depth-integrated enthalpy per unit area** ($HE$).

------------------------------------------------------------------------

## 2. The Governing Continuous Equations

To derive the conservative form, we must fuse the thermodynamic tracking of energy with the physical tracking of mass.

### A. The Advective Enthalpy Equation

The advective transport of specific enthalpy $E$ in a moving sigma-coordinate framework is given by the material derivative:

$$\frac{\partial E}{\partial t} + u \frac{\partial E}{\partial x} + v \frac{\partial E}{\partial y} + \dot{\sigma} \frac{\partial E}{\partial \sigma} = S$$

Where: \* $u, v$ are horizontal velocities. \* $\dot{\sigma}$ is the vertical velocity relative to the moving sigma layers. \* $S$ encompasses all source, sink, and diffusive terms (e.g., strain heating, Fourier conduction, drainage).

### B. The Sigma-Space Continuity Equation

Assuming ice is incompressible ($\nabla \cdot \mathbf{v} = 0$), mapping the 3D Cartesian continuity equation into sigma coordinates yields:

$$\frac{\partial H}{\partial t} + \frac{\partial (Hu)}{\partial x} + \frac{\partial (Hv)}{\partial y} + \frac{\partial \omega}{\partial \sigma} = 0$$

Where $\omega = H\dot{\sigma}$ is the scaled relative vertical velocity.

------------------------------------------------------------------------

## 3. The Role of the Kinematic Boundary Conditions (KBCs)

The relative vertical velocity ($\omega$) is defined by expanding the material derivative of the sigma coordinate mapping:

$$\omega = w - \left( \frac{\partial b}{\partial t} + u \frac{\partial b}{\partial x} + v \frac{\partial b}{\partial y} \right) - \sigma \left( \frac{\partial H}{\partial t} + u \frac{\partial H}{\partial x} + v \frac{\partial H}{\partial y} \right)$$

Evaluating this equation at the domain boundaries reveals that $\omega$ naturally encodes the mass fluxes defined by the Kinematic Boundary Conditions:

-   **At the surface (**$\sigma = 1$): $\omega_s = -\dot{a}$ (where $\dot{a}$ is the surface mass balance or accumulation rate).
-   **At the bed (**$\sigma = 0$): $\omega_b = -\dot{m}$ (where $\dot{m}$ is the basal melt rate).

Because $\omega$ respects these physical mass boundaries, the sigma-space continuity equation is globally mass-conserving when integrated over the vertical column.

------------------------------------------------------------------------

## 4. Synthesis: Deriving the Conservative Form

We now use the product rule to merge the thermodynamic equation with the mass continuity equation.

**Step 1:** Multiply the advective enthalpy equation by the ice thickness $H$. Recalling that $H\dot{\sigma} = \omega$, this yields:

$$H \frac{\partial E}{\partial t} + Hu \frac{\partial E}{\partial x} + Hv \frac{\partial E}{\partial y} + \omega \frac{\partial E}{\partial \sigma} = HS$$

**Step 2:** Multiply the sigma-space continuity equation by the specific enthalpy $E$:

$$E \frac{\partial H}{\partial t} + E \frac{\partial (Hu)}{\partial x} + E \frac{\partial (Hv)}{\partial y} + E \frac{\partial \omega}{\partial \sigma} = 0$$

**Step 3:** Add the two equations together. Because of the reverse product rule ($A\,dB + B\,dA = d(AB)$), every term perfectly collapses into the spatial or temporal derivative of a product:

-   **Time:** $H \frac{\partial E}{\partial t} + E \frac{\partial H}{\partial t} = \frac{\partial (HE)}{\partial t}$
-   **X-Advection:** $Hu \frac{\partial E}{\partial x} + E \frac{\partial (Hu)}{\partial x} = \frac{\partial (HEu)}{\partial x}$
-   **Y-Advection:** $Hv \frac{\partial E}{\partial y} + E \frac{\partial (Hv)}{\partial y} = \frac{\partial (HEv)}{\partial y}$
-   **Sigma-Advection:** $\omega \frac{\partial E}{\partial \sigma} + E \frac{\partial \omega}{\partial \sigma} = \frac{\partial (E\omega)}{\partial \sigma}$

This results in the **strong conservative enthalpy equation**:

$$\frac{\partial (HE)}{\partial t} + \frac{\partial (HEu)}{\partial x} + \frac{\partial (HEv)}{\partial y} + \frac{\partial (E\omega)}{\partial \sigma} = HS$$

------------------------------------------------------------------------

## 5. Implications for Numerical Implementation

When implementing this formulation in a finite volume solver, the shift from tracking $E$ to tracking $HE$ provides several critical numerical guarantees:

1.  **Perfect Telescopic Cancellation:** The horizontal flux leaving cell $i$ (e.g., $H_e E_e u_e$) is mathematically identical to the flux entering cell $i+1$. Summed over the entire domain, all internal fluxes cancel out exactly up to machine precision.
2.  **Robust Diagonal Dominance:** Because the cell volume $H$ is explicitly tied to the mass fluxes moving between cells, upwinded implicit solvers (like column-wise Newton-Vanka smoothers) naturally build a highly stable, positive-definite matrix diagonal.
3.  **Consistency Under Grid Deformation:** If the velocity field is zero ($u=v=\omega=0$), but the ice is rapidly thinning due to basal melt ($\frac{\partial H}{\partial t} < 0$), the equation simplifies to $\frac{\partial (HE)}{\partial t} = 0$. The total energy in the column remains perfectly conserved, automatically scaling the specific enthalpy $E$ as the control volume compresses.