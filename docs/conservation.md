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

Where: \* $u, v$ are horizontal velocities. \* $\dot{\sigma}$ is the vertical velocity relative to the moving sigma layers. \* $S$ encompasses all source, sink, and diffusive terms (e.g., strain heating, vertical diffusion).

### B. The 3D Cartesian Continuity Equation

To track mass, we begin with the standard assumption that ice is an incompressible fluid. In a fixed 3D Cartesian coordinate system $(x, y, z, t)$, mass conservation dictates that the divergence of the velocity field is zero:

$$\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} + \frac{\partial w}{\partial z} = 0$$

### C. The Sigma-Space Continuity Equation

By applying the chain rule for spatial derivatives to map from $(x, y, z)$ to $(x, y, \sigma)$, the incompressible continuity equation transforms into the **sigma-space continuity equation**:

$$\frac{\partial H}{\partial t} + \frac{\partial (Hu)}{\partial x} + \frac{\partial (Hv)}{\partial y} + \frac{\partial \omega}{\partial \sigma} = 0$$

Where $\omega = H\dot{\sigma}$ is the scaled relative vertical velocity.

------------------------------------------------------------------------

## 3. Deriving the Relative Vertical Velocity ($\omega$)

To understand the change of variables, we evaluate the vertical velocity in sigma space ($\dot{\sigma}$) by taking the material derivative of the coordinate definition $\sigma = \frac{z - b}{H}$:

$$\dot{\sigma} = \frac{d\sigma}{dt} = \frac{\partial \sigma}{\partial t} + u \frac{\partial \sigma}{\partial x} + v \frac{\partial \sigma}{\partial y} + w \frac{\partial \sigma}{\partial z}$$

Using the quotient rule, we find the partial derivatives of $\sigma$ with respect to time and space. For any variable $\xi \in \{x, y, t\}$:

$$\frac{\partial \sigma}{\partial \xi} = -\frac{1}{H} \left( \frac{\partial b}{\partial \xi} + \sigma \frac{\partial H}{\partial \xi} \right) \quad \text{and} \quad \frac{\partial \sigma}{\partial z} = \frac{1}{H}$$

Substituting these partial derivatives into the $\dot{\sigma}$ equation and multiplying the entire expression by the ice thickness $H$ gives us the scaled relative vertical velocity, $\omega$:

$$\omega = w - \left( \frac{\partial b}{\partial t} + u \frac{\partial b}{\partial x} + v \frac{\partial b}{\partial y} \right) - \sigma \left( \frac{\partial H}{\partial t} + u \frac{\partial H}{\partial x} + v \frac{\partial H}{\partial y} \right)$$

------------------------------------------------------------------------

## 4. The Kinematic Boundary Conditions (Mass)

The physical beauty of $\omega$ becomes apparent when we evaluate it at the domain boundaries ($\sigma = 1$ and $\sigma = 0$) and compare it to the Cartesian Kinematic Boundary Conditions.

In Cartesian coordinates, the KBCs describe the mass fluxes at the upper surface ($z = h_s$, where $h_s = b + H$) and the lower bed ($z = b$): \* **Surface KBC:** $w_s - \left( \frac{\partial h_s}{\partial t} + u_s \frac{\partial h_s}{\partial x} + v_s \frac{\partial h_s}{\partial y} \right) = -\dot{a}$ (where $\dot{a}$ is surface accumulation). \* **Bed KBC:** $w_b - \left( \frac{\partial b}{\partial t} + u_b \frac{\partial b}{\partial x} + v_b \frac{\partial b}{\partial y} \right) = -\dot{m}$ (where $\dot{m}$ is basal melt).

### A. At the Upper Surface ($\sigma = 1$)

Substitute $\sigma = 1$ into our $\omega$ equation. Expanding the $H$ terms as $(h_s - b)$ causes all the bed terms ($\frac{\partial b}{\partial t}, \frac{\partial b}{\partial x}, \frac{\partial b}{\partial y}$) to perfectly cancel out, leaving: $$\omega_s = w_s - \left( \frac{\partial h_s}{\partial t} + u_s \frac{\partial h_s}{\partial x} + v_s \frac{\partial h_s}{\partial y} \right)$$ This exactly matches the Cartesian Surface KBC. Therefore: $$\omega_s = -\dot{a}$$

### B. At the Lower Bed ($\sigma = 0$)

Substitute $\sigma = 0$ into our $\omega$ equation. The entire grid-stretching term collapses to zero: $$\omega_b = w_b - \left( \frac{\partial b}{\partial t} + u_b \frac{\partial b}{\partial x} + v_b \frac{\partial b}{\partial y} \right)$$ This exactly matches the Cartesian Bed KBC. Therefore: $$\omega_b = -\dot{m}$$

------------------------------------------------------------------------

## 5. Synthesis: Deriving the Conservative Form

We now use the product rule to merge the thermodynamic equation with the mass continuity equation.

**Step 1:** Multiply the advective enthalpy equation by the ice thickness $H$. Recalling that $H\dot{\sigma} = \omega$, this yields:

$$H \frac{\partial E}{\partial t} + Hu \frac{\partial E}{\partial x} + Hv \frac{\partial E}{\partial y} + \omega \frac{\partial E}{\partial \sigma} = HS$$

**Step 2:** Multiply the sigma-space continuity equation by the specific enthalpy $E$:

$$E \frac{\partial H}{\partial t} + E \frac{\partial (Hu)}{\partial x} + E \frac{\partial (Hv)}{\partial y} + E \frac{\partial \omega}{\partial \sigma} = 0$$

**Step 3:** Add the two equations together. Because of the reverse product rule ($A\,dB + B\,dA = d(AB)$), every term collapses into the spatial or temporal derivative of a product:

-   **Time:** $H \frac{\partial E}{\partial t} + E \frac{\partial H}{\partial t} = \frac{\partial (HE)}{\partial t}$
-   **X-Advection:** $Hu \frac{\partial E}{\partial x} + E \frac{\partial (Hu)}{\partial x} = \frac{\partial (HEu)}{\partial x}$
-   **Y-Advection:** $Hv \frac{\partial E}{\partial y} + E \frac{\partial (Hv)}{\partial y} = \frac{\partial (HEv)}{\partial y}$
-   **Sigma-Advection:** $\omega \frac{\partial E}{\partial \sigma} + E \frac{\partial \omega}{\partial \sigma} = \frac{\partial (E\omega)}{\partial \sigma}$

This results in the **strong conservative enthalpy equation**:

$$\frac{\partial (HE)}{\partial t} + \frac{\partial (HEu)}{\partial x} + \frac{\partial (HEv)}{\partial y} + \frac{\partial (E\omega)}{\partial \sigma} = HS$$

------------------------------------------------------------------------

## 6. Transforming the Thermodynamic Boundary Conditions

While the physical constraints on the domain (e.g., atmospheric temperature, geothermal heat) remain identical, shifting from tracking a *specific* quantity ($E$) to a *volume-integrated* quantity ($HE$) completely changes how boundary conditions are mathematically applied in a numerical solver. The formulation must shift from **state constraints** to **flux constraints**.

### A. The Upper Surface Boundary ($\sigma = 1$)

Physically, the surface boundary is determined by the atmospheric temperature and surface accumulation (snowfall/frost).

**In the Original (Advective) Form:**

This is treated as a classic **Dirichlet condition**. The specific enthalpy of the ice at the surface is prescribed directly based on the climate. It acts as a static anchor for the temperature profile: $$E(\sigma=1) = E_{surface}$$

**In the Conservative Form:**

Because an FVM solver calculates fluxes across cell faces, a Dirichlet condition is insufficient; the solver must track the *flux of total energy (*$HE$) crossing the $\sigma=1$ boundary. The Dirichlet condition is converted into an **advective flux constraint**. The enthalpy physically entering the domain through accumulation is defined by the surface KBC ($\omega_s = -\dot{a}$): $$Flux_{surface} = E_{surface} \omega_s = -E_{surface} \dot{a}$$ Instead of strictly overwriting the temperature of the top node, the solver actively adds $E_{surface} \dot{a}$ Joules of energy into the top control volume during every time step, perfectly balancing the added mass of the new snow.

### B. The Lower Bed Boundary ($\sigma = 0$)

Physically, the bed boundary balances the conductive heat flux into the ice with the geothermal heat flux ($q_{geo}$), frictional heating from sliding ($q_{fric}$), and energy lost to basal melting ($\dot{m}$).

**In the Original (Advective) Form:**

This is treated as a **Neumann condition** (a fixed gradient). In Cartesian coordinates, this relies on the vertical gradient of enthalpy: $-\kappa \left(\frac{\partial E}{\partial z}\right)_{z=b} = q_{geo} + q_{fric}$. To transform this into sigma coordinates, we apply the chain rule ($\frac{\partial E}{\partial z} = \frac{1}{H} \frac{\partial E}{\partial \sigma}$), yielding the **sigma-space basal gradient condition**: $$-\frac{\kappa}{H} \left(\frac{\partial E}{\partial \sigma}\right)_{\sigma=0} = q_{geo} + q_{fric}$$ *(Note the* $1/H$ scaling factor: thinner ice will mathematically produce steeper physical temperature gradients for the same $\Delta \sigma$ spacing).

**In the Conservative Form:**

The solver requires the *total energy flux* crossing the bottom cell face. This flux combines both the thermal gradient and the mass loss: 1. **Diffusive Flux:** The Neumann condition directly defines the conductive heat entering the ice: $q_{geo} + q_{fric}$. 2. **Advective Flux:** The energy leaving the domain because basal ice has melted relies on the basal KBC ($\omega_b = -\dot{m}$). The advected energy is $E_b \omega_b = -E_b \dot{m}$.

The conservative boundary condition combines these into a single **total flux constraint**: $$Flux_{bed} = -E_b \dot{m} + q_{geo} + q_{fric}$$ If ice melts at the bed, exactly $E_b \dot{m}$ enthalpy is removed from the system in the exact same step that the mass is removed, guaranteeing absolute global energy conservation.

------------------------------------------------------------------------

## 7. Implications for Numerical Implementation

When implementing this formulation in a finite volume solver, the shift from tracking $E$ to tracking $HE$ provides several critical numerical guarantees:

1.  **Perfect Telescopic Cancellation:** The horizontal flux leaving cell $i$ (e.g., $H_e E_e u_e$) is mathematically identical to the flux entering cell $i+1$. Summed over the entire domain, all internal fluxes cancel out exactly up to machine precision.
2.  **Robust Diagonal Dominance:** Because the cell volume $H$ is explicitly tied to the mass fluxes moving between cells, upwinded implicit solvers naturally build a highly stable, positive-definite matrix diagonal.
3.  **Consistency Under Grid Deformation:** If the velocity field is zero ($u=v=\omega=0$), but the ice is rapidly thinning due to basal melt ($\frac{\partial H}{\partial t} < 0$), the equation simplifies to $\frac{\partial (HE)}{\partial t} = 0$. The total energy in the column remains perfectly conserved, automatically scaling the specific enthalpy $E$ as the control volume compresses.