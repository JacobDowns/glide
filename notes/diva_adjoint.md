# The DIVA adjoint in GLIDE

This note traces the linearization and adjoint of GLIDE's DIVA stress balance from the mathematical dependency graph to the CUDA kernels that implement it. It is intended as a companion to [diva_numerics.md](diva_numerics.md): that note develops the forward model, while this one answers a narrower question:

> Given cotangents on the residual rows, which derivative paths must be followed to obtain cotangents on the state and model parameters?

The main result is that GLIDE uses a **reduced-state adjoint**. The globally coupled state remains $(u,v,H)$. Basal speed, the vertical viscosity profile, $\bar\eta$, and $\beta_{\mathrm{eff}}$ are eliminated by the cell-local DIVA closure; they are not added as global adjoint unknowns. Derivatives through that implicit closure are computed by running the converged local Newton calculations in dual-number arithmetic.

The coefficient path is differentiated with respect to velocity **and** thickness, so the DIVA momentum VJP is the exact transpose of the linearization over the whole coupled state, including

$$H\longrightarrow
(\bar\eta,\beta_{\mathrm{eff}})
\longrightarrow R_{u,v}.$$

Auxiliary fields -- notably the grounded fraction $\phi$ -- are still held fixed; see section 10.

## 1. The dependency graph

For a horizontal cell $c$, define the two velocity-dependent inputs to the local closure

$$q_c =
\begin{bmatrix}
q_{1,c}\\ q_{2,c}
\end{bmatrix}
=
\begin{bmatrix}
\dot\varepsilon_{\mathrm{mem},c}^2\\ \bar U_c
\end{bmatrix},$$

and the two coefficients required by the depth-integrated momentum residual

$$C_c=
\begin{bmatrix}
\bar\eta_c\\ \beta_{\mathrm{eff},c}
\end{bmatrix}
=\mathcal C_c(q_c;H_c,B_c,\beta_c,u_{c,c},m,\ldots).$$

Here $\mathcal C_c$ denotes the complete column calculation: vertical quadrature, the viscosity closures at the quadrature levels, the nonlinear basal-speed equation, and the final definitions of $\bar\eta_c$ and $\beta_{\mathrm{eff},c}$. The momentum residual can then be viewed as

$$R_{u,v}=\mathcal R_{u,v}(u,v,H,C(u,v,H)).$$

The currently implemented coefficient derivative treats $H$ as fixed, so its active graph is

``` mermaid
flowchart LR
    X["velocity facets: u, v"] --> Q["cell inputs: strain invariant, mean speed"]
    Q --> L["cell-local DIVA closure"]
    L --> C["cell coefficients: eta_bar, beta_eff"]
    C --> R["facet momentum residuals"]
    X --> R
    H["H and auxiliary fields: fixed in closure derivative"] --> L
    H --> R
```

There are two velocity paths into a momentum residual:

1.  a **direct stencil path**, in which $u$ and $v$ enter the stress and drag formulas while the DIVA coefficients are held fixed; and
2.  an **indirect coefficient path**, in which velocity changes $q_c$, the local closure changes $C_c$, and the changed coefficients alter nearby momentum rows.

Both paths are required. Freezing $\bar\eta$ and $\beta_{\mathrm{eff}}$ would retain only the first.

## 2. Cell-local coefficients do not imply cell-local residual rows

The closure output $C_c$ belongs to a cell, but the two momentum residuals are stored on facets. A $u$-residual is an $x$-momentum row on a vertical facet, and a $v$-residual is a $y$-momentum row on a horizontal facet. Stress divergences at those facets use cell-centered normal stresses and vertex-centered shear stresses. Consequently, one row reads several cell coefficients.

For the $u$-row on the facet between cells $l$ and $c$, use the local labels

``` text
tl   t
  |                 upper shear-stress vertex
 l | c     <- u residual on this facet
  |                 lower shear-stress vertex
bl   b
```

Its DIVA coefficient support is

| Momentum contribution | Cells supplying $\bar\eta$ | Cells supplying $\beta_{\mathrm{eff}}$ |
|------------------------|------------------------|------------------------|
| normal-stress difference | $l,c$ | -- |
| shear stress at upper vertex | $tl,t,l,c$ | -- |
| shear stress at lower vertex | $l,c,bl,b$ | -- |
| basal drag | -- | $l,c$ |

Thus one $u$-row can depend on $\bar\eta$ in the six-cell union $\{tl,t,l,c,bl,b\}$ and on $\beta_{\mathrm{eff}}$ in $\{l,c\}$. For the corresponding $v$-row, the pattern is rotated: the viscosity union is $\{tl,t,l,c,tr,r\}$ and the drag uses the cells $\{t,c\}$.

The cell-centered thickness residual $R_H$ does not read DIVA coefficients. It still couples directly to velocity through fluxes, and the momentum rows still depend directly on $H$ through $H\bar\eta$, driving stress, and other terms.

This distinction resolves an otherwise easy ambiguity: the **closure** is cell-local, but a **momentum residual row** is facet-local and reads the closures of nearby cells.

For code review, it is useful to classify the derivative terms row by row:

| Residual row | Direct state terms (coefficients fixed) | Indirect DIVA terms |
|------------------------|------------------------|------------------------|
| $R_u$ | stress-strain derivatives in $u,v$; $-\beta_{\mathrm{eff}}\,\delta u$; explicit $H$ derivatives of membrane and driving stresses | six-cell $\bar\eta$ path and two-cell $\beta_{\mathrm{eff}}$ path shown above |
| $R_v$ | rotated stress-strain derivatives; $-\beta_{\mathrm{eff}}\,\delta v$; explicit $H$ derivatives of membrane and driving stresses | rotated six-cell $\bar\eta$ path and two-cell $\beta_{\mathrm{eff}}$ path |
| $R_H$ | thickness time derivative, flux, calving, and mask terms; flux derivatives in $u,v$ | none through $\bar\eta$ or $\beta_{\mathrm{eff}}$ |

For either momentum row $i$, the indirect part can be written without hiding the stencil:

$$\delta_C R_i
=\sum_{c\in\mathcal N_\eta(i)}
  \frac{\partial R_i}{\partial\bar\eta_c}\,\delta\bar\eta_c
+\sum_{c\in\mathcal N_\beta(i)}
  \frac{\partial R_i}{\partial\beta_{\mathrm{eff},c}}
  \,\delta\beta_{\mathrm{eff},c}.$$

The two cell sets $\mathcal N_\eta(i)$ and $\mathcal N_\beta(i)$ are precisely the supports listed above. The first VJP pass is the transpose of these two sums.

## 3. The reduced Jacobian and its transpose

Let $x=(u,v)$ for the fixed-$H$ discussion, let $Q(x)$ assemble all cell inputs $q_c$, and let $C(Q)$ apply every cell closure. The reduced residual is

$$\widetilde R(x)=R(x,C(Q(x))).$$

Its Jacobian is

$$\frac{d\widetilde R}{dx}
=
\left.\frac{\partial R}{\partial x}\right|_C
+
\frac{\partial R}{\partial C}
\frac{\partial C}{\partial Q}
\frac{\partial Q}{\partial x}.$$

For a residual cotangent $\lambda$, the VJP is therefore

$$\left(\frac{d\widetilde R}{dx}\right)^T\lambda
=
\left.\frac{\partial R}{\partial x}\right|_C^T\lambda
+
\frac{\partial Q}{\partial x}^T
\frac{\partial C}{\partial Q}^T
\frac{\partial R}{\partial C}^T\lambda.$$

GLIDE evaluates this expression in four conceptual stages:

``` mermaid
flowchart RL
    Lam["residual cotangent: lambda"] --> Direct["direct fixed-coefficient stencil transpose"]
    Lam --> W["row to cell: W_eta, W_beta"]
    W --> AB["local closure transpose: A, B"]
    AB --> Scatter["cell-input transpose to velocity facets"]
    Direct --> Out["state cotangent"]
    Scatter --> Out
```

The runtime uses two VJP passes, with a closure-derivative launch between them:

1.  `vjp_body` computes the direct transpose and the row-to-cell coefficient cotangents $W_\eta$ and $W_\beta$.
2.  `compute_diva_derivs` evaluates and caches the cell closure Jacobians.
3.  `compute_diva_vjp_coeffs` applies the local closure transpose and scatters the two closure-input cotangents back to velocity facets.

Splitting the composite derivative at the cell is also useful computationally. The complete row-to-facet coefficient path reaches two grid spacings, whereas the row-to-cell and cell-to-facet halves each require only the existing one-cell halo.

## 4. Stage one: from residual rows to cell coefficients

Define the per-cell coefficient cotangents

$$W_{\eta,c}
=
\sum_i \lambda_i
\frac{\partial R_i}{\partial\bar\eta_c},
\qquad
W_{\beta,c}
=
\sum_i \lambda_i
\frac{\partial R_i}{\partial\beta_{\mathrm{eff},c}}.$$

The index $i$ runs over every momentum row that reads cell $c$. This is the transpose of the coefficient gather used by the forward residual: a forward row gathers neighboring coefficients, while the VJP scatters that row's cotangent back to those coefficient cells.

The local derivatives follow directly from the residual discretization:

- For a cell-centered membrane stress, the stored product is $H_c\bar\eta_c$, so $\partial(H_c\bar\eta_c)/\partial\bar\eta_c=H_c$.
- At a vertex, $H\bar\eta$ is the average of the four meeting cells, so each coefficient derivative contributes $H_c/4$ before multiplication by the shear-stress and divergence factors.
- DIVA drag at a $u$-facet is $-\tfrac12(\beta_{\mathrm{eff},l}+\beta_{\mathrm{eff},c})u$, so both coefficient partials are $-u/2$. The $v$-facet formula is the rotated analogue.

At the same time, `vjp_body` transposes every direct residual dependence on $u$, $v$, and $H$ with $\bar\eta$ and $\beta_{\mathrm{eff}}$ held fixed. The result after this first pass is therefore

- a partially accumulated state cotangent containing all direct paths; and
- the two cell arrays $W_\eta$ and $W_\beta$ containing the still-unfinished coefficient path.

## 5. Stage two: differentiating one implicit cell closure

For one cell, only the following $2\times2$ closure Jacobian is needed for the velocity VJP:

$$C_{q,c}
=
\begin{bmatrix}
\dfrac{\partial\bar\eta_c}{\partial q_{1,c}} &
\dfrac{\partial\bar\eta_c}{\partial q_{2,c}}\\[6pt]
\dfrac{\partial\beta_{\mathrm{eff},c}}{\partial q_{1,c}} &
\dfrac{\partial\beta_{\mathrm{eff},c}}{\partial q_{2,c}}
\end{bmatrix}.$$

There is no convenient explicit expression for this matrix because the closure contains nested implicit calculations. In particular, $U_b$ satisfies a scalar nonlinear equation of the form

$$F(U_b,q_c)=0,$$

and evaluation of $F$ involves the vertical integral $\mathcal I_2$, whose integrand contains viscosity values that are themselves found by scalar Newton solves at the quadrature levels.

GLIDE obtains the four entries by evaluating the complete templated closure twice with `DualFloat` inputs:

| Closure evaluation | Seed on $q_{1,c}$ | Seed on $q_{2,c}$ | Returned dual parts |
|-----------------|------------------:|------------------:|-----------------|
| strain-invariant derivative | 1 | 0 | $\bar\eta_{,q_1}$, $\beta_{\mathrm{eff},q_1}$ |
| mean-speed derivative | 0 | 1 | $\bar\eta_{,q_2}$, $\beta_{\mathrm{eff},q_2}$ |

A dual number stores a value and one directional derivative. Every overloaded arithmetic operation propagates both. Therefore the derivative follows the same program as the value:

$$q_c
\longrightarrow U_b
\longrightarrow \tau_b
\longrightarrow \eta(\zeta)
\longrightarrow \mathcal I_1,\mathcal I_2
\longrightarrow U_b
\longrightarrow \bar\eta,\beta_{\mathrm{eff}}.$$

The apparent loop in this diagram is the outer basal-speed Newton iteration. The dual arithmetic is run through that iteration, including the vertical quadrature and its inner viscosity Newton solves.

### 5.1 Why one extra dual Newton step is sufficient

Suppose a scalar root $y(x)$ is defined by $F(y,x)=0$, and Newton's map is

$$N(y,x)=y-\frac{F(y,x)}{F_y(y,x)}.$$

At a converged root, differentiating one application of this map gives

$$\dot y^+
=
\dot y-
\frac{F_y\dot y+F_x\dot x}{F_y}
=
-\frac{F_x}{F_y}\dot x.$$

The incoming derivative $\dot y$ cancels. Thus one dual-number Newton update performed **after the primal value has converged** replaces any derivative of the warm start or iteration history with the implicit root derivative. GLIDE uses this device for both the inner viscosity roots and the outer $U_b$ root.

This is not equivalent to differentiating only the final algebraic formulas after Newton. The final dual update still evaluates $F$, $F_y$, the vertical integrals, and all of their active dependencies. It is sufficient because the primal iterate is already at the root, where a Newton update applies the local inverse $F_y^{-1}$ required by the implicit-function theorem.

No additional global adjoint solve for $U_b$ is needed: every root is scalar and cell-local. An augmented formulation with $U_b$ and the quadrature-level viscosities as extra unknowns would produce local constraint blocks. Eliminating those blocks gives the same reduced derivative that the final dual Newton steps compute directly.

## 6. Stage three: coefficient cotangents to closure-input cotangents

Once $C_{q,c}$ is known, the two cotangents on the closure inputs are simply

$$\begin{aligned}
A_c
&=\frac{\partial J}{\partial q_{1,c}}
=W_{\eta,c}\frac{\partial\bar\eta_c}{\partial q_{1,c}}
+W_{\beta,c}\frac{\partial\beta_{\mathrm{eff},c}}{\partial q_{1,c}},\\
B_c
&=\frac{\partial J}{\partial q_{2,c}}
=W_{\eta,c}\frac{\partial\bar\eta_c}{\partial q_{2,c}}
+W_{\beta,c}\frac{\partial\beta_{\mathrm{eff},c}}{\partial q_{2,c}}.
\end{aligned}$$

In matrix form, $(A_c,B_c)^T=C_{q,c}^T(W_{\eta,c},W_{\beta,c})^T$. This small multiplication is the local closure VJP. It compresses all nested implicit details into two numbers per cell before the final stencil transpose.

## 7. Stage four: closure inputs back to velocity facets

The final step applies

$$\left(\frac{\partial Q}{\partial x}\right)^T
\begin{bmatrix}A\\B\end{bmatrix}
=
\sum_c
\left[
A_c\nabla_x q_{1,c}+B_c\nabla_x q_{2,c}
\right].$$

For cell $c$, the depth-averaged speed is assembled from its four bounding facets:

$$u_c=\frac{u_l+u_r}{2},\qquad
v_c=\frac{v_t+v_b}{2},\qquad
\bar U_c=\sqrt{u_c^2+v_c^2}.$$

For $\bar U_c>10^{-6}$, the four nonzero partials are

$$\frac{\partial\bar U_c}{\partial u_l}
=\frac{\partial\bar U_c}{\partial u_r}
=\frac{u_c}{2\bar U_c},
\qquad
\frac{\partial\bar U_c}{\partial v_t}
=\frac{\partial\bar U_c}{\partial v_b}
=\frac{v_c}{2\bar U_c}.$$

The implementation sets these partials to zero below that speed threshold.

The membrane invariant is

$$q_{1,c}
=\dot\varepsilon_{xx}^2
+\dot\varepsilon_{yy}^2
+\dot\varepsilon_{xx}\dot\varepsilon_{yy}
+\overline{\dot\varepsilon_{xy}^2},$$

where the normal strains use the four bounding facets and the shear term is an average of corner strains. Its transpose therefore reaches both the bounding facets and the neighboring facets used by the four corner strains. The device function `diva_scatter_cell_to_facets` reconstructs the same discrete strains as `membrane_eps_sq`, applies their analytic partials, and atomically adds the $A_c$ and $B_c$ contributions to all affected $u$- and $v$-facets.

This scatter completes the indirect path. Adding it to the direct result from stage one yields the fixed-$H$ velocity VJP.

## 8. End-to-end execution trace

For a supplied residual cotangent $\lambda=(\lambda_u,\lambda_v,\lambda_H)$, the implemented calculation is:

``` text
1. Zero output cotangents and the cell work arrays W_eta and W_beta.

2. Launch the DIVA residual VJP kernel.
   a. Transpose all direct residual dependencies into output u, v, and H.
   b. For every momentum row and every coefficient cell read by that row,
      accumulate lambda_row * d(row)/d(coefficient) into W_eta or W_beta.

3. Evaluate cell-local closure derivatives.
   a. Seed q1 = membrane strain invariant with dual part 1; run the full closure.
   b. Seed q2 = depth-averaged speed with dual part 1; run the full closure.
   c. Store the four derivatives of eta_bar and beta_eff.

4. Launch compute_diva_vjp_coeffs, one thread per cell.
   a. Form A and B by multiplying the transposed 2-by-2 closure Jacobian by
      (W_eta, W_beta).
   b. Scatter A * grad(q1) + B * grad(q2) to the velocity facets.

5. Add the indirect scatter to the direct cotangent already in the output.
```

The most useful code landmarks are:

| Role | Implementation |
|------------------------------------|------------------------------------|
| residual stencil and first VJP pass | `glide/cuda/residuals.cu`: `residual_body`, `vjp_body` |
| dual scalar arithmetic | `glide/cuda/common.cu`: `DualFloat` |
| complete local closure | `glide/cuda/diva.cu`: `diva_coeffs_cell` |
| closure derivative seeds | `glide/cuda/diva.cu`: `compute_diva_derivs` |
| local VJP and cell-to-facet scatter | `glide/cuda/diva.cu`: `compute_diva_vjp_coeffs`, `diva_scatter_cell_to_facets` |
| host-side two-pass orchestration | `glide/operators.py`: `_apply_diva_coeff_adjoints` |

Paths in this table are relative to the `glide` project directory.

## 9. Parameter and surface-velocity derivatives

The same cell work arrays can be reused for model parameters. For a cell-local parameter $p_c$,

$$\frac{dJ}{dp_c}
=W_{\eta,c}\frac{\partial\bar\eta_c}{\partial p_c}
+W_{\beta,c}\frac{\partial\beta_{\mathrm{eff},c}}{\partial p_c}.$$

`compute_diva_derivs` therefore also runs the closure with seeds on $\beta$, $u_c$, and $m$. These derivatives include both coefficient outputs; for example, changing basal drag can change $\bar\eta$ through basal traction, vertical shear, and the viscosity integral, as well as changing $\beta_{\mathrm{eff}}$ directly.

A surface-speed objective uses the same cell-to-facet scatter. Its cell cotangent is multiplied by the stored derivatives of surface speed with respect to $q_1$ and $q_2$, after which `diva_scatter_cell_to_facets` maps the result to velocity facets.

## 10. The thickness path

The direct VJP pass accumulates explicit derivatives with respect to $H$: it differentiates $H\bar\eta$ with $\bar\eta$ fixed, along with driving stress, flux, calving, and the other ordinary residual paths. On top of those sits the variation of the DIVA coefficients themselves with thickness,

$$\lambda^T
\frac{\partial R}{\partial C}
\frac{\partial C}{\partial H}\,\delta H,$$

which is nonzero because thickness changes the shear moments $\mathcal I_1,\mathcal I_2$, and through them the basal-speed constraint, the vertical viscosity profile, $\bar\eta$ and $\beta_{\mathrm{eff}}$.

Thickness reaches the closure through exactly one factor -- the moments carry $H$ because the quadrature runs over $\zeta\in[0,1]$ -- so the implementation is small:

1.  `H_c` is typed as the templated scalar `T` in `diva_coeffs_cell`, so a dual thickness propagates through the quadrature, both Newton solves and the closure root;
2.  `compute_diva_derivs` adds a sixth seed and stores $\partial\bar\eta/\partial H$, $\partial\beta_{\mathrm{eff}}/\partial H$ and $\partial u_s/\partial H$;
3.  `populate_diva_coeffs_dual` receives `d_H`, giving the JVP its thickness contribution;
4.  `compute_diva_vjp_coeffs` accumulates $W_\eta\bar\eta_{,H}+W_\beta\beta_{\mathrm{eff},H}$ into the cell thickness cotangent -- cell-local, since $H_c$ is the cell's own value and needs no scatter; and
5.  `compute_diva_us_vjp` does the same with $\mathrm{cot}\cdot u_{s,H}$ for surface objectives.

Only the Newton *denominators* stay primal, on the usual implicit-function-theorem argument (section 5.1).

Derivatives through diagnostic auxiliary fields, particularly grounded fraction $\phi(H)$ when it is recomputed from geometry, are still classified separately and are not included. Holding $\phi$ fixed and differentiating the diagnostic mapping $H\mapsto\phi$ describe different linearizations and should not be conflated.

## 11. Verification checklist

Three checks answer different questions:

1.  **Cell-closure finite differences.** Perturb $q_1$, $q_2$, $H$, and each active parameter independently and compare the resulting changes in $\bar\eta$, $\beta_{\mathrm{eff}}$, and surface speed with the stored dual derivatives. This is the sharpest check of the thickness path, because it isolates the closure from the residual's much larger direct $H$ terms.
2.  **Residual directional differences.** Compare $J_x\,\delta x$ against a finite difference of the full DIVA residual after recomputing the coefficients at both perturbed states, in both a velocity direction and a thickness direction. The thickness one is coarse -- removing the closure's $H$ seed moves it only from $5.7\times10^{-5}$ to $1.2\times10^{-4}$, because the residual's explicit $H$ dependence dominates.
3.  **JVP/VJP dot products.** Verify $\langle J\delta x,\lambda\rangle
    =\langle\delta x,J^T\lambda\rangle$ with $\delta H\ne0$. This confirms that the two implemented operators are transposes, but by itself cannot detect a physical dependency omitted from both operators -- which is precisely why check 1 exists.