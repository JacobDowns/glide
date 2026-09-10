# Building DIVA forward from a discrete variational principle

*The clean way to get a numerically self-adjoint stress balance — Doug's requirement — is to build it **forward**: write a discrete dissipation potential* $J_h(U)$ as a single scalar function of the velocity unknowns, then differentiate it to obtain the momentum equations. Because the equations are then the gradient of a scalar, the Jacobian is a Hessian and self-adjointness is automatic — you cannot lose it. This note carries that out for DIVA: the continuous variational principle, how the fields are represented / interpolated / quadratured on the staggered grid, the resulting discrete VP, and its differentiation into the DIVA forward model. One quadrature choice — how the basal term forms the cell speed — decides whether the drag is local; the sum-of-squares choice (which MOLHO already uses) gives a DIVA that is self-adjoint **and** local.

Companion notes:

-   [diva_drag_discretization.md](diva_drag_discretization.md) — the drag stencil and the C-grid staggering, with figures; consistent-vs-lumped mass in finite-element terms.
-   [diva_column_energy_derivation.md](diva_column_energy_derivation.md) — how the continuum column energy is condensed so that a functional of $\bar{\mathbf u}$ alone remains.

------------------------------------------------------------------------

## 1. The plan, and why building forward guarantees self-adjointness

A **discrete variational principle** is a scalar function $J_h(U)$ of the discrete unknowns $U$ — here the facet velocities — whose gradient is the discrete momentum residual:

$$
R(U)=\nabla_U J_h(U)
\qquad\Longrightarrow\qquad
\frac{\partial R}{\partial U}=\nabla_U^2 J_h .
$$

A gradient's Jacobian is a **Hessian**, hence symmetric. So the moment the equations are obtained by differentiating a single scalar, **self-adjointness is automatic — it cannot be broken.** The only decisions left in building $J_h$ — the quadrature rule and the interpolations — control accuracy and stencil locality, never symmetry.

This is the reverse of the usual finite-volume route, which assembles the discrete *equations* directly, term by term, and leaves self-adjointness as something to check afterward (the residual is self-adjoint iff it happens to be a gradient — the integrability condition). That backward route can fail, and for GLIDE's stock DIVA it did (§8). Building forward makes the failure impossible. It is also the original derivation of DIVA (Goldberg discretizes the functional, not the equations) and the way MOLHO is built.

The plan of the note is exactly the forward pipeline:

$$
\underbrace{J[\bar{\mathbf u}]}_{\S2}
\;\xrightarrow[\text{quadrature (}\S3\text{)}]{\text{represent / interpolate}}\;
\underbrace{J_h(U)}_{\S4}
\;\xrightarrow[\;]{\;\nabla_U\;}\;
\underbrace{R(U)=0}_{\S6}\;,\qquad \text{self-adjoint by construction (}\S7\text{).}
$$

------------------------------------------------------------------------

## 2. The continuous variational principle

DIVA's velocity minimizes a dissipation action. After the vertical/deformational structure is condensed (companion note), it is a functional of the single depth-averaged field $\bar{\mathbf u}=(\bar u,\bar v)$:

$$
J[\bar{\mathbf u}]
=\underbrace{\int_\Omega \psi\big(\dot\varepsilon(\bar{\mathbf u})\big)\,d\Omega}_{\text{viscous } J_{\rm visc}}
\;+\;\underbrace{\int_\Omega \phi_b\big(|\bar{\mathbf u}|\big)\,d\Omega}_{\text{basal } J_{\rm b}}
\;-\;\underbrace{\int_\Omega H\,\nabla s\cdot\bar{\mathbf u}\,d\Omega}_{\text{driving } \ell}.
$$

-   $\psi$ — the viscous dissipation **potential**; for Glen flow $\psi=\tfrac{2n}{n+1}B\,(\dot\varepsilon_e^2)^{\frac{n+1}{2n}}$, with $\dot\varepsilon_e^2$ the effective strain invariant (membrane strains plus the condensed vertical shear), and $\partial\psi/\partial\dot\varepsilon=\sigma$ the deviatoric stress.
-   $\phi_b$ — the basal dissipation, a function of the **speed** $|\bar{\mathbf u}|$, with $\partial\phi_b/\partial\bar{\mathbf u}=\beta_{\rm eff}\bar{\mathbf u}=\vec\tau_b$; equivalently $\phi_b'(s)=\beta_{\rm eff}(s)\,s$, so $\beta_{\rm eff}=\phi_b'(s)/s$.
-   $\ell$ — the driving term, **linear** in $\bar{\mathbf u}$ (units divided by $\rho g$). Being linear it has no Hessian, so it never affects self-adjointness.

The stress balance is the first variation, $R=\delta J/\delta\bar{\mathbf u}=0$.

------------------------------------------------------------------------

## 3. Representing, interpolating, and quadraturing the fields

To turn $J[\bar{\mathbf u}]$ into a function of finitely many numbers we make three choices — a **representation** of $\bar{\mathbf u}$, an **interpolation/differencing** rule to build each integrand, and a **quadrature** rule for the integrals.

**Representation (the staggered grid).** The unknowns are the velocities, placed on cell facets: $u$ on the E–W faces, $v$ on the N–S faces (fig. in the drag-discretization note). The scalars $H,B,\beta$ sit at cell centres.

**Interpolation / differencing.** How each integrand is built from the facet DOFs — and *where* it therefore lives — is forced by where a derivative is cheapest and second-order:

-   normal strains $\partial_x u=(u_r-u_l)/\Delta x$, $\partial_y v=(v_t-v_b)/\Delta x$ are differences of a cell's *own* two facets → they live at the **cell centre**;
-   the shear strain $\partial_y u+\partial_x v$ is only computable at a **vertex** (two $u$-facets stacked, two $v$-facets side by side), but the effective-strain *invariant* is assembled per **cell**, so the four surrounding vertex shears are folded back in as $\overline{\dot\varepsilon_{xy}^2}(c)=\tfrac14\sum_{\text{4 corners}}\dot\varepsilon_{xy}(q)^2$ — a nodal (sum-of-squares) average of the corners. The viscous potential is then a single per-cell scalar (§6.1);
-   the basal integrand needs the cell **speed** $|\bar{\mathbf u}_c|$, built from the cell's facets — *this is the one interpolation that is not forced* (§5);
-   the driving integrand $H\,\partial_x s$ is an $x$-derivative of cell scalars → it lives on the **u-facet**.

**Quadrature.** With each integrand at its natural point, every integral is the **one-point midpoint rule** — evaluate the integrand there, multiply by the area:

$$
\int_\Omega f\,d\Omega\;\approx\;\sum_{\text{points}} f(x_{\rm pt})\,\Delta x^2 .
$$

This is the standard Arakawa-C / MAC discretization; the one-point rule is what makes the differentiated energy land on the usual finite-difference stencils.

**Finite-element reading (optional).** Equivalently this is a mixed finite-element method with the velocity components in face-based spaces and **nodal quadrature** (quadrature points at the DOF locations); nodal quadrature is precisely what collapses the element integrals to these differences. In that language the one interpolation choice of §5 is "which quadrature do I use for $\int_c\beta|u|^2$" — nodal (at the facets) vs a centroid rule (interpolate to the centre) — i.e. lumped vs consistent mass.

------------------------------------------------------------------------

## 4. The discrete variational principle

Assembling §3 (a cell's four facets are $u_l,u_r,v_t,v_b$; $A=\Delta x^2$):

$$
\boxed{\;
J_h(U)=\sum_{\text{cells }c}\psi\big(\dot\varepsilon_e^2(c)\big)\,A
\;+\;\sum_{\text{cells }c}\phi_b\big(s_c\big)\,A
\;-\;\sum_{\text{facets }f}(H\,\partial_x s)_f\,u_f\,A
\;}
$$

-   $\psi\big(\dot\varepsilon_e^2(c)\big)=\tfrac{2n}{n+1}B_cH_c\big(\dot\varepsilon_e^2(c)\big)^{\frac{n+1}{2n}}$ — a single viscous potential per **cell**. Its invariant $\dot\varepsilon_e^2(c)$ carries **both** the cell's normal strains and the corner-averaged shear $\overline{\dot\varepsilon_{xy}^2}(c)$ (§3, written out in §6.1); the vertex shear stresses fall out when this one term is differentiated, so there is no separate vertex energy;
-   $\phi_b(s_c)$ — basal potential at the **cell**, needing the cell speed $s_c$;
-   the driving — one **linear** term per **facet**.

This scalar $J_h(U)$ *is* the discrete variational principle. Everything after this is bookkeeping: differentiate it, and self-adjointness comes with it.

------------------------------------------------------------------------

## 5. The one free choice: how to form the cell speed

The basal term needs $s_c^2=|\bar{\mathbf u}_c|^2$ from the four facets. Two second-order options:

$$
\textbf{(A) square-of-average:}\quad
s_c^2=\Big(\tfrac12(u_l+u_r)\Big)^2+\Big(\tfrac12(v_t+v_b)\Big)^2
\quad(\text{centroid rule / consistent mass}),
$$

$$
\textbf{(B) sum-of-squares:}\quad
s_c^2=\tfrac12\big(u_l^2+u_r^2\big)+\tfrac12\big(v_t^2+v_b^2\big)
\quad(\text{nodal rule / lumped mass}).
$$

They agree when the velocity is uniform across the cell and differ only by the sub-cell **variance**, $s^2_{\rm (B)}-s^2_{\rm (A)}=\tfrac14(u_l-u_r)^2+\tfrac14(v_t-v_b)^2=O(h^2\partial u)$, so both are legitimate. (A) carries a facet–facet **cross-term** $2u_lu_r$; (B) does not. That single difference is the whole story below. (B) is MOLHO's choice.

------------------------------------------------------------------------

## 6. Differentiate $J_h$ term by term — matched to the code

The discrete momentum equations are one facet-row each of $R(U)=\nabla_U J_h(U)=0$. We now compute those partial derivatives explicitly and match each to the stencil the code actually evaluates. To keep the algebra honest we do the exercise **twice**: first for **pure SSA**, where $\psi$ and $\phi_b$ are elementary closed forms, then for **DIVA**, which reuses the identical skeleton with a condensed $\psi$ and a small column solve standing in for $\phi_b$. Everywhere below, $A=\Delta x^2$ and a cell $c$ carries facets $(u_l,u_r,v_t,v_b)$.

Two conventions make the match exact. First, the code's momentum residual is the balance *per unit area* — the $A$ that multiplies every term of $J_h$ divides straight back out, so $\partial J_h/\partial u_f$ compared against a code stencil always loses one factor of $A=\Delta x^2$. Second, GLIDE writes the residual as $R=\text{driving}-\text{divergence}-\text{drag}$, i.e. as $-\nabla_U J_h$ (stationary points are the same); the signs below are written to land on the code's, so read $R=-\partial J_h/\partial(\cdot)$.

### 6.1 The membrane term $\psi$ (identical for SSA and DIVA)

Write out the cell effective strain the code forms ([viscosity.cu:177](glide/glide/cuda/viscosity.cu:177), `membrane_eps_sq`):

$$
\dot\varepsilon_e^2(c)=\underbrace{(\partial_x u)^2+(\partial_y v)^2+(\partial_x u)(\partial_y v)}_{\text{normal, from the cell's own 4 facets}}
\;+\;\underbrace{\overline{\dot\varepsilon_{xy}^2}(c)}_{\text{shear, corner-averaged}}+\varepsilon_{\rm reg},
\qquad
\overline{\dot\varepsilon_{xy}^2}(c)=\tfrac14\!\!\sum_{q\in\text{4 corners}}\!\!\dot\varepsilon_{xy}(q)^2,
$$

with $\partial_x u=(u_r-u_l)/\Delta x$, $\partial_y v=(v_t-v_b)/\Delta x$, and each corner shear $\dot\varepsilon_{xy}(q)=\tfrac12(\partial_y u+\partial_x v)$ from the two $u$-facets stacked and two $v$-facets beside that vertex ([viscosity.cu:158–175](glide/glide/cuda/viscosity.cu:158)). This is *itself* a nodal (sum-of-squares) quadrature of the shear over the cell's four corners — the same forward trick §5 is about to make for the drag. The viscous potential and viscosity are

$$
\psi(c)=\frac{2n}{n+1}\,B_c H_c\,\big(\dot\varepsilon_e^2(c)\big)^{\frac{n+1}{2n}},
\qquad
\eta_c=\tfrac12 B_c\big(\dot\varepsilon_e^2(c)\big)^{\frac{1-n}{2n}}
\quad(\text{so }\partial\psi/\partial\dot\varepsilon_e^2=2\eta_c H_c),
$$

matching [viscosity.cu:196–198](glide/glide/cuda/viscosity.cu:196). Differentiating $\sum_c\psi(c)A$ with respect to the facet $u_f$ (which is $u_r$ of the left cell and $u_l$ of the right cell $c$), the chain rule gives, for the right cell's contribution,

$$
\frac{\partial\psi(c)}{\partial u_f}
=\underbrace{\frac{\partial\psi}{\partial\dot\varepsilon_e^2}}_{2\eta_cH_c}
\cdot\underbrace{\frac{\partial\dot\varepsilon_e^2}{\partial\dot\varepsilon_{xx}}}_{2\dot\varepsilon_{xx}+\dot\varepsilon_{yy}}
\cdot\underbrace{\frac{\partial\dot\varepsilon_{xx}}{\partial u_f}}_{-1/\Delta x}
\;+\;(\text{shear corners})
=-\frac{1}{\Delta x}\,\underbrace{2\eta_cH_c\big(2\dot\varepsilon_{xx}+\dot\varepsilon_{yy}\big)}_{=\,\sigma_{xx}(c)} .
$$

Dividing by $A$ and adding the left cell (opposite sign) reproduces exactly $R_u\supset(\sigma_{xx,c}-\sigma_{xx,l})/\Delta x=\partial_x\sigma_{xx}$, the code's two half-terms at [residuals.cu:188](glide/glide/cuda/residuals.cu:188) and [204](glide/glide/cuda/residuals.cu:204), with $\sigma_{xx}=H\eta(4\partial_xu+2\partial_yv)$ from [stress.cu:97–107](glide/glide/cuda/stress.cu:97). The shear corners feed the $\partial_y\sigma_{xy}$ vertex terms ([residuals.cu:207–254](glide/glide/cuda/residuals.cu:207), $\sigma_{xy}$ at [stress.cu:223–231](glide/glide/cuda/stress.cu:223)) the same way.

*Frozen vs full.* The **residual** holds $\eta$ fixed (it reads a pre-computed `eta_local`), so it is the gradient of the *frozen-viscosity* quadratic $\tfrac12\!\int 2\eta H\,\dot\varepsilon{:}\dot\varepsilon$ — a Picard step, and symmetric. The **JVP** re-differentiates $\eta$ through $\dot\varepsilon_e^2$ as well, giving the true Hessian $\nabla^2\!\int\psi$; because $\dot\varepsilon_e^2(c)$ is one scalar per cell (and the corner shears enter as squares), that Hessian is symmetric too. This is precisely the caveat Goldberg flags — "ignoring the dependence of viscosity on strain rate" is the frozen step; keeping it is the full Hessian; **both are self-adjoint here.**

### 6.2 The driving term $\ell$ (linear, no Hessian)

$\ell=-\sum_f(H\partial_x s)_f\,u_f\,A$ differentiates to $\partial\ell/\partial u_f=-(H\partial_x s)_f A$, i.e. the per-facet driving stress $H_{\rm avg}(s_r-s_l)/\Delta x$ of [stress.cu:730](glide/glide/cuda/stress.cu:730) (`get_tau_dx_jac`, assembled at [residuals.cu:305](glide/glide/cuda/residuals.cu:305)). Being linear in $U$ it has **zero** second derivative, so it never touches symmetry — it only sets the right-hand side.

### 6.3 The basal term $\phi_b$ for SSA — and why GLIDE's SSA is not yet a gradient

The basal potential is a genuine closed form; it is already in the code, written out inside the DIVA closure as `F_basal` ([diva.cu:469–471](glide/glide/cuda/diva.cu:469)):

$$
\textbf{Weertman:}\quad
\phi_b(s)=\frac{\beta_g}{m+1}\,(s^2+u_{\rm reg})^{\frac{m+1}{2}}+\tfrac12\,w_d\,s^2,
\qquad
\textbf{reg. Coulomb:}\quad
\phi_b(s)=\beta_g\big[s-u_c\log(s{+}u_c)\big]+\tfrac12 w_d s^2 .
$$

Differentiate once and the drag coefficient falls out, confirming $\phi_b'(s)=\beta_{\rm eff}(s)\,s=\tau_b$:

$$
\beta_{\rm eff}(s):=\frac{\phi_b'(s)}{s}
=\begin{cases}\beta_g\,(s^2+u_{\rm reg})^{\frac{m-1}{2}}+w_d & \text{Weertman}\\[2pt]
\dfrac{\beta_g}{s+u_c}+w_d & \text{Coulomb,}\end{cases}
$$

which are exactly the coefficients coded in `get_tau_bx_jac` ([stress.cu:354, 372](glide/glide/cuda/stress.cu:354)). Now use the **nodal cell speed** (B) of §5, $s_c^2=\tfrac12(u_l^2+u_r^2)+\tfrac12(v_t^2+v_b^2)$, so $\partial s_c^2/\partial u_f=u_f$. Differentiating $\sum_c\phi_b(s_c)A$:

$$
\frac{\partial\phi_b(s_c)}{\partial u_f}
=\phi_b'(s_c)\,\frac{1}{2 s_c}\,\frac{\partial s_c^2}{\partial u_f}
=\big(\beta_{\rm eff}(s_c)\,s_c\big)\frac{u_f}{2 s_c}
=\tfrac12\,\beta_{\rm eff}(s_c)\,u_f .
$$

The facet $u_f$ sits between two cells (left $\ell$, right $c$); summing both and dropping $A$,

$$
\boxed{\;(\tau_{bx})_f=-\tfrac12\big(\beta_{\rm eff}(s_\ell)+\beta_{\rm eff}(s_c)\big)\,u_f\;}
$$

— which is **verbatim** `get_tau_bx_diva_jac` ([stress.cu:607–609](glide/glide/cuda/stress.cu:607)), $\tau_{bx}=-\tfrac12(\beta_{{\rm eff},\ell}+\beta_{{\rm eff},c})\,u_f$. So the *forward SSA drag is the DIVA drag stencil*, fed by a cell-wise $\beta_{\rm eff}(s_c)$.

**What the code actually does instead.** GLIDE's SSA (`get_tau_bx_jac`) does **not** build one $s_c$ per cell. It builds a speed **at each facet** ([stress.cu:350](glide/glide/cuda/stress.cu:350)):

$$
s_f^2=u_f^2+\tfrac14\big(v_{tl}^2+v_{tr}^2+v_{bl}^2+v_{br}^2\big),
\qquad
(\tau_{bx})_f=-\big[\beta_{\rm eff}\,(s_f^2+u_{\rm reg})^{\frac{m-1}{2}}+w_d\big]u_f ,
$$

with $\beta_{\rm eff}=\tfrac12(\beta_\ell\phi_\ell+\beta_r\phi_r)$ the *facet*-averaged coefficient. Each momentum row carries its **own** speed. Test the integrability (mixed-partial) condition directly. From the code ([stress.cu:359](glide/glide/cuda/stress.cu:359)), the $u$-row's coupling to a neighbouring $v_g$ is

$$
\frac{\partial(\tau_{bx})_f}{\partial v_g}
=-\beta_{\rm eff}\,\tfrac12\,c'(s_f)\,u_f\,v_g ,
\qquad c'(s):=\tfrac{d}{ds^2}\big[(s^2{+}u_{\rm reg})^{\frac{m-1}{2}}\big],
$$

evaluated at the $u$-facet speed $s_f$; the transposed entry $\partial(\tau_{by})_g/\partial u_f$ is the same shape but evaluated at the $v$-facet speed $s_g$ with the $v$-facet's own $\beta_{\rm eff}$. Since $s_f\neq s_g$ and the two facet-averaged $\beta_{\rm eff}$ differ, the mixed partials **disagree** — there is no scalar $J_h$ with these as $\partial^2 J_h/\partial u_f\partial v_g$. Hence **GLIDE's SSA is not the gradient of any discrete VP** for $m\neq1$. The lone *structural* exception is $m=1$: then $c'\equiv0$, the cross-coupling vanishes identically, the drag is the diagonal $-\beta u_f$ (trivially $\partial(\tfrac12\beta u^2)$), and the operator is symmetric for **any** state. For every other $m$ the mismatch is nonzero, but its measured size scales with $|m-1|$ **and** with the sub-cell speed variation $s_f-s_g$, so it is state-dependent. The sliding-law sweep (`ssa_sliding_symmetry.py`, and the finer `ssa_msweep.py`) shows exactly this: a clean V with its floor *at* $m=1$ ($9\times10^{-8}$), climbing on the superlinear side ($1.8\times10^{-4}$ at $m=1.5$, $1.2\times10^{-3}$ at $m=3$) but sitting near round-off on the sublinear side of a **smooth** test ($9\times10^{-7}$ at $m=1/3$). That $m=1/3$ reading is not integrability — it is the antisymmetric part merely falling below round-off: roughen the field (raise the $\beta$ contrast) and it re-emerges ($3\times10^{-5}$ at $m=1/3$, and $m=0.5$ crosses into visibly asymmetric). So $m=1$ is the only genuine gradient; every other exponent is non-integrable, with a magnitude the discretization happens to hide on smooth sublinear fields.

**The slight change that fixes it.** Replace the per-facet speed with the per-cell nodal speed $s_c$: compute $\beta_{\rm eff}(s_c)$ **once per cell**, average two cells onto the facet, multiply by the facet's own $u_f$ — i.e. adopt the boxed drag above (the DIVA stencil). Same ingredients ($\beta$, the sliding law, the four facets), merely regrouped from *facet-wise* to *cell-wise*. The result is $\partial J_h/\partial u_f$ of a real discrete VP and is self-adjoint for every $m$ — the same reorganization MOLHO already uses (its cell speed $\tfrac12(u_l^2+u_c^2+v_{tl}^2+v_{bl}^2)$, \[stress.cu MOLHO tree\]).

### 6.4 The basal term $\phi_b$ for DIVA

DIVA keeps §6.1–§6.2 unchanged and only enriches $\phi_b$: the vertical shear is condensed into the column so the basal *speed* $U_b$ differs from the depth-average by the deformational part ($\bar u=U_b+\tau_bF_2$). The column solve ([diva.cu, `diva_coeffs_cell`](glide/glide/cuda/diva.cu:181)) evaluates that same $\phi_b$ at $U_b$ and returns the diagnosed cell coefficient $\beta_{\rm eff}=\text{coeff}/(1+\text{coeff}\cdot F_2)$ ([diva.cu:488](glide/glide/cuda/diva.cu:488)). The momentum row then reads that field and forms the **identical** local drag $-\tfrac12(\beta_{{\rm eff},\ell}+\beta_{{\rm eff},c})u_f$ (`get_tau_bx_diva_jac`). So DIVA's drag *form* was always the forward one (§6.3, choice B).

The only thing that was off was the **speed the closure fed to** $\phi_b$. Stock DIVA formed the cell speed as $U_{\rm bar}^2=\bar u_c^2+\bar v_c^2=\big(\tfrac12(u_l+u_r)\big)^2+\big(\tfrac12(v_t+v_b)\big)^2$ — the square-of-average (A). That makes $\beta_{\rm eff}$ a function of $\bar u_c$, whose gradient $\partial\bar u_c/\partial u_g$ reaches the cell's *other* facet, while the drag's leading factor is the *local* $u_f$. The two no longer come from one $s_c^2$, so the full closure+momentum tangent is asymmetric (below). Switch the closure's eight speed sites to the nodal $s_c^2=\tfrac12(u_l^2+u_r^2)+\tfrac12(v_t^2+v_b^2)$ ([diva.cu:536 ff.](glide/glide/cuda/diva.cu:536)) and $\beta_{\rm eff}$ becomes a function of the same $s_c$ whose derivative is the local $u_f$ — restoring $R=\nabla J_h$.

------------------------------------------------------------------------

## 7. Self-adjointness, for free — and where locality comes from

Because $R=\nabla J_h$, the Jacobian is the Hessian $\nabla^2 J_h$, symmetric by construction. The basal off-diagonal, written from the boxed forward drag with $\partial s_c^2/\partial u_f=u_f$ and $\partial\beta_{\rm eff,c}/\partial u_g=\tfrac{\beta_{\rm eff}'}{2s_c}\,\partial s_c^2/\partial u_g$, is

$$
\frac{\partial^2 J_{\rm b,h}}{\partial u_f\,\partial u_g}
=\frac{\beta_{\rm eff}'(s_c)}{4 s_c}\,\Big(\tfrac{\partial s_c^2}{\partial u_f}\Big)\Big(\tfrac{\partial s_c^2}{\partial u_g}\Big)
=\frac{\beta_{\rm eff}'(s_c)}{4 s_c}\,u_f\,u_g ,
$$

**manifestly symmetric in** $f\leftrightarrow g$ — the same cell scalar $s_c$ read by both rows. That single shared $s_c$ is the whole point: in the broken SSA of §6.3 the $u$-row used $s_f$ and the $v$-row used $s_g$, so this symmetry could not hold. Locality is read off the same expression: $\partial s_c^2/\partial u_f=u_f$ is nonzero only for facets **of that cell**, so the coupling is within-cell. (Had we chosen (A), $\partial s_c^2/\partial u_f=\bar
u_c$ would reach the outer facets — still symmetric, but non-local; that symmetric-but-non-local operator is "DIVA-A".)

------------------------------------------------------------------------

## 8. Forward vs backward, and where stock DIVA and stock SSA sat

Everything above is the **forward** construction: one scalar in, differentiate, symmetric out. GLIDE's two stress-balance drags were built the **backward** way and sat at two different failure points:

-   **Stock SSA** wrote a per-facet speed $s_f$ directly into the drag. The $u$-row and $v$-row reference different speeds, so the mixed partials disagree — not a gradient (for $m\neq1$).
-   **Stock DIVA** wrote the *local* factor $u_f$ (from B) but evaluated $\beta_{\rm eff}$ from the *centroid* speed $\bar u_c$ (from A). Linearizing, $\partial(\tau_b)_f/\partial u_g\propto u_f\,\beta_{\rm eff}'\,\bar
    u_c/2s_c$ against transpose $u_g\,\beta_{\rm eff}'\,\bar u_c/2s_c$ — differ ($u_f$ vs $u_g$), asymmetric.

Both mismatches are impossible in the forward construction, where the velocity factor and the speed-gradient come from the *same* $s_c^2$. The fix in both cases is the same one line of modeling: form one **nodal** cell speed and let $\phi_b$ ride on it.

| balance | how the drag was obtained | cell speed | drag | self-adjoint? | local? |
|------------|------------|------------|------------|------------|------------|
| stock SSA | backward, per-facet speed | $s_f$ (one per facet) | $-[\beta_{\rm eff}c(s_f)]u_f$ | ❌ rows disagree ($m\neq1$) | ✅ |
| stock DIVA | backward, mixed A/B | \(B\) factor, (A) coeff | $-\tfrac12(\beta_\ell+\beta_c)u_f$ | ❌ not a gradient | ✅ |
| forward, choice (A) | differentiate $J_h$ | centroid $\bar u_c$ | $-\tfrac12(\beta_\ell\bar u_\ell+\beta_c\bar u_c)$ | ✅ Hessian | ❌ cross-term |
| **forward, choice (B)** | **differentiate** $J_h$ | **nodal** $s_c$ | $-\tfrac12(\beta_\ell+\beta_c)u_f$ | ✅ **Hessian** | ✅ **local** |

Empirically the fix makes DIVA's momentum-Jacobian symmetric to round-off at **every** sliding exponent — `selfadjoint_test.py` (single floor) gives asymmetry $9\times10^{-8}$ ($m=1$), $1.8\times10^{-7}$ ($m=1/3$), $1.6\times10^{-7}$ ($m=2$), $7\times10^{-8}$ ($m=3$), versus $2.6\times10^{-4}$ before — while the forward keeps converging at every resolution (`gl_mg_convergence.py`, L5→L0 all reduce 4–5 orders, no coarse stall). MOLHO-MI's cell-speed drag is likewise symmetric for all $m$ ($\sim10^{-6}$, `molho_sliding_symmetry.py`), and stock SSA follows the V above (`ssa_sliding_symmetry.py`): the one balance that is a genuine gradient at all $m$ is the one built forward from a single cell speed.

**Moral.** Build a stress balance forward — discretize the dissipation potential, then differentiate — and self-adjointness is free; the remaining freedom is the quadrature, which you choose for locality. GLIDE's viscous and driving terms were already forward (hence symmetric); only the two drags were written backward. Rebuilding them forward with the nodal (sum-of-squares) cell speed makes **both** SSA and DIVA genuine gradients of a discrete VP — self-adjoint, local, at stock cost, and discretized exactly like MOLHO.