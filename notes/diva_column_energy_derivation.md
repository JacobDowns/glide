# DIVA as variational static condensation of a local column problem

This note rewrites the "column energy" view of DIVA in a more explicit way. The goal is not to change the model, but to show carefully that:

1. Goldberg's DIVA approximation still comes from a scalar functional.
2. After that approximation, the deformational part of the velocity is local to each column.
3. Eliminating that local field by minimization produces a reduced functional in the membrane velocity alone.
4. The DIVA residual is the gradient of that reduced functional, and the exact condensed tangent is its Hessian.

The derivation follows Goldberg (2011), especially Eq. (12) for the approximated action and Eqs. (31)-(41) for the vertical closure, but it reorganizes the logic around a local constrained minimization.

Companion notes:

- [variational_condensation_primer.md](/home/bizon/glaciers/notes/variational_condensation_primer.md)
- [stress_balance_variational_structure.md](/home/bizon/glaciers/notes/stress_balance_variational_structure.md)

Primary sources:

- [Goldberg 2011 PDF](/home/bizon/glaciers/papers/a-variationally-derived-depth-integrated-approximation-to-a-higher-order-glaciological-flow-model.pdf)
- [MOLHO paper PDF](/home/bizon/glaciers/papers/tc-16-179-2022.pdf)

## 1. Starting point: the approximated Goldberg functional

Let

$$
x = \dot\varepsilon_e^2.
$$

For Glen flow we write

$$
\eta(x) = \frac12 B x^p,
\qquad
p = \frac{1-n}{2n}.
$$

The scalar potential whose first variation gives internal virtual work is

$$
\varphi(x) = \frac{B}{p+1} x^{p+1}
= \frac{2n}{n+1} B x^{\frac{n+1}{2n}},
$$

so that

$$
\varphi'(x) = B x^p = 2\eta(x).
$$

This is the integrand that appears in Goldberg's action. It is worth fixing this normalization at the start, because in code it is common to carry instead

$$
\Psi(x) = \frac12 \varphi(x),
\qquad
\Psi'(x)=\eta(x),
$$

and that factor of 2 matters when differentiating the energy.

Let $\Omega$ be the three-dimensional ice domain, $\Omega_p$ its horizontal projection, and $\mathbf u=(u,v)$ the horizontal velocity. Goldberg's approximated action (his Eq. (12)) is

$$
\begin{aligned}
\mathcal A_{\rm G}[\mathbf u]
= {}& \int_\Omega \varphi\!\left(
u_x^2+v_y^2+u_xv_y
+\frac14(u_y+v_x)^2
+\frac14(u_z^2+v_z^2)
\right)\,d\Omega \\
&+\int_\Omega \rho g\,\mathbf u\cdot\nabla s\,d\Omega
+\int_{\Gamma_b}F_{\rm basal}(|\mathbf u_b|)\,d\Gamma
+I_{\rm lat}[\bar{\mathbf u}].
\end{aligned}
$$

Here $\mathbf u_b=\mathbf u|_{z=b}$ and

$$
\nabla_{\mathbf u_b}F_{\rm basal}(|\mathbf u_b|)
=\mathbf t_b^\Gamma
=f(U_b)\frac{\mathbf u_b}{U_b},
\qquad U_b=|\mathbf u_b|.
$$

Thus $f(U_b)$ is the magnitude of basal drag per unit basal area. On a graph bed, $d\Gamma=\gamma_b\,dA$ with $\gamma_b=\sqrt{1+b_x^2+b_y^2}$. Since the condensed action is integrated over projected area, its basal term is $\gamma_bF_{\rm basal}$ and its traction is

$$
\mathbf t_b=\gamma_b\mathbf t_b^\Gamma
=\gamma_b f(U_b)\frac{\mathbf u_b}{U_b}.
$$

For a flat bed, $\gamma_b=1$. To avoid carrying this geometric factor through every equation, $F_{\rm basal}$, $f$, and $\mathbf t_b$ below mean these projected-area versions. This convention reproduces the slope factor in Goldberg's basal condition. $I_{\rm lat}$ is Goldberg's depth-independent lateral stress-boundary term. It is irrelevant to an interior column, but remains part of the full horizontal action.

Goldberg's DIVA approximation replaces the depth-varying velocity in the four horizontal strain-rate terms by its depth average, while retaining vertical shear in the invariant. Define

$$
\bar{\mathbf u}(x,y)=\frac1H\int_b^s\mathbf u(x,y,z)\,dz,
\qquad H=s-b,
$$

and the membrane invariant

$$
q(\bar{\mathbf u})
=\bar u_x^2+\bar v_y^2+\bar u_x\bar v_y
+\frac14(\bar u_y+\bar v_x)^2.
$$

Then the invariant in a fixed column is

$$
x(z)=q(\bar{\mathbf u})+\frac14|\partial_z\mathbf u(z)|^2.
$$

The scalar $q$ is therefore not an extra column unknown: it is supplied by the retained horizontal field and is constant in $z$ in that column.

The gravitational term is not discarded. Depth integration gives exactly

$$
\int_\Omega \rho g\,\mathbf u\cdot\nabla s\,d\Omega
=\int_{\Omega_p}\rho gH\,\bar{\mathbf u}\cdot\nabla s\,dA.
$$

It depends only on the prescribed mean $\bar{\mathbf u}$, not on the shape of $\mathbf u(z)$ about that mean. It therefore has no effect on the inner profile minimization, but is retained in the reduced action.

## 2. The exact condensation statement

Once the membrane strains are evaluated from $\bar{\mathbf u}$ alone, the quantity $q$ is constant in $z$ within a given column. The column functional

$$
J_{\rm col}[\mathbf u;q]
= \int_b^s \varphi\!\left(q+\frac14|\mathbf u_z|^2\right)\,dz
+ F_{\rm basal}(|\mathbf u_b|)
$$

contains:

- no horizontal derivatives of the profile $\mathbf u(z)$;
- no coupling to neighboring columns except through the supplied values of $q$ and $\bar{\mathbf u}$.

So at fixed $(q,\bar{\mathbf u})$ the vertical profile is determined by a one-column minimization problem. Define its minimized value by

$$
\Phi(q,\bar{\mathbf u})
=\min_{\mathbf u(\cdot)}
\left\{
J_{\rm col}[\mathbf u;q]
\ \middle|\
\frac1H\int_b^s\mathbf u\,dz=\bar{\mathbf u}
\right\}.
$$

The original DIVA minimization is equivalently the reduced two-dimensional problem

$$
\boxed{
\mathcal A_{\rm red}[\bar{\mathbf u}]
=\int_{\Omega_p}
\left[
\Phi\big(q(\bar{\mathbf u}),\bar{\mathbf u}\big)
+\rho gH\,\bar{\mathbf u}\cdot\nabla s
\right]dA
+I_{\rm lat}[\bar{\mathbf u}].
}
$$

This is static condensation: $\bar{\mathbf u}$ is retained as the global unknown, and the full vertical field $\mathbf u(x,y,z)$ is eliminated by independent local minimizations. No term of Goldberg's action has been removed; the driving and lateral terms simply bypass the local minimization because they depend only on retained variables.

## 3. The local constrained minimization problem

Fix one column and the quantities supplied by the horizontal solve:

$$
q = q(\bar{\mathbf u}),
\qquad
\bar{\mathbf u} = \frac1H \int_b^s \mathbf u(z)\,dz,
\qquad
H=s-b.
$$

Define the condensed column energy

$$
\Phi(q,\bar{\mathbf u})
= \min_{\mathbf u(\cdot)}
\left\{
\int_b^s \varphi\!\left(q+\frac14|\mathbf u_z|^2\right)\,dz
+ F_{\rm basal}(|\mathbf u_b|)
\ \middle|\
\frac1H\int_b^s \mathbf u\,dz=\bar{\mathbf u}
\right\}.
$$

This is the precise version of "minimize a local column problem for the deformational velocity and keep only a reduced functional over the membrane velocity."

The mean constraint is the device that ties the local profile to the global unknown $\bar{\mathbf u}$. The gravitational and lateral terms are absent from this displayed minimization only because they are constant over the feasible profiles with this same mean.

## 4. Lagrange multiplier and first variation

The mean constraint has two components, so its multiplier is a horizontal vector $\vec\lambda$. The local Lagrangian is

$$
\mathcal L[\mathbf u,\vec\lambda;q,\bar{\mathbf u}]
=\int_b^s\varphi\!\left(q+\frac14|\mathbf u_z|^2\right)dz
+F_{\rm basal}(|\mathbf u_b|)
-\vec\lambda\cdot\left(
\frac1H\int_b^s\mathbf u\,dz-\bar{\mathbf u}
\right).
$$

Let $x(z)=q+\tfrac14|\mathbf u_z|^2$. For a vector variation $\delta\mathbf u$,

$$
\delta x=\frac12\mathbf u_z\cdot\delta\mathbf u_z,
\qquad
\delta\!\int_b^s\varphi(x)dz
=\int_b^s\eta\mathbf u_z\cdot\delta\mathbf u_z\,dz.
$$

The last equality uses $\varphi'(x)=2\eta(x)$. Define the horizontal shear-traction vector on a horizontal plane by

$$
\vec\tau(z)=\eta(z)\mathbf u_z(z).
$$

The variations of the remaining terms are

$$
\delta F_{\rm basal}=\mathbf t_b\cdot\delta\mathbf u_b,
\qquad
\delta\mathcal L_{\rm constraint}
=-\frac1H\int_b^s\vec\lambda\cdot\delta\mathbf u\,dz.
$$

Thus

$$
\delta\mathcal L
=\int_b^s\vec\tau\cdot\delta\mathbf u_z\,dz
+\mathbf t_b\cdot\delta\mathbf u_b
-\frac1H\int_b^s\vec\lambda\cdot\delta\mathbf u\,dz.
$$

Integrating the first term by parts, without yet applying any boundary condition, gives

$$
\delta\mathcal L
=\vec\tau(s)\cdot\delta\mathbf u_s
+\big[\mathbf t_b-\vec\tau(b)\big]\cdot\delta\mathbf u_b
-\int_b^s\left(\partial_z\vec\tau+
\frac{\vec\lambda}{H}\right)\cdot\delta\mathbf u\,dz.
$$

## 5. Why stationarity gives separate interior and boundary conditions

This is a standard, but often compressed, step in the calculus of variations. For a sliding bed, the basal velocity is not prescribed, so admissible variations may have arbitrary values at $z=b$. Likewise, the stress-free surface does not prescribe $\mathbf u_s$. We may choose three kinds of variation independently:

- a smooth variation supported strictly inside $(b,s)$, with zero traces at both ends;
- a variation with an arbitrary trace at the upper surface;
- a variation with an arbitrary trace at the bed.

For the first class, both boundary terms vanish. Since the remaining interior function $\delta\mathbf u(z)$ is arbitrary, the fundamental lemma of the calculus of variations gives

$$
\partial_z\vec\tau=-\frac{\vec\lambda}{H}
\qquad (b<z<s).
$$

After this interior equation has been imposed, the remaining endpoint traces can be chosen independently. Stationarity for every upper-surface trace and every basal trace then gives

$$
\vec\tau(s)=\mathbf0,
\qquad
\vec\tau(b)=\mathbf t_b.
$$

The first is the stress-free surface condition. The second is the natural sliding boundary condition. If the bed is frozen, $\mathbf u_b=\mathbf0$ is an essential boundary condition instead, so $\delta\mathbf u_b=\mathbf0$ and this sliding-boundary argument is not used there.

## 6. Euler-Lagrange profile and multiplier meaning

Because $\vec\lambda$ is constant within the column, the interior equation and $\vec\tau(s)=0$ give

$$
\vec\tau(z)
=\vec\lambda\frac{s-z}{H}
=\vec\lambda\zeta,
\qquad
\zeta=\frac{s-z}{H}.
$$

At the bed, $\zeta=1$, so

$$
\vec\lambda
=\vec\tau(b)
=\mathbf t_b.
$$

Thus the multiplier is not an auxiliary artifact: it is exactly the basal traction vector. The linear vertical shear-stress profile used by DIVA is the Euler--Lagrange equation of the constrained local minimization, not an independent ansatz.

## 7. Recover the DIVA closure equation

The profile equation is vector-valued:

$$
\mathbf u_z(z)=\frac{\mathbf t_b\zeta}{\eta(z)}.
$$

Integrating from the bed and then averaging gives

$$
\bar{\mathbf u}-\mathbf u_b
=\mathbf t_b H\int_0^1\frac{\zeta^2}{\eta(\zeta)}\,d\zeta.
$$

Define

$$
F_2(q,T_b)=H\int_0^1\frac{\zeta^2}{\eta(\zeta;q,T_b)}\,d\zeta,
\qquad T_b=|\mathbf t_b|.
$$

The vector closure is therefore

$$
\boxed{
\bar{\mathbf u}=\mathbf u_b+F_2(q,T_b)\,\mathbf t_b,
\qquad
\mathbf t_b=f(U_b)\frac{\mathbf u_b}{U_b}.
}
$$

Because the sliding law is isotropic, $\mathbf u_b$, $\mathbf t_b$, and $\bar{\mathbf u}$ are collinear. For $\bar U=|\bar{\mathbf u}|>0$, write their common direction as $\mathbf e=\bar{\mathbf u}/\bar U$. Then $\mathbf u_b=U_b\mathbf e$ and $\mathbf t_b=T_b\mathbf e$, reducing the vector equation to Goldberg's scalar speed closure

$$
\boxed{
\bar U=U_b+f(U_b)F_2\big(q,f(U_b)\big).
}
$$

This scalar reduction is a consequence of isotropic basal drag, not a suppression of the $v$ component. The viscosity entering $F_2$ is determined pointwise by

$$
\eta(\zeta)
=\frac12B\left(q+\frac{T_b^2\zeta^2}{4\eta(\zeta)^2}\right)^p.
$$

Thus the vertical field has been condensed to a local nonlinear solve for $U_b$ (or equivalently $T_b$), after which the full vector profile is recovered along $\mathbf e$.

## 8. The reduced functional and its first derivatives

By definition,

$$
\Phi(q,\bar{\mathbf u})
= \mathcal L[\mathbf u^*,\vec\lambda^*;q,\bar{\mathbf u}],
$$

where $\mathbf u^*$ and $\vec\lambda^*$ solve the stationarity system above.

Now apply the envelope theorem: when differentiating the minimized value $\Phi$, the derivatives through the optimizer $(\mathbf u^*,\vec\lambda^*)$ drop out, because the first variations with respect to both local variables vanish at the optimum.

### 8.1 Derivative with respect to the mean velocity

The only explicit $\bar{\mathbf u}$ dependence in $\mathcal L$ is the term $+\vec\lambda\cdot\bar{\mathbf u}$, so

$$
\nabla_{\bar{\mathbf u}}\Phi
= \nabla_{\bar{\mathbf u}}\mathcal L
=\vec\lambda
=\mathbf t_b.
$$

That is the vector drag term in the reduced stress balance.

### 8.2 Derivative with respect to the membrane invariant

The quantity $q$ appears only inside the viscous integrand, so

$$
\frac{\partial\Phi}{\partial q}
= \int_b^s \varphi'(x)\,\frac{\partial x}{\partial q}\,dz
= \int_b^s 2\eta\,dz
= 2H\bar\eta,
$$

where

$$
\bar\eta = \frac1H\int_b^s \eta(z)\,dz.
$$

So the gradient of the condensed column functional is

$$
\boxed{
\nabla\Phi(q,\bar{\mathbf u})
= \left(2H\bar\eta,\ \mathbf t_b\right).
}
$$

This is exactly the pair of coefficients that appear in DIVA:

- membrane coefficient $2H\bar\eta$;
- drag vector $\mathbf t_b$, or equivalently $\beta_{\rm eff}\bar{\mathbf u}$ when that representation is defined.

## 9. From the condensed column energy to the DIVA residual

For a horizontal test field $\delta\bar{\mathbf u}$, differentiate the complete reduced action from Section 2:

$$
\begin{aligned}
\delta\mathcal A_{\rm red}
=\int_{\Omega_p}\big[&
2H\bar\eta\,\delta q(\bar{\mathbf u};\delta\bar{\mathbf u})
+\mathbf t_b\cdot\delta\bar{\mathbf u}\\
&+\rho gH\,\nabla s\cdot\delta\bar{\mathbf u}
\big]dA
+\delta I_{\rm lat}[\bar{\mathbf u};\delta\bar{\mathbf u}].
\end{aligned}
$$

This is the DIVA weak residual. The first term is membrane virtual work; integrating its horizontal derivatives by parts produces the usual SSA-shaped membrane operator. The second is basal drag. The third is the driving term, which was retained throughout; its sign in a strong-form residual follows from the selected residual convention.

## 10. The reduced Hessian is a Schur complement

The exact condensed tangent is obtained by differentiating the reduced functional a second time.

It is best to write this once in general form. Let

$$
a=(q,\bar{\mathbf u})
$$

be the retained variables and let $w$ denote the eliminated column variables. Depending on taste, $w$ can mean:

- the full profile $\mathbf u(z)$ together with the multiplier $\vec\lambda$, or
- a finite-dimensional representation of the profile, or
- after isotropic reduction, the scalar basal speed $U_b$.

Write $\widehat J(a,w)$ for the local KKT Lagrangian, with $w$ containing both the profile and its mean-constraint multiplier. Its stationary point $w^*(a)$ is defined by

$$
\partial_w \widehat J(a,w^*(a))=0.
$$

Then

$$
\Phi(a)=\widehat J(a,w^*(a)).
$$

The first derivative is the envelope result

$$
\nabla_a\Phi = \partial_a \widehat J.
$$

Differentiating again and using the implicit-function theorem,

$$
\partial_a w^*
= -\left(\partial_{ww}^2\widehat J\right)^{-1}\partial_{wa}^2\widehat J,
$$

gives

$$
\boxed{
\nabla_{aa}^2\Phi
= \partial_{aa}^2\widehat J
- \partial_{aw}^2\widehat J
\left(\partial_{ww}^2\widehat J\right)^{-1}
\partial_{wa}^2\widehat J.
}
$$

This is exactly the Schur complement of the eliminated column block.

Because all blocks come from second derivatives of one scalar $\widehat J$, the mixed partials satisfy

$$
\partial_{aw}^2\widehat J
= \left(\partial_{wa}^2\widehat J\right)^T,
$$

so the condensed Hessian is symmetric.

That is the clean variational statement behind the desired result:

- the DIVA residual is the gradient of the condensed column energy;
- the exact DIVA tangent is the Hessian of that condensed energy;
- therefore the exact condensed tangent is symmetric.

## 11. What standard DIVA keeps and what it usually drops

The forward DIVA solve uses the correct reduced first derivative:

- solve the local closure to obtain $\mathbf t_b$ and $\bar\eta$,
- assemble the membrane and drag residual from those coefficients.

The subtle part is the second derivative. A frozen-coefficient linearization keeps only the direct derivative of the residual with respect to the membrane field while holding the diagnosed column coefficients fixed. The exact variational tangent also includes the response of the minimizing column state to $(q,\bar{\mathbf u})$, which is precisely the Schur-complement term above.

So if one wants a residual and Hessian that truly come from the condensed variational principle, one must differentiate through the local minimizer, not only through the frozen SSA-shaped operator.

## 12. Bottom line

The DIVA approximation can be viewed as:

1. start from Goldberg's approximated action;
2. for each column, minimize the viscous-plus-basal energy over vertical profiles with prescribed mean velocity;
3. define the minimized value as a condensed column functional $\Phi(q,\bar{\mathbf u})$;
4. assemble the horizontal stress balance from the gradient of $\Phi$;
5. assemble the exact condensed tangent from the Hessian of $\Phi$.

The central formulas are:

$$
\Phi(q,\bar{\mathbf u})
= \min_{\mathbf u(\cdot)}
\left\{
\int_b^s \varphi\!\left(q+\frac14|\mathbf u_z|^2\right)\,dz
+ F_{\rm basal}(|\mathbf u_b|)
\ \middle|\
\frac1H\int_b^s\mathbf u\,dz=\bar{\mathbf u}
\right\},
$$

$$
\vec\tau(z)=\mathbf t_b\zeta,
\qquad
\bar{\mathbf u}-\mathbf u_b=F_2\mathbf t_b,
\qquad
\bar U-U_b-f(U_b)F_2=0,
$$

$$
\nabla\Phi = (2H\bar\eta,\mathbf t_b),
$$

$$
\nabla^2\Phi
= \partial_{aa}^2\widehat J
- \partial_{aw}^2\widehat J
\left(\partial_{ww}^2\widehat J\right)^{-1}
\partial_{wa}^2\widehat J.
$$

That is the variational static-condensation picture of DIVA.
