# The DIVA stress balance in GLIDE

The depth-integrated viscosity approximation (DIVA) is a two-dimensional
momentum balance for depth-averaged horizontal velocity, coupled to a
one-dimensional vertical closure in every horizontal cell. It retains the
membrane stresses of the shallow-shelf approximation (SSA), but restores the
vertical shear that controls deformation in slow grounded ice. In this way one
formulation covers both the shear-dominated and membrane-dominated flow regimes
without introducing a globally coupled three-dimensional velocity unknown.

This note develops that statement in the same order as the physics: first the
depth-integrated balance, then the vertical closure, then the basal closure that
makes the system two-dimensional. Only after those relationships are defined do
we describe their discretization and their place in GLIDE's nonlinear solver
and adjoint. The presentation is informed by the
[Yelmo DIVA overview](https://fesmc.github.io/yelmo/physics/momentum/diva.html),
while the notation and implementation details here are specific to GLIDE.

The notation follows Arthern et al. (2015), particularly for the shear integrals
$\mathcal I_1$ and $\mathcal I_2$. Goldberg (2011) is the primary derivation
of the variational, depth-integrated approximation.

- Goldberg (2011), *A variationally derived, depth-integrated approximation to a
  higher-order glaciological flow model*. Local
  [PDF](../../docs/a-variationally-derived-depth-integrated-approximation-to-a-higher-order-glaciological-flow-model.pdf),
  [DOI](https://doi.org/10.3189/002214311795306763).
- Arthern et al. (2015), *Flow speed within the Antarctic ice sheet and its
  controls inferred from satellite observations*. Local
  [PDF](<../../docs/JGR Earth Surface - 2015 - Arthern - Flow speed within the Antarctic ice sheet and its controls inferred from satellite.pdf>),
  [DOI](https://doi.org/10.1002/2014JF003239).

References such as G31 and A7 denote equations in Goldberg and Arthern. See
[diva.md](diva.md) for the original design and [diva_history.md](diva_history.md)
for the development record. For a detailed, code-oriented trace of the DIVA
linearization and VJP, see [diva_adjoint.md](diva_adjoint.md).

## 1. What DIVA approximates

SSA assumes that horizontal velocity is independent of depth. That assumption
is appropriate for ice shelves and fast streams, where membrane stresses
dominate, but it omits the vertical shear responsible for most deformation in
the slow grounded interior. A first-order or Blatter--Pattyn model retains both
effects by solving for horizontal velocity throughout the ice thickness, at the
cost of a three-dimensional nonlinear system.

DIVA occupies the useful middle ground. It makes the horizontal strain rates
independent of depth, as in SSA, but retains the vertical shear rates in Glen's
flow law. The viscosity can therefore vary through the column even though the
globally coupled unknown is only the depth-averaged horizontal velocity

$$
\bar{\mathbf u}=\frac{1}{H}\int_b^s\mathbf u(z)\,dz
               =(\bar u,\bar v).
$$

The result can be viewed as an SSA-shaped membrane balance supplied with a
local higher-order closure. Relative to SSA, the momentum solve sees two
modified coefficients:

1. the membrane viscosity is the depth average $\bar\eta$ of the
   depth-varying viscosity $\eta(z)$; and
2. the basal sliding coefficient is replaced by an effective drag
   $\beta_{\mathrm{eff}}$ relating basal traction to $\bar{\mathbf u}$ rather
   than to the basal velocity $\mathbf u_b$.

Those coefficients are not prescribed material fields. They depend on the
current velocity and are recomputed by a local column calculation as the
nonlinear momentum solve proceeds.

> DIVA keeps the global unknown and stencil two-dimensional. Vertical shear
> enters through column-local constitutive calculations, not through a global
> three-dimensional solve.

## 2. Continuum equations

### 2.1 Geometry and conventions

The vertical coordinate $z$ increases upward. The bed is $b$, the surface is
$s$, thickness is $H=s-b$, and scaled depth is

$$
\zeta=\frac{s-z}{H},
\qquad \zeta=0\text{ at the surface},
\qquad \zeta=1\text{ at the bed}.
$$

The depth-averaged, basal, and surface velocities are
$\bar{\mathbf u}$, $\mathbf u_b$, and $\mathbf u_s$, with speeds
$\bar U$, $U_b$, and $U_s$. The column closure is scalar because the
current rheology and sliding laws are isotropic.

Strain rates use the tensor convention

$$
\dot\varepsilon_{xy}=\tfrac12(\partial_y\bar u+\partial_x\bar v),
\qquad
\dot\varepsilon_{xz}=\tfrac12\partial_z u,
\qquad
\dot\varepsilon_{yz}=\tfrac12\partial_z v.
$$

The Glen viscosity is

$$
\eta=\tfrac12B(\dot\varepsilon_e^2)^p,
\qquad B=A^{-1/n},
\qquad p=\frac{1-n}{2n}.
$$

In closure equations, $\tau_b\ge0$ is the magnitude of basal resistance. We
write $\vec{\tau}_b$ for the drag vector aligned with velocity, so the traction
on the ice is $-\vec{\tau}_b$ and the momentum equations contain a negative
drag term.

### 2.2 Depth-integrated momentum

DIVA replaces $u,v$ by their depth averages in horizontal strain rates while
retaining $\partial_z u,\partial_z v$ in viscosity and the basal condition.
Goldberg applies this approximation to the action functional. The resulting
equations and boundary conditions are consistent, and the frozen-viscosity
operator is self-adjoint. The full nonlinear Jacobian need not be assumed
symmetric.

The $x$-momentum equation is

$$
\begin{aligned}
&\partial_x[2\bar\eta H(2\partial_x\bar u+\partial_y\bar v)]
+\partial_y[\bar\eta H(\partial_y\bar u+\partial_x\bar v)]
-\tau_{b,x}\\
&\hspace{5cm}=\rho gH\partial_xs,
\end{aligned}
$$

together with the basal-drag relation

$$
\vec{\tau}_b=\beta_{\mathrm{eff}}\bar{\mathbf u}.
$$

Here

$$
\bar\eta=\frac{1}{H}\int_b^s\eta(z)\,dz.
$$

This is the same differential operator used for SSA, but $\bar\eta$ and
$\beta_{\mathrm{eff}}$ now contain the effects of vertical shear. The
right-hand side is the depth-integrated driving stress. The corresponding
$y$-momentum equation is obtained by interchanging $x$ and $y$ (G43--G44; A1).

GLIDE omits Goldberg's geometric bed-slope multiplier, consistent with
Arthern's small-slope Robin condition. This does not affect flat-bed ISMIP-HOM
experiment C but should be quantified for experiments A and B.

### 2.3 Effective viscosity and vertical shear

The strain-rate invariant is

$$
\dot\varepsilon_e^2
=\underbrace{\dot\varepsilon_{xx}^2+\dot\varepsilon_{yy}^2
 +\dot\varepsilon_{xx}\dot\varepsilon_{yy}+\dot\varepsilon_{xy}^2}
 _{\dot\varepsilon_{\mathrm{mem}}^2}
+\dot\varepsilon_{xz}^2+\dot\varepsilon_{yz}^2+\varepsilon_0^2.
$$

The membrane part comes from $\bar{\mathbf u}$ and is depth-independent.
`membrane_eps_sq<T>` in `viscosity.cu` is the shared unregularized
definition. The vertical shear terms are what distinguish DIVA
from SSA: they alter the deformation rate and hence the viscosity in
shear-dominated columns.

Integrating the shear stress from the stress-free surface gives (G31; A4--A5)

$$
\tau_{xz}=\tau_{b,x}\zeta,
\qquad \tau_{yz}=\tau_{b,y}\zeta,
$$

and therefore

$$
\partial_zu=\frac{\tau_{b,x}\zeta}{\eta},
\qquad
\partial_zv=\frac{\tau_{b,y}\zeta}{\eta}.
$$

Substitution into Glen's law gives the implicit level equation

$$
\boxed{
\eta=\tfrac12B\left[A_{\mathrm{reg}}
 +\frac{\tau_b^2\zeta^2}{4\eta^2}\right]^p},
\qquad
A_{\mathrm{reg}}=\dot\varepsilon_{\mathrm{mem}}^2+\varepsilon_0^2.
$$

Viscosity occurs on both sides because vertical strain rate is stress divided
by viscosity. Thus, given the membrane strain rate and basal traction, each
height in the column requires a scalar constitutive solve. Arthern obtains a
cubic for $n=3$; GLIDE uses Newton's method for general $n$.

### 2.4 From vertical shear to the velocity profile

Following Arthern (A7), define the shear integrals

$$
\mathcal I_\alpha
=\int_b^s\frac{1}{\eta(z)}
 \left(\frac{s-z}{H}\right)^\alpha dz.
$$

Integrating $\partial_z\mathbf u=\vec{\tau}_b\zeta/\eta$ upward from the bed
gives the velocity at any height:

$$
\mathbf u(z)=\mathbf u_b
+\vec{\tau}_b\int_b^z\frac{1}{\eta(z')}
 \left(\frac{s-z'}{H}\right)dz'.
$$

Two special cases are all that the depth-integrated system needs. At the
surface, and after averaging through the column, respectively,

$$
\boxed{\mathbf u_s=\mathbf u_b+\vec{\tau}_b\mathcal I_1},
\qquad
\boxed{\bar{\mathbf u}=\mathbf u_b+\vec{\tau}_b\mathcal I_2}.
$$

The $\mathcal I_1$ relation connects the model state to surface observations.
The $\mathcal I_2$ relation connects the basal and depth-averaged velocities
and is therefore the one needed to close the momentum balance. Both follow from
the vertical-shear closure and are independent of the choice of sliding law.

### 2.5 Basal stress closure and effective drag

The momentum equation is solved for $\bar{\mathbf u}$, but a basal friction law
relates traction to the velocity at the bed, $\mathbf u_b$. The purpose of the
basal closure is therefore to answer a local question: **given the
depth-averaged speed $\bar U$, what basal speed $U_b$ and basal traction
$\tau_b$ are consistent with the vertical shear in this column?**

Let

$$
U_b=\lVert\mathbf u_b\rVert,
\qquad
\tau_b=\lVert\vec{\tau}_b\rVert.
$$

Recall that $\vec{\tau}_b$ denotes the positive drag vector, aligned with the
basal velocity; the physical traction acting on the ice is
$-\vec{\tau}_b$. An isotropic sliding law can then be written in equivalent
vector and scalar forms:

$$
\vec{\tau}_b=c(U_b)\mathbf u_b,
\qquad
\tau_b=f(U_b)=c(U_b)U_b.
$$

The scalar equation follows by taking the magnitude of the vector equation.
For $U_b>0$, both vectors point along
$\widehat{\mathbf t}=\mathbf u_b/U_b$, so
$\mathbf u_b=U_b\widehat{\mathbf t}$ and
$\vec{\tau}_b=\tau_b\widehat{\mathbf t}$. At $U_b=0$, the direction is
undefined but both vectors vanish and the scalar relation remains well
defined.

GLIDE supports

$$
\begin{array}{ll}
\text{Weertman:}&c(U)=\beta\varphi
(U^2+u_{\mathrm{reg}})^{(m_s-1)/2}+w_d,\\[3pt]
\text{regularized Coulomb:}&c(U)=\dfrac{\beta\varphi}
{\sqrt{U^2+u_{\mathrm{reg}}}+u_c}+w_d.
\end{array}
$$

Grounding is included here through $\beta\varphi$ and must not be applied
again in the momentum stencil.

Because both the rheology and the sliding law are isotropic,
$\mathbf u_b$, $\vec{\tau}_b$, and $\bar{\mathbf u}$ are parallel. Taking
magnitudes in the depth-average relation from Section 2.4 gives

$$
\bar U=U_b+\tau_b\mathcal I_2(\tau_b).
$$

Substituting the sliding law $\tau_b=f(U_b)$ leaves one unknown, $U_b$:

$$
\boxed{R(U_b)=U_b+f(U_b)\mathcal I_2(f(U_b))-\bar U=0}.
$$

This is the basal-speed closure solved in each cell. The dependence
$\mathcal I_2(\tau_b)$ is essential: changing $U_b$ changes traction, which
changes vertical shear and viscosity, which in turn changes $\mathcal I_2$.
Even a linear sliding law therefore produces a nonlinear closure.

The root lies in $[0,\bar U]$, and for a monotone law

$$
R'=1+f'\mathcal I_2+ff'\frac{d\mathcal I_2}{d\tau_b}\ge1.
$$

It is therefore unique and well conditioned. Both bounds are provable, and they come from
different places, which is what licenses the clamped warm start and bisection as an
independent reference.

*Upper bound, from the kinematics alone.* $\mathcal I_2>0$ strictly, since $\eta$ is
finite and positive at every depth, and any physical law has $\tau_b\ge0$ for
$U_b\ge0$. So

$$\bar U-U_b=\tau_b\mathcal I_2\ \ge\ 0
\qquad\Longrightarrow\qquad U_b\le\bar U,$$

with equality iff $\tau_b\mathcal I_2=0$: deformation can only add to sliding, never
subtract. This uses no property of $f$ beyond its sign, so it holds for a learned sliding
law as much as for Weertman.

*Lower bound, from the sign of $R$ at the endpoints.*

$$R(0)=f(0)\mathcal I_2-\bar U=-\bar U\le0,
\qquad R(\bar U)=f(\bar U)\mathcal I_2\ge0,$$

using $f(0)=0$, which holds for both laws because $f=c(U)U$ carries an explicit factor of
$U$ and $c(0)$ is finite — which is what $u_{\mathrm{reg}}$ is for, since $c\sim U^{m_s-1}$
would otherwise diverge at $U=0$ for $m_s<1$. So $R$ changes sign on $[0,\bar U]$ and,
being strictly increasing, does so exactly once.

The bracket also gives the natural diagnostic, the sliding fraction
$U_b/\bar U=1-\tau_b\mathcal I_2/\bar U\in[0,1]$: 1 for plug flow, 0 for a frozen bed. Once $U_b$ is known, the column
also supplies $\tau_b=f(U_b)$. Momentum needs the corresponding drag vector as
a function of its own unknown $\bar{\mathbf u}$. Since the two vectors are
parallel,

$$
\vec{\tau}_b=\frac{\tau_b}{\bar U}\bar{\mathbf u}
             =\beta_{\mathrm{eff}}\bar{\mathbf u}.
$$

Using $\bar U=U_b[1+c(U_b)\mathcal I_2]$ and
$\tau_b=c(U_b)U_b$ gives

$$
\boxed{
\tau_b=\beta_{\mathrm{eff}}\bar U,
\qquad
\beta_{\mathrm{eff}}=\frac{c(U_b)}{1+c(U_b)\mathcal I_2}.
$$

This algebraic expression, rather than an explicit numerical division by
$\bar U$, is used at rest. The coefficient $\beta_{\mathrm{eff}}$ is the
traction-to-velocity ratio required to insert the column response into the
two-dimensional momentum equation.

### 2.6 Limiting cases and analytic checks

The equations recover the expected end members:

- **SSA limit:** when vertical shear becomes negligible,
  $\mathbf u_b$, $\bar{\mathbf u}$, and $\mathbf u_s$ coincide and the
  depth-integrated balance reduces to SSA.
- **SIA limit:** when membrane-stress gradients become negligible, basal drag
  balances the driving stress and the vertical closure reproduces the
  shear-dominated velocity profile.
- **Frozen-bed limit:** as $U_b\to0$,
  $\beta_{\mathrm{eff}}\to1/\mathcal I_2$.

For a uniform slab, the shear integrals reduce to

$$
\mathcal I_1=\frac{2AH\tau_b^{n-1}}{n+1},
\qquad
\mathcal I_2=\frac{2AH\tau_b^{n-1}}{n+2}.
$$

These expressions are useful checks on both the normalization and the vertical
quadrature.

## 3. From the equations to GLIDE

### 3.1 The local column calculation

The preceding equations now define the computational dependency that each
nonlinear residual evaluation must follow:

$$
\bar{\mathbf u}
\longrightarrow(\dot\varepsilon_{\mathrm{mem}}^2,\bar U)
\longrightarrow U_b
\longrightarrow(\bar\eta,\mathcal I_1,\mathcal I_2,
                 \beta_{\mathrm{eff}},U_s).
$$

The first arrow uses horizontal finite differences to compute membrane strain
rates. The remaining work is cell-local: a short vertical quadrature, scalar
viscosity solves at its nodes, and a scalar basal-speed solve. GLIDE therefore
does not solve for or store a three-dimensional velocity field. It stores the
column integrals and selected diagnostics needed by the two-dimensional
momentum equation and inversion objective.

In code, the notation maps as follows:

| Quantity | Meaning | Code |
|---|---|---|
| $\bar\eta=H^{-1}\int_b^s\eta\,dz$ | depth-averaged viscosity | `eta_bar` |
| $\dot\varepsilon_{\mathrm{mem}}^2$ | membrane invariant | `eps_mem_sq` |
| $\mathcal I_1$ | basal-to-surface shear integral | `F1` |
| $\mathcal I_2$ | basal-to-mean shear integral | `F2` |
| $U_b,U_s$ | basal and surface speeds | `u_b`, `u_s` |
| $\beta_{\mathrm{eff}}$ | drag coefficient in the nonlinear residual | `beta_eff` |
| $\varphi$ | grounded fraction | `phi` |

Goldberg's double integral satisfies

$$
\omega=H\mathcal I_2,
\qquad \omega/H=\mathcal I_2=\texttt{F2}.
$$

This identity removes the common factor-of-$H$ ambiguity. The symbol $m_s$
denotes the Weertman exponent (`m` in code), avoiding collision with
Goldberg's bed-slope factor. The continuum equations above use physical stress
units; GLIDE divides them by $\rho g$, so `B` is stored as
$A^{-1/n}/(\rho g)$. `eps_reg` and `eps_reg_shear` are *squared*
strain rates. The diagnostics `u_b` and `u_s` currently store speeds, with
direction inherited from $\bar{\mathbf u}$.

### 3.2 Drag in the residual and its linearizations

The nonlinear basal drag defines a curve $\tau_b(\bar U)$. Evaluating the
residual and differentiating the residual require two different quantities:
the ratio $\tau_b/\bar U$ and the derivative $d\tau_b/d\bar U$. They coincide
only for a linear drag law. In GLIDE, three related expressions appear:

1. The nonlinear residual uses the traction-to-speed ratio
   $\beta_{\mathrm{eff}}=c/(1+c\mathcal I_2)$. This rewrites the drag vector as
   $\vec{\tau}_b=\beta_{\mathrm{eff}}\bar{\mathbf u}$ at the current state; it
   is not a derivative.
2. The Vanka preconditioner needs a local derivative. It freezes
   $\mathcal I_2$ while differentiating the basal closure, giving
   $$
   \left.\frac{d\tau_b}{d\bar U}\right|_{\mathcal I_2}
   =\frac{f'}{1+f'\mathcal I_2}.
   $$
   The Vanka block does not represent this derivative at all: it reads
   $\beta_{\mathrm{eff}}$ as a frozen coefficient, which is the *secant*
   $\tau_b/\bar U$ rather than any tangent (section 5.2).
3. The fixed-$H$ velocity-block JVP and adjoint include the change in
   $\mathcal I_2$ as traction changes. The full derivative is
   $$
   \frac{d\tau_b}{d\bar U}
   =\frac{f'}{1+f'\mathcal I_2+ff'd\mathcal I_2/d\tau_b}.
   $$

Thus the residual coefficient, the preconditioner derivative, and the exact
derivative have distinct roles. Freezing $\mathcal I_2$ changes only the
preconditioner; it does not change the converged equations or their exact
linearization.

## 4. The DIVA coefficient refresh

The existing SSA solver already knows how to solve a two-dimensional membrane
stress balance. At each residual or smoothing step it needs two constitutive
quantities: a membrane viscosity and a coefficient multiplying velocity in the
basal-drag term. DIVA preserves that solver and supplies different values for
those two quantities:

$$
\text{SSA-shaped operator inputs}
\quad\longleftarrow\quad
\bar\eta,\ \beta_{\mathrm{eff}}
\quad\longleftarrow\quad
\text{DIVA column closure}.
$$

The division of responsibility is important:

| Quantity | Where it lives | Role |
|---|---|---|
| $\bar{\mathbf u}$ | staggered horizontal grid | globally coupled velocity solved by multigrid |
| $U_b$ | one scalar per cell | local unknown solved by the DIVA basal closure |
| $\bar\eta$ | one scalar per cell | membrane-viscosity coefficient read by the SSA stencil |
| $\beta_{\mathrm{eff}}$ | one scalar per cell | basal-drag coefficient read by the SSA stencil |
| $\mathcal I_1,\mathcal I_2,U_s$ | one scalar per cell | closure diagnostics and surface-observation terms |

The essential cell-local problem is to determine the basal speed $U_b$ that is
consistent with the current depth-averaged speed $\bar U$. Once $U_b$ is
known, the sliding law gives the basal traction, and the column calculation can
return the effective viscosity $\bar\eta$ and effective drag
$\beta_{\mathrm{eff}}$ required by the SSA-shaped operator.

The difficulty is that $U_b$ cannot be determined independently of the
vertical column. Its nonlinear closure contains $\mathcal I_2$, and its Newton
derivative contains $d\mathcal I_2/d\tau_b$. Evaluating either quantity
requires a vertical integral involving $1/\eta(z)$, while $\eta(z)$ is itself
defined implicitly by Glen's law and the DIVA shear-stress closure. This
dependency is the reason for the nested viscosity solves, quadrature, and
basal-speed Newton iteration described below.

$U_b$ is therefore not a new global degree of freedom. It is one scalar
diagnosed independently in each cell. Its direction is inherited from
$\bar{\mathbf u}$; only its magnitude must be solved. Once horizontal finite
differences have supplied the membrane strain rate, columns do not communicate
while determining $U_b$, $\bar\eta$, or $\beta_{\mathrm{eff}}$.

### 4.1 The refresh--use--update loop

DIVA is coupled to the SSA solver through a repeated coefficient refresh:

$$
(\bar{\mathbf u},H)^{(k)}
\xrightarrow{\text{DIVA column refresh}}
(U_b,\bar\eta,\beta_{\mathrm{eff}},\mathcal I_1,\mathcal I_2,U_s)^{(k)}
\xrightarrow{\text{SSA residual or smoother}}
(\bar{\mathbf u},H)^{(k+1)}.
$$

One pass through this loop has three stages:

1. From the current velocity field, compute $\bar U$ and the membrane strain
   invariant in every cell.
2. Independently in each cell, solve the basal-speed closure and vertical
   constitutive problem. This produces the coefficients $\bar\eta$ and
   $\beta_{\mathrm{eff}}$ required by the momentum stencil.
3. Apply the usual SSA-shaped residual or Vanka block using those coefficient
   fields. The resulting velocity update makes the old coefficients stale, so
   they are refreshed before the next residual or smoothing application.

This is a segregated nonlinear solve: the multigrid machinery updates the
horizontal velocity, while the DIVA kernel makes the column state consistent
with the latest velocity. There is no separate global solve for $U_b$ and no
three-dimensional velocity array.

### 4.2 Stage 1: form the inputs to each column

A coefficient refresh begins with the current multigrid state
$(\bar{\mathbf u},H)$. Horizontal finite differences of
$\bar{\mathbf u}$ give the membrane strain invariant
$\dot\varepsilon_{\mathrm{mem}}^2$, and interpolation to the cell center gives
the depth-averaged speed $\bar U$. Each cell also reads its thickness,
rheology, grounded fraction, and sliding-law parameters.

These are the only inputs needed by the DIVA column calculation:

$$
(\dot\varepsilon_{\mathrm{mem}}^2,\bar U,H,B,
  \beta,\varphi,u_c,m_s,\ldots)_c.
$$

Computing the membrane strain rate is the only part of the refresh that uses
neighboring velocities. Once these cell-centered inputs are available, every
column can be solved independently.

### 4.3 Stage 2: solve the column and form its coefficients

For each cell, the goal is to determine how much of $\bar U$ comes from basal
sliding and how much comes from internal shear. The unknown is the scalar basal
speed $U_b$. A trial value of $U_b$ determines the basal traction through the
sliding law, but evaluating whether that trial is correct requires the
viscosity and shear integrals throughout the column. The calculation is
therefore nested:

1. The outer solve proposes $U_b$ and computes $\tau_b=f(U_b)$.
2. At each vertical quadrature node, an inner solve determines the viscosity
   consistent with that traction.
3. The quadrature gives $\bar\eta$, $\mathcal I_1$, and $\mathcal I_2$.
4. The outer solve uses $\mathcal I_2$ to update $U_b$.

The basal-speed equation depends directly on $\mathcal I_2$, not on
$\bar\eta$. Both quantities nevertheless require the same viscosity profile:
$\mathcal I_2$ integrates $1/\eta$, whereas $\bar\eta$ integrates $\eta$.
GLIDE therefore accumulates them together during each evaluation of the
basal-speed residual.

#### 4.3.1 Why the viscosity requires a local Newton solve

The viscosity closure is the discrete form of the continuum relation derived
in Section 2.3. DIVA assumes that shear stress decreases linearly from its
basal value to zero at the surface,

$$
\tau_{xz}=\tau_{b,x}\zeta,
\qquad
\tau_{yz}=\tau_{b,y}\zeta.
$$

Since $\tau_{xz}=\eta\,\partial_z u$ and
$\tau_{yz}=\eta\,\partial_z v$, the vertical shear contribution to the
strain-rate invariant is

$$
\dot\varepsilon_{xz}^2+\dot\varepsilon_{yz}^2
=\frac{\tau_b^2\zeta^2}{4\eta^2}.
$$

Substitution into Glen's law gives, at each quadrature node,

$$
\boxed{
\eta=\tfrac12B
\left(
A_{\mathrm{reg}}+\frac{\tau_b^2\zeta^2}{4\eta^2}
\right)^p},
\qquad
A_{\mathrm{reg}}
=\dot\varepsilon_{\mathrm{mem}}^2+\varepsilon_0^2.
$$

This equation is implicit: increasing viscosity reduces the shear strain rate,
but the reduced strain rate feeds back into the viscosity. Equivalently, with
$k=(\tau_b\zeta/2)^2$,

$$
F_\eta(\eta)=\eta-G(\eta)=0,
\qquad
G(\eta)=\tfrac12B(A_{\mathrm{reg}}+k/\eta^2)^p.
$$

GLIDE solves this scalar equation with Newton's method,

$$
\eta\leftarrow\eta-\frac{\eta-G}{1-G'},
\qquad
G'=-2p\frac{k/\eta^2}{A_{\mathrm{reg}}+k/\eta^2}\frac{G}{\eta}.
$$

A Newton solve is used because $\eta$ appears on both sides and GLIDE supports
general Glen exponent $n$. For $n=3$ the equation can be rearranged as the
cubic

$$
8A_{\mathrm{reg}}\eta^3+8k\eta-B^3=0,
$$

but the corresponding Cardano formula suffers float32 cancellation in the
shear-dominated regime and does not extend to general $n$. The Newton problem
is well behaved: at the root, $1/n\le1-G'\le1$, so its denominator stays
bounded away from zero.

#### 4.3.2 Integrate the converged node viscosities

For a given trial traction, the node solves provide the viscosity values needed
for Gauss--Legendre quadrature on $\zeta\in[0,1]$:

$$
\bar\eta\approx\sum_kw_k\eta_k,
\qquad
\mathcal I_\alpha\approx
H\sum_kw_k\frac{\zeta_k^\alpha}{\eta_k}.
$$

No vertical array is stored; one CUDA thread solves and accumulates the nodes
for one cell. The nodes and weights are generated and cached by
`ForwardOperators._quadrature`. In the shear-dominated limit,
$\eta\sim\zeta^{1-n}$, so the $\mathcal I_1$ and $\mathcal I_2$
integrands behave like $\zeta^n$ and $\zeta^{n+1}$. For $n=3$, ideal-slab
values are exact at two and three nodes; the default is $N_\sigma=8$.

Interior Gauss nodes provide the complete integrals needed by the
depth-integrated equations, but not partial integrals as a function of height.
Reconstructing a full $\mathbf u(z)$ profile will therefore require a separate
cumulative scheme.

The implementation evaluates two regularized viscosities at each node:

$$
\begin{array}{lll}
\eta_{\mathrm{mem}}:&
\dot\varepsilon_{\mathrm{mem}}^2+\varepsilon_0^2
&\longrightarrow\bar\eta,\\
\eta_{\mathrm{sh}}:&
\dot\varepsilon_{\mathrm{mem}}^2+\varepsilon_{0,\mathrm{sh}}^2
&\longrightarrow\mathcal I_1,\mathcal I_2.
\end{array}
$$

Defaults are $10^{-6}$ and $10^{-12}$. This split is an implementation
regularization, not part of DIVA theory. `eta_bar` retains the SSA
floor so the membrane operator remains bounded and reproduces the exact
zero-shear SSA limit. The shear integrals use the smaller floor because the SSA
floor makes slow columns spuriously soft. Setting the values equal recovers a
single-viscosity calculation.

#### 4.3.3 Enforce the basal-speed closure

The quadrature for a trial $U_b$ supplies the value of
$\mathcal I_2(\tau_b)$ needed by the depth-average relation. As derived in
Section 2.5, the correct basal speed is the root of

$$
R(U_b)=U_b+f(U_b)\mathcal I_2(f(U_b))-\bar U=0.
$$

This is a second, outer scalar Newton solve. It is needed even for a linear
sliding law: changing $U_b$ changes $\tau_b$, which changes the vertical shear,
which changes the viscosity and therefore $\mathcal I_2$. Each evaluation of
$R$ consequently runs the node-viscosity solves and quadrature described
above.

The Newton update is

$$
U_b\leftarrow U_b-\frac{R(U_b)}{R'(U_b)},
$$

with the complete derivative

$$
R'
=1+f'\mathcal I_2
 +ff'\frac{d\mathcal I_2}{d\tau_b}.
$$

The last term accounts for the response of the column viscosity to traction;
it is obtained by differentiating the implicit viscosity closure at the
quadrature nodes. Omitting it lags the column response and caused a high-drag
two-cycle. For the monotone sliding laws and shear-thinning rheology used here,
all three contributions are non-negative, so $R'\ge1$ and the root in
$[0,\bar U]$ is unique.

The complete per-cell calculation in `diva_coeffs_cell<T>` is

```text
inputs: membrane strain invariant, U_bar, H, rheology, and sliding parameters
clamp stored U_b to [0, U_bar] as a warm start

outer Newton iteration on R(U_b):
    tau_b = c(U_b) U_b
    for each quadrature node:
        inner Newton solve for eta_mem using eps_reg
        inner Newton solve for eta_sh  using eps_reg_shear
        accumulate eta_bar, I1, I2, and dI2/dtau_b
    update U_b using the complete derivative R'(U_b)

perform a final quadrature at the converged traction
write eta_bar, F1, F2, u_b, beta_eff, and u_s
```

The inner viscosity solve starts from the smaller of its shear-free and
shear-dominated asymptotic estimates. The outer solve starts from the stored
cell value of `u_b`, which is a warm start rather than an independently
advanced state variable. The viscosity and basal-speed iteration caps are 12
and 20, respectively, although typical warm-started counts are much lower.
Both solves use a relative tolerance and stagnation detection.

Once a primal solve has converged, one extra Newton step brings a dual
derivative to the fixed-point sensitivity. Per-cell `cap_flags`
report any iteration backstop hit; both cap counts should remain zero.

#### 4.3.4 Form the coefficients returned to SSA

At the converged $U_b$, the final quadrature gives a mutually consistent
$\bar\eta$, $\mathcal I_1$, and $\mathcal I_2$. The sliding coefficient
$c(U_b)$ then gives

$$
\beta_{\mathrm{eff}}
=\frac{c(U_b)}{1+c(U_b)\mathcal I_2}.
$$

The two fields required by the SSA-shaped momentum operator are therefore

$$
\boxed{\bar\eta_c,\qquad \beta_{\mathrm{eff},c}}.
$$

The same pass stores $U_b$, $\mathcal I_1$, $\mathcal I_2$, and

$$
U_s=U_b+\tau_b\mathcal I_1
$$

for warm starts, diagnostics, and surface-velocity objectives. The kernel exits
only after quadrature at the accepted traction, so all returned quantities
describe the same column state. Only the owning non-halo thread writes these
cell fields.

### 4.4 Stage 3: apply the SSA-shaped operator

The coefficient refresh is complete once every cell has supplied
$\bar\eta$ and $\beta_{\mathrm{eff}}$. The existing residual or smoother then
uses those fields exactly where the SSA path uses membrane viscosity and basal
drag. During that one operator application the fields are held fixed. When the
multigrid step updates $\bar{\mathbf u}$, the inputs in Stage 1 have changed and
the loop begins again with another coefficient refresh.

Section 5 describes where these refreshes occur in the residual, Vanka
smoother, and multigrid hierarchy.

## 5. Using the coefficients in SSA multigrid

Once the column refresh has produced $\bar\eta$ and
$\beta_{\mathrm{eff}}$, the rest of the forward solve deliberately looks as
much like SSA as possible. The residual stencil, Vanka layout, FAS transfers,
and global velocity unknowns are retained.

### 5.1 Residual and refresh points

`residual_body<bool DIVA>` is shared by SSA and DIVA. On the DIVA
path it reads `eta_bar` and applies the basal term
$-\beta_{\mathrm{eff}}\bar{\mathbf u}$. On the SSA path, viscosity and the
sliding law are evaluated inline.

`compute_diva_coeffs()` runs before every DIVA residual evaluation
and before each Vanka relaxation step. The returned fields are held fixed only
while that one residual or local block is evaluated. After the smoother changes
velocity, the fields are stale until the next refresh. They are not treated as
fixed coefficients for the nonlinear solve or for an entire multigrid V-cycle.

### 5.2 The Vanka block

The DIVA smoother's local block has exactly the same unknowns as SSA's,

$$\mathbf x=(u_l,u_r,v_t,v_b,H_c),$$

and reads $\bar\eta$ and $\beta_{\mathrm{eff}}$ as **frozen coefficients**. That is the
whole of it. The block is structurally identical to SSA's, differing only in where its two
coefficients come from, and it never sees $U_b$ — the kernel is not even passed it.

**Ownership.** `compute_diva_coeffs` alone diagnoses $U_b$, $\bar\eta$,
$\beta_{\mathrm{eff}}$, $\mathcal I_1$, $\mathcal I_2$ and $U_s$ from $(u,v,H)$, and
refreshes them before every residual and every smoothing application. No solver block
treats any of them as an unknown. This is the same contract SSA's smoother has with its
frozen viscosity, and it is why the closure needs no representation inside the block: it is
re-solved to tolerance immediately afterwards regardless of what the block did.

#### Why $U_b$ is not an augmented unknown

Earlier versions carried $U_b$ as a sixth local unknown with its closure row,

$$
\begin{bmatrix}\mathsf A&\mathbf b\\\mathbf c^T&d\end{bmatrix}
\begin{bmatrix}\delta\mathbf x\\\delta U_b\end{bmatrix}
=\begin{bmatrix}\mathbf r\\r_{U_b}\end{bmatrix},
\qquad d=1+f'\mathcal I_2\ge1,
$$

and eliminated it exactly, $(\mathsf A-\mathbf b\mathbf c^T/d)\delta\mathbf x
=\mathbf r-\mathbf b r_{U_b}/d$. That rank-1 update is the Newton tangent for the drag,
and it was correct algebra. It was removed anyway, for three reasons.

**It destabilizes the smoother for sublinear sliding.** Since

$$
\frac{\partial\beta_{\mathrm{eff}}}{\partial U_b}
=\frac{c'(U_b)}{(1+c\mathcal I_2)^2},
\qquad c'\propto(m_s-1)U ,
$$

the correction is negative for $m_s<1$, and the momentum diagonals are already negative
because drag resists. On Greenland at $m_s=1/3$, $\Delta t=10$ a it drove
$\max|u|$ through $4.8\to6.8\times10^4\to1.1\times10^7$ and overflowed float32 to NaN
inside one V-cycle, at every multigrid level but one, while SSA on the same geometry was
clean everywhere.

**The failure was local, which is why damping could not fix it.** From one warmed state,
enabling the correction left the *median* correction unchanged, $0.00289\to0.00300$, while
the maximum went $5.31\to4108$ with six cells above $10^3$: the update nearly cancels the
block diagonal in a handful of cells and leaves the $5\times5$ near-singular there. A global
$\omega$ scales every cell alike, so $\omega=0.05$ still diverged.

**It never bought anything.** Measured where it is stable, it ties at best:

| | $U_b$ frozen | $U_b$ condensed |
|---|---|---|
| $m_s=1/3$, $\Delta t=0.1$ | 4 sweeps to a $10\times$ residual drop | diverges |
| $m_s=1/2$, $\Delta t=0.1$ | 4 sweeps, $\lvert r\rvert\to0.018$ | 4 sweeps, $\lvert r\rvert\to0.026$ |
| $m_s=1/2$, $\Delta t=1$ | 5 sweeps | 18 sweeps |
| $m_s=1/3$, $\Delta t=1$ | 5 sweeps | NaN |

At $m_s=1$ the correction is identically zero, so a linear law never saw it either way.

None of this touches the answer. The residual, the JVP and the adjoint all carry the full
closure response; only the preconditioner lags it, exactly as SSA's smoother uses a frozen
viscosity while its VJP carries $d\eta/du$. `diva_smoother_stability_test.py` pins the
outcome; `lu_6x6_solve` and its test existed only to verify the condensation and were
removed with it.

### 5.3 The adjoint smoother and off-equilibrium states

The adjoint smoother needs one safeguard the forward one does not, and the reason is
structural rather than incidental.

**What DIVA loses.** SSA's momentum residual is the gradient of an action functional
*including* the viscosity's dependence on strain rate, so its velocity-block Jacobian is a
Hessian and is symmetric — measured at $1.2\times10^{-7}$, round-off. That is what lets
`vjp_body` seed the viscosity tile with $\lambda$, and it is what makes $\omega=0.5$ safe
by the standard argument for a definite operator.

Goldberg derives DIVA variationally too, but his self-adjointness claim is explicitly
conditional — self-adjoint *"ignoring dependence of viscosity on strain rate."* The
implementation honours exactly that: freeze the coefficient path and DIVA's velocity block
is symmetric to $2.3\times10^{-7}$; include it and the asymmetry is $6.3\times10^{-3}$.
The term outside the guarantee is precisely the term that makes DIVA more than SSA.

**What that costs.** Asymmetry removes the a priori bound on $\omega$, not convergence
itself — the preconditioned spectrum stays in the right half plane, so a small enough step
still converges. But the smoother acquires a growing eigenmode when it is linearised about
a state far from equilibrium, which multigrid produces at every coarse level: a restricted
fine solution does not satisfy the coarse equations (forward residual $4019$ against
$0.0845$ for a native solve). Perturbing a *native* state with pure noise destabilises it
identically once far enough off equilibrium, so restriction is the trigger, not the cause.

**Why it is tractable.** The timescales separate. Amplification of a random adjoint field
at a restricted state:

| sweeps | 1 | 10 | 30 | 60 | 150 |
|---|---|---|---|---|---|
| $\lVert\lambda\rVert/\lVert\lambda_0\rVert$ | 0.85 | **0.67** | 1.98 | 518 | $1.1\times10^{10}$ |

The modes a smoother exists to kill die in the first ten sweeps; the growing mode needs
twenty to thirty to emerge. So a smoother doing its actual job never meets it. The failure
appeared only because the Greenland example inherited `post_steps=150` from the *forward
SSA* configuration — seven times this solver's own default, and far beyond what a smoother
is for. SSA is indifferent to the excess; DIVA is not.

**The safeguard.** `vanka_config.smoother_growth_check` (default 5) measures the adjoint
residual every few sweeps and stops the sweep when it rises: a smoother that is increasing
the residual is doing nothing useful, and because the iteration is linear with a fixed
state the growth will not reverse. It exploits the separation above rather than fighting
the spectrum, and costs nothing when nothing is wrong — on a well-conditioned solve the
gradient is bit-identical with the check on or off. At level 1 with `post_steps=150`, the
configuration that reached $2.4\times10^{16}$, the adjoint now converges in two V-cycles
($6.9\times10^{-2}\to9.4\times10^{-3}$).

Global under-relaxation was tried and rejected. `diva_omega` does stabilise the mode, but
it scales the whole spectrum, so taming one eigenvalue cripples every mode the smoother
exists to damp: on a well-conditioned case the adjoint needs 2 V-cycles at $\omega=0.5$
and more than 200 at $\omega=0.3$. It remains available as an escape hatch, defaulting to
$0.5$.

Two things that look like fixes and are not. Making the block represent the closure path
is *verified correct* — it matches finite differences to $1.7\times10^{-2}$ — and still
diverges, because 47% of that path is cross-cell coupling a $5\times5$ block cannot hold,
and half a term is worse than none. And equalising `eps_reg_shear` with `eps_reg` reduces
the asymmetry fiftyfold, but against a frozen-operator baseline of $2.3\times10^{-7}$ it
is a contributor, not the cause.

### 5.4 Multigrid levels and convergence

The FAS hierarchy continues to transfer and correct the same globally coupled
state as SSA. Each level also owns cell-centered arrays for `u_b`,
`eta_bar`, and `beta_eff`. A restricted `u_b`
can provide a useful initial guess on the coarse level, but it is not accepted
as the coarse-level solution: the DIVA kernel recomputes the closure and
coefficients from that level's own velocity, thickness, and grid spacing before
the coarse operator uses them.

There are consequently two useful convergence diagnostics:

1. The momentum residual is evaluated after refreshing the DIVA coefficients,
   so it measures the actual nonlinear DIVA equations rather than an old
   frozen-coefficient surrogate.
2. The basal-closure residual can be measured immediately before a refresh to
   show how far the stored column state drifted while the preceding velocity
   update was applied.

They are reported separately because the momentum residual has stress-balance
units whereas the basal-closure residual has velocity units. At convergence,
both the global momentum equations and the independent cell closures are
consistent.

## 6. Linearization and adjoint

This section summarizes the adjoint structure needed to understand the
numerical method. A step-by-step dependency trace, including the facet stencil,
the nested implicit derivatives, and the CUDA call sequence, is given in
[diva_adjoint.md](diva_adjoint.md).

Let the coupled forward state be

$$
\mathbf x=(u,v,H),
$$

and let the two DIVA coefficients returned by cell $c$ be

$$
\mathbf C_c(\mathbf x)
=\left(\bar\eta_c(\mathbf x),\,
       \beta_{\mathrm{eff},c}(\mathbf x)\right).
$$

A momentum residual row depends on the state in two ways. There is a direct
dependence through the usual SSA stencil, and an indirect dependence through
the DIVA coefficients of the nearby cells used by that stencil:

$$
r_i=r_i\!\left(\mathbf x,\{\mathbf C_c(\mathbf x)\}_{c\in\mathcal N(i)}\right).
$$

For a perturbation $\delta\mathbf x$,

$$
\delta r_i
=
\left.\frac{\partial r_i}{\partial\mathbf x}\right|_{\mathbf C}
\delta\mathbf x
+\sum_{c\in\mathcal N(i)}
\left(
\frac{\partial r_i}{\partial\bar\eta_c}\,\delta\bar\eta_c
+
\frac{\partial r_i}{\partial\beta_{\mathrm{eff},c}}\,
\delta\beta_{\mathrm{eff},c}
\right).
$$

The first term is the familiar fixed-coefficient SSA linearization. The second
term differentiates the cell-local column refresh. The Vanka smoother
approximates the second term by freezing the column viscosity and
$\mathcal I_2$ while eliminating the local $U_b$ increment, as described in
Section 5.2. That approximation is appropriate for a preconditioner. A JVP or
adjoint intended to represent the converged nonlinear equations must include
the complete second term.

### 6.1 Differentiating one cell closure

Velocity enters the closure in cell $c$ through two cell-centered scalars,

$$
q_{1,c}=\dot\varepsilon_{\mathrm{mem},c}^2,
\qquad
q_{2,c}=\bar U_c.
$$

The required closure derivatives are

$$
\bar\eta_{,q_1},\quad \bar\eta_{,q_2},
\qquad
\beta_{\mathrm{eff},q_1},\quad
\beta_{\mathrm{eff},q_2}.
$$

GLIDE obtains them by running the complete cell calculation in dual arithmetic,
not by deriving separate formulas for every nested dependency.
`compute_diva_derivs` seeds $q_1$ and $q_2$ in turn and executes the
basal-speed Newton solve, every node-viscosity Newton solve, and the vertical
quadrature in `DualFloat`. The derivative carried by the converged
output therefore includes the paths through $U_b$, $\tau_b$, $\eta(z)$, and
$\mathcal I_2$.

For a forward JVP, `populate_diva_coeffs_dual` instead seeds a
velocity direction directly. The finite-difference and interpolation operations
produce directional derivatives of $q_1$ and $q_2$, and the same dual column
solve returns the corresponding directional derivatives of $\bar\eta$ and
$\beta_{\mathrm{eff}}$.

### 6.2 The two-pass coefficient transpose

For an adjoint vector $\vec{\lambda}$, the coefficient part of
$\mathbf J^T\vec{\lambda}$ is evaluated in two passes.

First, `vjp_body<true>` visits the momentum residual rows. In addition
to transposing the direct fixed-coefficient stencil, it accumulates the
sensitivity of all residual rows touching cell $c$ into two cell-centered
weights:

$$
W_{\eta,c}
=\sum_i\lambda_i\frac{\partial r_i}{\partial\bar\eta_c},
\qquad
W_{\beta,c}
=\sum_i\lambda_i
 \frac{\partial r_i}{\partial\beta_{\mathrm{eff},c}}.
$$

These weights answer a simple question: after contraction with the adjoint,
how strongly does the objective depend on each coefficient produced by this
cell?

Second, `compute_diva_vjp_coeffs` combines those weights with the
closure derivatives:

$$
A_c
=W_{\eta,c}\bar\eta_{,q_1}
 +W_{\beta,c}\beta_{\mathrm{eff},q_1},
\qquad
B_c
=W_{\eta,c}\bar\eta_{,q_2}
 +W_{\beta,c}\beta_{\mathrm{eff},q_2}.
$$

It then scatters

$$
A_c\frac{\partial q_{1,c}}{\partial(u,v)}
+
B_c\frac{\partial q_{2,c}}{\partial(u,v)}
$$

back to the velocity facets that contribute to cell $c$. Splitting the
transpose into residual-to-coefficient and coefficient-to-velocity passes keeps
each kernel within a one-cell halo. Dirichlet facets are skipped.

Together with the direct transpose from the first pass, this is the exact
velocity-block transpose at fixed $H$. No symmetry of the full nonlinear
Jacobian is assumed. The adjoint Vanka smoother instead transposes the same
frozen-coefficient block used by the forward preconditioner.

### 6.3 Thickness dependence of the closure

The coefficient map also depends on thickness,

$$
\mathbf C_c
=\mathbf C_c(q_{1,c},q_{2,c},H_c),
$$

because

$$
\mathcal I_\alpha
=H_c\int_0^1\frac{\zeta^\alpha}{\eta(\zeta)}\,d\zeta .
$$

Changing $H_c$ changes $\mathcal I_2$, the solved $U_b$ and $\tau_b$, and
therefore both $\bar\eta$ and $\beta_{\mathrm{eff}}$. Thickness is the only
input that enters through the quadrature factor alone, which is why typing
`H_c` as the templated scalar in `diva_coeffs_cell` captures the whole path.

The coupled linearization therefore includes

$$
K_c
=W_{\eta,c}\bar\eta_{,H}
 +W_{\beta,c}\beta_{\mathrm{eff},H},
$$

added to the thickness component of $\mathbf J^T\vec{\lambda}$. Unlike the
velocity inputs, which are assembled from surrounding facets and need a scatter,
$H_c$ belongs to the cell itself, so $K_c$ lands on one cell with no stencil.
`compute_diva_vjp_coeffs` accumulates it into the thickness cotangent that the
first VJP pass already built from the residual's explicit $H$ terms.

The JVP counterpart seeds the thickness direction in
`populate_diva_coeffs_dual`, so a direction with $\delta H\ne0$ carries the
closure's response and not merely the explicit terms.

Consequently the DIVA JVP and adjoint are now exact for the coupled $(u,v,H)$
velocity/thickness block at fixed auxiliary fields. If the grounded fraction is
treated as a diagnostic function of $H$, its derivative remains a separate
auxiliary-field path and is still outside the DIVA coefficient transpose
described here.

### 6.4 Parameter derivatives

The same dual-number mechanism supplies derivatives with respect to the three
supported sliding parameters. `compute_diva_derivs` currently uses
six seeds in total:

| Seed | Purpose |
|---|---|
| $q_1$, $q_2$ | velocity-block transpose and surface objective |
| $H$ | thickness-block transpose and surface objective |
| $\beta$ | drag gradient and surface explicit term |
| $u_c$ | Coulomb gradient and surface explicit term |
| $m_s$ | Weertman-exponent gradient and surface explicit term |

Each seed yields derivatives of $\bar\eta$,
$\beta_{\mathrm{eff}}$, and $U_s$, for 18 stored fields. For a parameter $p$,
the coefficient contribution is

$$
\sum_c
\left(
W_{\eta,c}\bar\eta_{,p}
+W_{\beta,c}\beta_{\mathrm{eff},p}
\right).
$$

Both paths matter: changing $\beta$, for example, changes the basal drag and
also changes viscosity through the resulting vertical shear.

### 6.5 Surface-speed objectives

A surface-speed objective depends on the reconstructed surface speed,

$$
U_s=U_b+\tau_b\mathcal I_1,
$$

rather than directly on the depth-averaged velocity solved by multigrid.
`diva_surface_misfit_rhs` uses $U_{s,q_1}$ and $U_{s,q_2}$ to
transpose the surface misfit back to velocity facets. The objective also has
the explicit parameter term

$$
\left.\frac{\partial J}{\partial p}\right|_{\mathrm{explicit}}
=\sum_c\frac{\partial J}{\partial U_{s,c}}
        \frac{\partial U_{s,c}}{\partial p},
$$

returned by `diva_surface_param_gradient`. Under SSA these helpers
return `None`.

As with the residual coefficients, $U_s$ also depends on $H$ through
$\mathcal I_1$, $\mathcal I_2$, and the solved $U_b$. That derivative is stored
as $U_{s,H}$ and `diva_surface_misfit_rhs` scatters it into the thickness
right-hand side alongside the velocity one, so a surface objective is
differentiated with respect to the full state.

## 7. Verification, code map, and status

The verification suite follows the same layers as the implementation: analytic
limits check the continuum formulas, cell tests check the coefficient refresh,
operator tests check its coupling to SSA, derivative tests check the
implemented linearization in both velocity and thickness directions, and full
solves check the multigrid integration.

The tests are not interchangeable, and the useful thing to know is which are sharp.
Strongest first:

| strength | mechanism | resolves to | tests |
|---|---|---|---|
| exact | bit-for-bit against a reference | 0 | `ssa_regression`, `diva_residual` |
| exact-arithmetic identity | an identity that must hold in any precision | round-off, $\sim$1e-7 | `diva_dotproduct` |
| external truth | closed-form analytic solution | round-off if the formulation is right | `diva_slab` |
| independent reference | the same quantity by a deliberately different algorithm | round-off | `diva_closure` |
| finite differences | perturb and re-solve | $\sim$1e-3 at best | `diva_derivs`, `diva_gradient`, `diva_surface_gradient`, `diva_surface_torch` |
| physical consistency | limits, invariants, determinism, boundedness | varies | `diva_solve`, `diva_smoother_stability` |

Two consequences shape how the adjoint is verified. Finite differences cannot resolve it —
the floor on this problem is $\sim$3e-3 ([open_questions.md](open_questions.md) Q7), so FD
agreement at 1e-3 is consistent with a real defect. The load-bearing adjoint check is
therefore the dot-product identity, which uses no FD and resolves to round-off. Where FD is
the only instrument it is run against an **SSA control** — the same objective and harness
through SSA's independently written exact adjoint — so the comparison measures against the
shared floor rather than a guessed tolerance. And no single FD step is trustworthy:
truncation falls like $\varepsilon^2$ while round-off grows like $1/\varepsilon$, so every
FD test sweeps $\varepsilon$ and reports the curve.

A caution the thickness work added: an end-to-end FD check can also be too *insensitive* to
be the instrument. Removing the closure's $H$ path moved the residual FD by only a factor of
two, because the residual's explicit $H$ dependence dominates it; the derivative had to be
checked where its own signal was largest, in isolation. Test each link where it is loudest.

| Test | Establishes |
|---|---|
| `ssa_regression_test.py` | SSA remains bit-for-bit identical |
| `diva_residual_test.py` | DIVA with SSA coefficients reproduces SSA; catches grounding errors |
| `diva_closure_test.py` | coefficients match independent bisection/Picard algorithms |
| `diva_slab_test.py` | shear integrals match the analytic slab |
| `diva_smoother_stability_test.py` | the smoother stays bounded for a sublinear sliding law |
| `diva_derivs_test.py` | all 18 closure derivatives match finite differences, including the three thickness ones |
| `diva_dotproduct_test.py` | velocity- and thickness-direction JVP checks, and the JVP/VJP transpose identity |
| `diva_surface_torch_test.py` | surface-velocity and thickness objectives through the torch wrapper |
| `diva_adjoint_test.py` | adjoint FAS converges |
| `diva_gradient_test.py` | all three parameter gradients |
| `diva_surface_gradient_test.py` | surface objective and SSA limit |
| `diva_solve_test.py` | convergence, determinism, bounds, and multigrid consistency |

Run from `glide/`:

```bash
for test in tests/diva_*_test.py tests/ssa_regression_test.py; do
    uv run python "$test" || exit 1
done
```

Float32 finite differences have a visible floor, so the dot-product identity is
the load-bearing test that the implemented JVP and VJP are transposes. It
cannot detect a dependency omitted from both. The independent finite-difference
check currently perturbs velocity only; a corresponding thickness-direction
check is needed. The independent closure implementation checks the algorithm,
while the slab solution can expose a shared formulation error.

| File | DIVA responsibility |
|---|---|
| `glide/cuda/diva.cu` | closure, quadrature, derivatives, coefficient and surface transposes |
| `glide/cuda/viscosity.cu` | shared membrane invariant |
| `glide/cuda/residuals.cu` | residual, fixed-$H$ velocity JVP, first VJP pass |
| `glide/cuda/vanka.cu` | frozen-coefficient local block |
| `glide/operators.py` | dispatch, refresh, gradients, surface helpers |
| `glide/multigrid.py` | forward and adjoint FAS integration |
| `glide/grid.py` | configuration and field allocation |

Select DIVA at construction with `stress_scheme='diva'` (a compile-time choice: the DIVA
kernels are built with `-DGLIDE_DIVA=1` on the `GLIDE_MOLHO=0` path); `'ssa'` is the default.
`n_sigma` defaults to 8 and `eps_reg_shear` to $10^{-12}$.

Implemented: isothermal DIVA for both sliding laws, forward and adjoint FAS,
the coupled $(u,v,H)$ JVP/VJP including the closure's thickness path, all three
sliding-parameter gradients, surface-speed objectives, and the analytic/SSA
limits. A
development benchmark on a $256^2$, five-level case measured about 11% overhead
(107 ms versus 96 ms); this is hardware- and
configuration-specific.

Remaining work, in priority order:

1. Validate ISMIP-HOM C against trusted first-order/full-Stokes reference data.
2. Resolve the broader `eps_reg` modeling policy; see
   [open_questions.md](open_questions.md).
3. Add partial integrals for $\mathbf u(z)$, then vertical velocity and thermal
   coupling.
4. Allow depth-varying $B(z)$.
5. Quantify the omitted bed-slope factor before validating steep-bed cases.

## Appendix A. Shear-integral derivative

At one quadrature node, write

$$
G(\eta,\tau_b)=\tfrac12BE^p,
\quad E=A_{\mathrm{reg}}+s_{\mathrm{sh}},
\quad s_{\mathrm{sh}}=\frac{\tau_b^2\zeta^2}{4\eta^2}.
$$

From the root $\eta=G$,

$$
\frac{d\eta}{d\tau_b}=\frac{G_{\tau_b}}{1-G_\eta},
\quad
G_\eta=-2p\frac{s_{\mathrm{sh}}}{E},
\quad
G_{\tau_b}=2p\frac{s_{\mathrm{sh}}\eta}{E\tau_b}.
$$

Therefore

$$
\frac{d\mathcal I_2}{d\tau_b}
=H\int_0^1\zeta^2\left(-\eta^{-2}
\frac{d\eta}{d\tau_b}\right)d\zeta.
$$

The kernel rewrites $s_{\mathrm{sh}}/\tau_b$ as
$\tau_b\zeta^2/(4\eta^2)$, giving zero cleanly at $\tau_b=0$.

## Appendix B. Invariants worth protecting

Each row below is a place where a plausible-looking edit is wrong, with the test that
catches it:

| change | consequence | caught by |
|---|---|---|
| re-applying $\varphi$ in the momentum kernels | grounding counted twice | `diva_residual_test` |
| collapsing the two level solves into one | $\mathcal I_2$ moves with $\varepsilon_0^2$, SSA limit drifts | `diva_slab_test` |
| any non-smooth op inside `diva_coeffs_cell` (`min`, clamp, magnitude branch) | kink in the differentiated path | `diva_dotproduct_test` |
| breaking either adaptive loop on the primal criterion | primal stays right, derivative degrades $10\times$ | `diva_dotproduct_test`, `diva_derivs_test` |
| depositing on Dirichlet rows in the coefficient scatter | unreducible adjoint convergence floor | adjoint V-cycle norms |
| letting the smoother's block see $U_b$ again | divergence for $m_s<1$, NaN on real geometry | `diva_smoother_stability_test` |
| disabling `smoother_growth_check`, or raising the adjoint's `post_steps` far past its default | the adjoint smoother's growing mode reappears at fine resolution | adjoint V-cycle norms |
| passing $H$ to the closure as a plain `float` | the thickness derivative silently becomes zero | `diva_derivs_test` |

And the standing invariants:


| Invariant | Primary test |
|---|---|
| Grounding enters `beta_eff` exactly once | `diva_residual_test.py` |
| `eta_bar` keeps the SSA regularization | `diva_closure_test.py` |
| `F1`, `F2` use the shear regularization | `diva_slab_test.py` |
| Closure slope includes $d\mathcal I_2/d\tau_b$ | `diva_closure_test.py` |
| The smoother reads $\bar\eta,\beta_{\mathrm{eff}}$ only, and stays bounded for $m_s<1$ | `diva_smoother_stability_test.py` |
| A converged dual Newton solve takes the extra step | `diva_derivs_test.py`, `diva_dotproduct_test.py` |
| Only owning threads write column fields | `diva_solve_test.py` |
| VJP scatter skips Dirichlet facets | `diva_adjoint_test.py` |
| Coefficients are recomputed on every multigrid level | `diva_solve_test.py`, `diva_adjoint_test.py` |
