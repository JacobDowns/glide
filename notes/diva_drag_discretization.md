# The DIVA operator and two discretizations of its basal drag: consistent vs mass-lumped

*A clear statement of the DIVA stress-balance operator, then a finite-element account of how one
choice — the way the basal-drag term is quadratured — produces the two operators we call **stock
DIVA** and **self-adjoint DIVA-A**. Includes a from-scratch explanation of mass lumping, since the
term recurs below. The punchline: stock DIVA is the **mass-lumped** discretization of DIVA-A's
**consistent** one, and that single quadrature choice is the entire origin of stock DIVA's
non-self-adjointness.*

Companion notes:

- [stress_balance_variational_structure.md](/home/bizon/glaciers/notes/stress_balance_variational_structure.md)
  — why DIVA's Jacobian is non-symmetric while SSA/MOLHO are self-adjoint; the variational anchor and
  the "closure path" framing this note refines.
- [diva_column_energy_derivation.md](/home/bizon/glaciers/notes/diva_column_energy_derivation.md)
  — DIVA as the condensation of a per-column dissipation potential $\Phi$; source of "residual $=\nabla\Phi$".
- [variational_condensation_primer.md](/home/bizon/glaciers/notes/variational_condensation_primer.md)
  — functional / self-adjoint / condensation / Schur complement from the ground up.

---

## 1. What the DIVA operator is

DIVA solves for a **single** horizontal velocity field, the depth-averaged velocity
$\bar{\mathbf u}=(\bar u,\bar v)$. Its stress balance is the force balance

$$
\underbrace{\nabla\!\cdot\!\big(2H\,\bar\eta\,\dot{\varepsilon}'(\bar{\mathbf u})\big)}_{\text{membrane stress divergence}}
\;-\;
\underbrace{\beta_{\mathrm{eff}}\,\bar{\mathbf u}}_{\text{basal drag }\;\vec{\tau}_b}
\;=\;
\underbrace{H\,\nabla s}_{\text{driving stress}},
$$

and the operator (residual) whose root we solve for is $\mathbf R(\bar{\mathbf u}) = (\text{LHS}) - (\text{RHS})$.
(GLIDE works in units divided by $\rho g$, which is why the driving stress is $H\nabla s$ with no density
or gravity; see `glide/glide/cuda/stress.cu`.) The three terms are:

- **Membrane.** $\dot{\varepsilon}'$ is the horizontal (SSA-shaped) strain-rate tensor of
  $\bar{\mathbf u}$; e.g. the $x$-row carries $\partial_x[2H\bar\eta(2\dot\varepsilon_{xx}+\dot\varepsilon_{yy})]
  +\partial_y[2H\bar\eta\,\dot\varepsilon_{xy}]$. $\bar\eta$ is the **depth-averaged** effective viscosity.
- **Basal drag.** $\vec{\tau}_b=\beta_{\mathrm{eff}}\bar{\mathbf u}$, where $\beta_{\mathrm{eff}}$ is the
  **effective** drag coefficient: DIVA's drag physically acts on the *basal* velocity $\mathbf u_b$, and the
  vertical closure eliminates $\mathbf u_b$ in favour of $\bar{\mathbf u}$, leaving
  $\beta_{\mathrm{eff}}=\beta/(1+\beta F_2)$ (Goldberg's shear factors $F_1,F_2$).
- **Driving.** $H\nabla s$, the only forcing.

**The two coefficients $\bar\eta$ and $\beta_{\mathrm{eff}}$ are state-dependent.** They are outputs of the
DIVA vertical closure, which ties them to the local strain and the cell-centre speed — schematically
$C=(\bar\eta,\beta_{\mathrm{eff}})$ is a function of $Q=(\dot\varepsilon_{\mathrm{mem}}^2,\ |\bar{\mathbf u}|)$
(the notation of the companion note). Hold that thought — it is the reason a purely geometric quadrature
choice ends up controlling self-adjointness (§5).

**Variational structure.** As the column-energy note shows, this operator is the gradient of a scalar
dissipation potential, $\mathbf R=\nabla\Phi$. In particular the basal-drag term is the first variation of
the basal dissipation

$$
D_b[\bar{\mathbf u}]=\tfrac12\int_\Omega \beta_{\mathrm{eff}}\,|\bar{\mathbf u}|^2\,d\Omega,
\qquad
\frac{\delta D_b}{\delta\bar{\mathbf u}}=\beta_{\mathrm{eff}}\,\bar{\mathbf u}=\vec{\tau}_b .
$$

So in the **continuum** the DIVA operator is variational (self-adjoint) — full stop. Everything below is
about whether a given **discretization** preserves that.

---

## 2. GLIDE's grid, the weak form, and the drag "mass matrix"

**GLIDE's grid, concretely.** GLIDE is *not* a Galerkin finite-element code — it is a finite-volume /
finite-difference discretization on a staggered (Arakawa C-) grid. The velocity unknowns live on cell
**facets**: the $x$-velocity $u$ on the east–west faces, the $y$-velocity $v$ on the north–south faces.
**Each facet carries exactly one momentum residual** — the momentum balance for the control volume centred on
that facet — so "the residual (or test function $N_f$) at DOF $f$" means precisely *the momentum equation at
facet $f$*. The cell **centres** hold the scalars $H,\beta,\bar\eta,\beta_{\mathrm{eff}}$ and the thickness
(mass) residual.

![Staggered C-grid: cell centres carry H, beta_eff, eta_bar and the normal stresses; u-facets carry the
x-velocity and its drag/driving; v-facets the y-velocity and its drag/driving; vertices carry the shear
stress.](figures/fig_cgrid.png)

**The momentum residual, term by term.** The $x$-momentum residual on a $u$-facet is assembled as

$$
r_u \;=\; \partial_x\sigma_{xx} \;+\; \partial_y\sigma_{xy} \;+\; \tau_{bx} \;-\; \tau_{dx},
$$

(and $r_v$ analogously on each $v$-facet). Each piece is placed on the C-grid as follows — **SSA and DIVA share
this layout exactly; only $\bar\eta$ and the drag coefficient differ** (§1):

| term | lives on | built from | SSA vs DIVA |
|---|---|---|---|
| $\sigma_{xx},\ \sigma_{yy}$ — normal membrane stress | **cell centres** | $\sigma_{xx}=2H\bar\eta\,(2\,\partial_x u+\partial_y v)$, from the cell's own four facets: $\partial_x u$ from its E/W $u$-facets, $\partial_y v$ from its N/S $v$-facets | same stencil; $\bar\eta$ computed inline (SSA) vs depth-averaged (DIVA) |
| $\sigma_{xy}$ — shear membrane stress | **vertices** | $\sigma_{xy}=2(H\bar\eta)_{\text{vtx}}\cdot\tfrac12(\partial_y u+\partial_x v)$; $H\bar\eta$ averaged over the 4 cells at the vertex | same; $\bar\eta$ as above |
| $\tau_{bx}$ — basal drag | **facet** | SSA: $-\beta\,|u_f|^{m-1}u_f$, with $|u_f|$ from the facet's own $u$ and its 4 neighbour $v$-facets. DIVA: $-\tfrac12(\beta_{\mathrm{eff},\ell}+\beta_{\mathrm{eff},c})\,u_f$, $\beta_{\mathrm{eff}}$ diagnosed by the closure at the two cells | **the one genuinely different term** (§4) |
| $\tau_{dx}$ — driving stress | **facet** | $\tau_{dx}=H_f\,\partial_x s$, with $H_f=\tfrac12(H_{\text{west}}+H_{\text{east}})$ and $\partial_x s$ from the two cell surfaces ($s$ follows flotation where the ice is afloat) | same |

The $u$-facet residual then differences the neighbouring pieces onto itself:
$\partial_x\sigma_{xx}=[\sigma_{xx}(\text{east cell})-\sigma_{xx}(\text{west cell})]/\Delta x$ and
$\partial_y\sigma_{xy}=[\sigma_{xy}(\text{vertex above})-\sigma_{xy}(\text{vertex below})]/\Delta x$. (The thickness /
mass residual is separate — a flux divergence on each cell centre — and shared by both models too.) So DIVA reuses
SSA's entire 2-D operator; the two coefficients from the vertical closure ($\bar\eta$, $\beta_{\mathrm{eff}}$) are
the only things that change, and of those only $\beta_{\mathrm{eff}}$'s placement — cell-centred, then applied
per-facet — is what the rest of this note is about.

So the finite-element "basis function / mass matrix" language below is a **lens**, not literal hat functions:
$N_f$ ↔ the momentum equation at facet $f$, and the drag "mass matrix" $B_{fg}$ ↔ the discrete stencil for how
the drag entering facet $f$'s residual depends on velocity $g$. Mass lumping — diagonal vs neighbour-coupling —
is a property of that stencil, and reads the same whether you reached it through an FE overlap integral or an FV
flux.

Put the balance in weak form: multiply by a test function $\vec{\phi}$ and integrate. Every term
becomes a bilinear/linear form; the one we care about is the basal drag,

$$
a_b(\bar{\mathbf u},\vec{\phi})=\int_\Omega \beta_{\mathrm{eff}}\,\bar{\mathbf u}\cdot\vec{\phi}\,d\Omega .
$$

Expand the velocity and the test function in the grid's basis functions $\{N_f\}$ (one per velocity DOF):
$\bar{\mathbf u}=\sum_g u_g N_g$, $\vec{\phi}=N_f$. Then the drag contributed to DOF $f$ is

$$
(\vec{\tau}_b)_f=\sum_g B_{fg}\,u_g,
\qquad
B_{fg}=\int_\Omega \beta_{\mathrm{eff}}\,N_f\,N_g\,d\Omega .
$$

$B$ is a **weighted mass matrix** (the weight is $\beta_{\mathrm{eff}}$). It is symmetric, and it has
**off-diagonal** entries wherever two basis functions overlap — i.e. the drag on DOF $f$ couples to
neighbouring velocities. This matrix is the *only* place stock DIVA and DIVA-A differ; the membrane and
driving discretizations are identical between them.

---

## 3. Mass lumping, in general

A **mass matrix** $M_{fg}=\int N_f N_g$ is not diagonal: neighbouring basis functions overlap, so a DOF's
"mass" is shared with its neighbours. **Mass lumping** replaces $M$ by a diagonal matrix $M_L$ that keeps
each row's *total* mass but drops the sharing:

$$
(M_L)_{ff}=\sum_g M_{fg},\qquad (M_L)_{fg}=0\ \ (f\neq g).
$$

Each row's off-diagonal mass is "lumped" onto the diagonal.

**Canonical 1-D example.** Linear (hat) basis functions on a uniform grid of spacing $h$. The consistent
mass matrix is tridiagonal, with interior row

$$
M:\quad \frac{h}{6}\,\big[\,1\ \ 4\ \ 1\,\big],
$$

(the $\tfrac{h}{6}$ on the neighbours is the overlap of adjacent hats). Its row sum is $h$, so lumping gives

$$
M_L=h\,I .
$$

The consistent matrix spreads a node's mass onto its neighbours (the two $1$'s); the lumped one keeps it
all on the node.

**Why anyone does this.** (i) In explicit time-stepping you must invert the mass matrix every step; a
diagonal $M_L$ inverts trivially, a full $M$ needs a solve. (ii) It localizes the stencil and damps
certain spurious modes. The cost is accuracy — but for these elements lumping is $O(h^2)$, the *same*
order as the discretization error already present, so the solution is unchanged to leading order.

**The structural fact we need.** Lumping a **symmetric** matrix yields a **symmetric** (diagonal) matrix.
*Mass lumping by itself never breaks self-adjointness.* DIVA's asymmetry therefore cannot come from lumping
alone — it needs one more ingredient (§5).

---

## 4. The two DIVA discretizations

Both schemes use the same membrane and driving discretization; they differ only in whether the drag matrix
$B$ of §2 is used **consistently** or **lumped**.

Recall a cell's centre velocity is the average of its two facets, $\bar u_c=\tfrac12(u_{\text{left}}+u_{\text{right}})$,
and a cell's basal drag $\beta_{\mathrm{eff},c}\,\bar u_c$ is distributed to that cell's facets. The picture below
is a 1-D $x$-slice through one $u$-facet $f$ and the two cells $\ell,c$ that share it — this is the whole stencil:

![Basal-drag stencil at a shared u-facet. The consistent (DIVA-A) drag on the facet reaches the two outer
facets u_left and u_right through the cell-centre velocities; the mass-lumped (stock DIVA) drag collapses both
onto the facet's own velocity.](figures/fig_drag_stencil.png)

The three vertical bars are the $u$-facets (x-velocity DOFs); the two dots are the cell centres where $\bar u$
lives. The consistent drag on $u_f$ pulls in the two *outer* facets $u_{left}, u_{right}$ through $\bar u_\ell,\bar u_c$;
lumping collapses both onto $u_f$ itself.

**DIVA-A — consistent (cell-centred).** Keep $B$. The drag on a $u$-facet $f$ shared by cells $\ell,c$ is

$$
(\tau_b)_f=\tfrac12\big(\beta_{\mathrm{eff},\ell}\,\bar u_\ell+\beta_{\mathrm{eff},c}\,\bar u_c\big),
\qquad
\bar u_\ell=\tfrac12(u_{\text{left}}+u_f),\quad \bar u_c=\tfrac12(u_f+u_{\text{right}}).
$$

Each cell's drag uses **that cell's centre velocity $\bar u$**, so facet $f$ is coupled to its neighbouring
facets through $\bar u_\ell,\bar u_c$. This is the full (neighbour-coupling) mass matrix. Because the
coefficient $\beta_{\mathrm{eff}}$ *and* the velocity both live on the same $\bar u$, the discrete drag is
exactly $\partial D_{b,h}/\partial u_f$ of a discrete potential $D_{b,h}$, and its Jacobian is that
potential's Hessian — **symmetric**.

**Stock DIVA — mass-lumped.** Lump $B$ onto its diagonal: the cell's drag lands entirely on each facet's
**own** velocity,

$$
(\tau_b)_f=\tfrac12\big(\beta_{\mathrm{eff},\ell}+\beta_{\mathrm{eff},c}\big)\,u_f .
$$

The drag on facet $f$ now depends only on $u_f$ (diagonal). This is literally the row-sum lumping of the
consistent form: sending $\bar u_\ell,\bar u_c\to u_f$ collapses
$\tfrac12(\beta_\ell\bar u_\ell+\beta_c\bar u_c)\to\tfrac12(\beta_\ell+\beta_c)u_f$.

These two expressions are exactly GLIDE's DIVA-A drag (`variational_drag` branch) and stock DIVA drag
(`get_tau_bx_diva_jac`, which returns $-\tfrac12(\beta_{\mathrm{eff},\ell}+\beta_{\mathrm{eff},c})u_f$).

---

## 5. Why lumping breaks self-adjointness *here* (and not in general)

From §3, lumping a symmetric matrix stays symmetric. So if $\beta_{\mathrm{eff}}$ were a **fixed field**,
*both* discretizations in §4 would be self-adjoint, and there would be nothing to discuss.

The asymmetry comes from the one extra fact flagged in §1: **$\beta_{\mathrm{eff}}$ is state-dependent.**
The DIVA closure ties it to the cell-centre speed,

$$
\beta_{\mathrm{eff},c}=\beta_{\mathrm{eff}}\big(|\bar{\mathbf u}_c|\big),
$$

so linearizing the drag picks up a **coefficient-response** term $\partial\beta_{\mathrm{eff}}/\partial u$.

- **Consistent (DIVA-A):** both the coefficient and the velocity are functions of the same $\bar u$. The
  linearization is a genuine Hessian $\partial^2 D_{b,h}/\partial u_f\partial u_g$ — symmetric under
  $f\leftrightarrow g$.
- **Lumped (stock):** the velocity is now the facet value $u_f$, but the coefficient still tracks $\bar u$.
  The response term is

$$
\frac{\partial(\tau_b)_f}{\partial u_g}\;\supset\;
\tfrac12\,u_f\,\frac{\partial(\beta_{\mathrm{eff},\ell}+\beta_{\mathrm{eff},c})}{\partial u_g}
\;=\;\tfrac12\,u_f\;\big(\text{a cell-speed sensitivity in }u_g\big),
$$

a **facet** velocity $u_f$ multiplied by a **cell-speed** sensitivity — not symmetric under $f\leftrightarrow g$.

So the asymmetry is the **interaction** of mass-lumping with the state-dependent closure, not either alone.
In the language of [stress_balance_variational_structure.md](/home/bizon/glaciers/notes/stress_balance_variational_structure.md)
§3, this is exactly the "closure path" $\partial R/\partial C\cdot\partial C/\partial Q\cdot\partial Q/\partial x$
that has no matching transpose: the consistent drag is precisely what turns that diagnostic elimination into
the **symmetric Schur complement** $\nabla^2\Phi$ of §10 of the column-energy note; the lumped drag is the
elimination that is *not* that Schur complement. (Full discrete symmetry also needs a single regularization
floor for the membrane and shear invariants — the separate `eps_reg`/`eps_reg_shear` condition noted in the
companion notes — but the drag quadrature is the piece at issue here.)

---

## 6. Consequences (what actually differs)

- **Same continuum operator, same limit.** Both are consistent discretizations of the same drag. Their
  difference is a quadrature (lumping) error $\sim\tfrac12\beta(\bar u-u_f)\sim O(h\,\partial u/\partial x)$,
  which the elliptic solve smooths to $O(h^2)$ in the *solution*. Measured on Greenland: depth-averaged
  velocity agrees to $\sim\!10^{-4}$ at $0.9$ km and $\sim\!1\%$ at $7.2$ km, the gap concentrated at fast
  margins (large $\partial u/\partial x$) and vanishing in smooth interior ice.
- **Self-adjointness.** Consistent (DIVA-A) is self-adjoint to round-off ($\langle Jx,y\rangle$ vs
  $\langle Jy,x\rangle$ $\sim\!10^{-7}$); lumped (stock) carries an antisymmetric Jacobian part $\sim\!10^{-3}$
  at moderate driving, growing to $\sim\!10^{-2}$ as the drag/deformation grows.
- **Solver behaviour (the practical catch).** The consistent drag is **non-local** — it couples a facet to
  its neighbours through $\bar u$. A **local** block smoother (GLIDE's FAS–Vanka) freezes the outer facet the
  drag reaches, so the nonlinear forward solve stalls at the ice margin, worse as the grid coarsens
  (converges $\lesssim\!2$ km, stalls at $\sim\!10$ km). The lumped drag is **local** (diagonal), so the same
  smoother converges at every resolution. This is the price of structure preservation against a *local*
  smoother, not intrinsic ill-conditioning: a global (Krylov) solver would converge the consistent operator,
  and its true converged solution is $\sim\!1\%$ from stock (the spurious margin blow-up at $\sim\!10$ km is
  the *stalled iterate*, not the solution). See `stress_balance_comparison/gl_mg_convergence.py`,
  `gl_l3_solution.py`, `gl_residual_probe.py`.

---

## 7. Bottom line

- The DIVA stress-balance operator is $\mathbf R=\nabla\Phi$ for a dissipation potential $\Phi$; its basal
  drag is $\delta D_b/\delta\bar{\mathbf u}=\beta_{\mathrm{eff}}\bar{\mathbf u}$, with
  $\beta_{\mathrm{eff}}=\beta_{\mathrm{eff}}(|\bar{\mathbf u}|)$ set by the vertical closure.
- **Stock DIVA and self-adjoint DIVA-A are the mass-lumped and consistent discretizations of that one drag
  term.** Everything else — membrane, driving, closure — is identical.
- **Mass lumping** replaces the drag's symmetric, neighbour-coupling mass matrix by its diagonal row-sum.
  On its own it preserves symmetry; combined with DIVA's state-dependent $\beta_{\mathrm{eff}}(\bar u)$ it
  introduces an $O(h^2)$ antisymmetric Jacobian term — the entire source of stock DIVA's non-self-adjointness.
- Therefore "numerically self-adjoint" is a **property of the discretization** (a structure-preserving vs a
  lumped quadrature), *not* of the underlying variational principle, which both inherit from the continuum.
  Stock DIVA is a mass-lumped — hence $O(h^2)$ non-self-adjoint — discretization of a variational operator;
  the structure-preserving version (DIVA-A) sits next to it at the same cost, and coincides with it as the
  grid is refined.
