# DIVA in GLIDE: mathematics and numerical method

**Branch:** `diva` (off `sliding-laws`) · **Status:** forward model, exact adjoint and `dJ/d(beta)` all implemented and verified; ISMIP-HOM validation outstanding · **Companion:** `notes/diva.md` (the design proposal and staged plan)

**See also:** `notes/diva.md` (design and staged plan) and `notes/open_questions.md` (issues found in existing upstream code while doing this work -- not addressed here).

**Primary reference:** Goldberg, D. N. (2011), *A variationally derived, depth-integrated approximation to a higher-order glaciological flow model*, J. Glaciol. **57**(201), 157–170. PDF in the workspace `docs/`. Equation numbers of the form (G-nn) refer to it.

This note is written to be read next to the code. It states what DIVA is mathematically, then describes exactly how it is discretised and solved in GLIDE, with emphasis on **what is shared with the existing SSA path and what is genuinely new**.

------------------------------------------------------------------------

## 1. Where DIVA sits

The full-Stokes momentum balance is too expensive for continental-scale, long-timescale or inverse work. The standard ladder of approximations is:

| model | horizontal velocity | resistance | cost |
|------------------|------------------|------------------|------------------|
| **SIA** | none solved (diagnostic) | vertical shear only | trivial |
| **SSA** (GLIDE today) | depth-independent, 2-D PDE | membrane + basal drag | one 2-D solve |
| **DIVA** (this branch) | depth-averaged 2-D PDE + reconstructed profile | membrane + vertical shear + basal drag | one 2-D solve + cell-local work |
| **Blatter–Pattyn (first order)** | full 3-D field | all but vertical normal stresses | 3-D solve |
| **full Stokes** | full 3-D field | everything | 3-D saddle point |

The first-order (Blatter–Pattyn) equations are

$$\partial_x\big[\nu(4u_x + 2v_y)\big] + \partial_y\big[\nu(v_x+u_y)\big] + \partial_z(\nu u_z) = \rho g s_x$$

and its $y$ counterpart (G-1,2), with the effective viscosity (G-3)

$$\nu = \frac{B}{2}\Big[u_x^2 + v_y^2 + u_xv_y + \tfrac14(u_y+v_x)^2 + \tfrac14 u_z^2 + \tfrac14 v_z^2\Big]^{\frac{1-n}{2n}}$$

Solving this needs a 3-D PDE system because the **horizontal** stress terms ($u_x,u_y,v_x,v_y$) vary with depth.

**The DIVA approximation** is to replace $u,v$ by their depth averages $\bar u,\bar v$ *in the horizontal stress terms only*, while retaining the vertical shear terms $u_z,v_z$ in both the viscosity and the basal boundary condition. The approximation is made **to the action functional, not to the equations** (G-12), and the equations follow as its Euler–Lagrange equations. Two consequences matter here:

1.  the approximated equations and boundary conditions are guaranteed mutually consistent;
2.  the resulting operator is **self-adjoint**, in Goldberg's words *"ignoring dependence of viscosity on strain rate"* — which is exactly the level at which GLIDE's smoother already freezes viscosity (§5.5). The sliding law enters the functional as a basal dissipation potential $\int_{\Gamma_b}F(u_b)\,d\Gamma$, whose derivative is the traction, so a **nonlinear** sliding law is just a different $F$ and does not break the variational structure.

------------------------------------------------------------------------

## 2. The equations DIVA solves

### 2.1 Momentum

Identical in form to SSA — this is the reason DIVA fits GLIDE so cheaply (G-43):

$$\partial_x\big[H\bar\eta(4\bar u_x + 2\bar v_y)\big]
+ \partial_y\big[H\bar\eta(\bar v_x + \bar u_y)\big]
- \tau_{bx} = \rho g H s_x$$

Only two coefficients change meaning: the viscosity $\bar\eta$ is now a **depth average of a depth-varying** $\eta$, and the basal drag $\tau_b$ is an **effective** drag. Everything else — the stencil, the sparsity pattern, the boundary conditions — is SSA's.

### 2.2 Depth-varying viscosity

With the horizontal terms depth-averaged, the effective strain-rate invariant gains the vertical shear contributions (G-16):

$$\eta(z) = \frac{B}{2}\Big[\underbrace{\bar u_x^2 + \bar v_y^2 + \bar u_x\bar v_y + \tfrac14(\bar u_y+\bar v_x)^2}_{\dot\varepsilon^2_{\mathrm{mem}}} + \dot\varepsilon_{xz}^2 + \dot\varepsilon_{yz}^2 + \varepsilon_{\mathrm{reg}}\Big]^{\frac{1-n}{2n}}$$

The braced group is the membrane invariant $\dot\varepsilon^2_{\mathrm{mem}}$ — all that SSA has. The two shear terms are what DIVA adds.

### 2.3 The shear ansatz

Integrating the vertical-shear part of the momentum balance once gives a shear stress that is **linear in depth** — the full basal drag at the bed, zero at the stress-free surface (G-31):

$$\tau_{xz}(z) = \tau_{bx}\,\frac{s-z}{H},
\qquad\text{so}\qquad
\dot\varepsilon_{xz}(z) = \tfrac12 u_z = \frac{\tau_{bx}}{2\eta(z)}\,\zeta ,
\qquad \zeta \equiv \frac{s-z}{H}\in[0,1].$$

Note the thickness **cancels** in $\dot\varepsilon_{xz}$; $H$ survives only as a prefactor in $F_2$ below. This is why the scheme degrades gracefully as $H\to0$.

### 2.4 The shear integral $F_2$

Integrating a second time and depth-averaging relates the depth-averaged and basal velocities (G-32–35):

$$\bar{\mathbf u} = \mathbf u_b + \tau_b F_2$$

$$F_2 \equiv \int_b^s \frac{1}{\eta}\Big(\frac{s-z}{H}\Big)^2 dz
= H\int_0^1 \frac{\zeta^2}{\eta}\,d\zeta$$

In words: **depth-averaged velocity = sliding + internal deformation**, with the deformation proportional to the basal drag through $F_2$. Goldberg writes this with $\omega$ (G-35), the double integral of $(s-z')/(H\eta)$. The two are related by $\omega = H F_2$, and that identity pins our convention with no stray factors. This relation is **independent of the sliding law**.

### 2.5 The sliding closure

The sliding law supplies a second, independent relation at the bed, $|\tau_b| = f(U_b) = c(U_b)\,U_b$, where $U_b=|\mathbf u_b|$ and $c$ is the drag *coefficient*:

| law                 | $c(U)$                                          |
|---------------------|-------------------------------------------------|
| Weertman            | $\beta\,(U^2+u_{\mathrm{reg}})^{(m-1)/2} + w_d$ |
| regularized Coulomb | $\beta/(\sqrt{U^2+u_{\mathrm{reg}}}+u_c) + w_d$ |

Because drag and viscosity are isotropic, $\tau_b \parallel \mathbf u_b \parallel
\bar{\mathbf u}$ (Goldberg: *"*$\vec\tau$ will always be in the same direction as $(\bar u,\bar v)$"), so the vector relation collapses to a **scalar equation for the basal speed**:

$$R(U_b) = U_b + f(U_b)\,F_2 - \bar U = 0$$

- **Linear** $f$: closed form, $U_b = \bar U/(1+\beta F_2)$.
- **Nonlinear** $f$: a per-cell root find — Goldberg's (G-38,39), *"solved at a location along the base independently of other locations."*

It is well posed: physical laws are monotone ($f'\ge0$) and $F_2>0$, so $R'(U_b) = 1 + f'(U_b)F_2 \ge 1 > 0$. The root is **unique** and Newton converges in a few iterations from any start.

### 2.6 The effective drag

$$\tau_b = \beta_{\mathrm{eff}}\,\bar U,
\qquad
\beta_{\mathrm{eff}} = \frac{c(U_b)}{1 + c(U_b)F_2}
\qquad \text{(G-41)}$$

Three properties worth noting:

- it reduces to $\beta/(1+\beta F_2)$ for a linear law;
- it is **strictly non-negative**, so the drag remains a dissipative (coercive) term in the elliptic operator — the property the momentum solve depends on;
- in the **frozen-bed limit** $U_b\to0$ it tends to $1/F_2 = H/\omega$ (G-40), a *finite* deformational resistance, where SSA would give $\beta_{\mathrm{eff}}=0$ and lose all interior traction. This is the main physical gain of DIVA over SSA.

**Implementation note.** We work with the coefficient $c(U)$ rather than the drag $f(U)=c U$ precisely so that $\beta_{\mathrm{eff}}$ needs no division by $\bar U$, which would be $0/0$ in every stagnant or ice-free cell.

### 2.7 Secant versus tangent

The residual uses the **secant** drag above. The Jacobian needs the **tangent**:

$$\frac{\partial\tau_b}{\partial\bar U} = \frac{f'(U_b)}{1+f'(U_b)F_2}
\qquad\text{vs.}\qquad
\beta_{\mathrm{eff}} = \frac{c(U_b)}{1+c(U_b)F_2}.$$

They coincide for a linear law ($c$ constant $\Rightarrow f'=c$) and differ for any nonlinear one. Getting this right is the single most error-prone part of the scheme; §5.6 explains how the implementation obtains the tangent without deriving it by hand.

------------------------------------------------------------------------

## 3. SSA versus DIVA at a glance

|   | SSA (`stress_balance = 0`) | DIVA (`stress_balance = 1`) |
|------------------------|------------------------|------------------------|
| primary unknowns | $u,v,H$ | **unchanged** |
| operator stencil / sparsity | 5-point membrane + drag | **unchanged** |
| viscosity | $\eta(\bar u)$, explicit formula, computed **inline** | $\bar\eta$: depth average of an **implicit** $\eta(z)$, computed by a **separate kernel** and read as a field |
| vertical structure | none (plug flow) | $N_\sigma$ sigma levels, integrated away per cell |
| basal drag | sliding law evaluated on $\bar u$ | effective drag $\beta_{\mathrm{eff}}$ from the closure, evaluated on $\bar u$ |
| basal speed | $=\bar u$ by assumption | separate quantity $U_b\le\bar U$ from the closure |
| extra state | — | `u_b`, `eta_bar`, `F2`, `beta_eff` (all cell-centred 2-D) |
| block size | 5 ($u_l,u_r,v_t,v_b,H$) | 6 (adds $U_b$), condensed back to 5 |
| cost | baseline | **≈ 11 %** more (measured, §6) |

The essential point: **DIVA changes coefficients, not structure.** The elliptic operator, the multigrid hierarchy, and the transfer operators are untouched.

------------------------------------------------------------------------

## 4. What is genuinely new, numerically

Two nonlinearities must be distinguished, because only one is new.

**(a) Viscosity depends on velocity — not new.** SSA already has Glen's-law shear thinning, and GLIDE already handles it with a well-established tripartite pattern: viscosity is *frozen* in the smoother block (Picard), *true* in the residual, and *exactly differentiated* in the JVP/adjoint via the `DualFloat` dual-number path. DIVA inherits this unchanged.

**(b) Viscosity is implicit in itself — new.** The shear ansatz gives $\dot\varepsilon_{xz} = \tau_b\zeta/(2\eta)$, so $\eta$ appears inside the expression for its own argument:

$$\eta_k = \tfrac12 B\Big[\dot\varepsilon^2_{\mathrm{mem}} + \big(\tfrac{\tau_b \zeta_k}{2\eta_k}\big)^2 + \varepsilon_{\mathrm{reg}}\Big]^{\frac{1-n}{2n}} .$$

There is no closed form for $n=3$, so this needs a local fixed-point iteration. SSA has no analogue. (There *is* precedent for iterating locally inside a kernel: the Vanka smoother already runs an inner Newton per block.)

**Structural consequence.** SSA computes $\eta$ *inline* inside every residual/Vanka/JVP kernel and never stores it. DIVA cannot — a vertical quadrature does not fit inside a per-stencil call — so $\bar\eta$ becomes a **stored, lagged field** refreshed before each evaluation. This is precisely Goldberg's "iteration on viscosity" (G-41–44), itself the classical SSA solution method of MacAyeal & Thomas (1986).

------------------------------------------------------------------------

## 5. The numerical method as implemented

### 5.1 Vertical discretisation

$N_\sigma$ sigma levels (default 8, `rheology.n_sigma`) with midpoint-rule quadrature in $\zeta\in[0,1]$:

$$\zeta_k = \frac{k+\tfrac12}{N_\sigma},\qquad w_k = \frac1{N_\sigma},\qquad
\bar\eta \approx \sum_k w_k\,\eta_k,\qquad
F_2 \approx H\sum_k \frac{w_k\,\zeta_k^2}{\eta_k}.$$

$\zeta=0$ is the surface, $\zeta=1$ the bed. **No 3-D array is stored**: the loop over $k$ lives inside one kernel and only its 2-D integrals are written out. `B` is currently uniform in $z$ (isothermal); note the vertical shear effect is captured *regardless*, because $\dot\varepsilon_{xz}\propto\zeta$ makes $\eta$ depth-varying even for constant `B`. Thermomechanical $B(z)$ is a later stage.

### 5.2 The coefficient kernel

`compute_diva_coeffs` (`cuda/diva.cu`) is the one genuinely novel kernel. Per cell:

```         
ThanU_b <- stored u_b, clamped to [0, U_bar]        (warm start; the root is unique)
repeat coupling_iters (3):
    for k in 0..N_sigma-1:
        eta_k <- membrane-only value            (this is exactly the SSA viscosity)
        repeat eta_iters (3):
            eps_shear = tau_b * zeta_k / (2*eta_k)
            eta_k     = 0.5*B*(eps_mem^2 + eps_shear^2 + eps_reg)^((1-n)/2n)
        accumulate eta_bar, F2
    Newton on R(U_b) = U_b + f(U_b)*F2 - U_bar  (newton_iters = 4)
    tau_b = c(U_b)*U_b
write eta_bar, F2, u_b, beta_eff = c/(1 + c*F2)
```

Design choices worth reviewing:

- **Fixed iteration counts, no convergence test.** Uniform cost per thread and no warp divergence; the outer nonlinear iteration absorbs the remainder.
- **Only interior (non-halo) threads write.** `u_b` is read in place as the warm start, so each cell must have exactly one writer or the result would depend on block scheduling. This is what makes the kernel deterministic.
- **Grounding is folded into `beta_eff` here** (via $\beta\cdot\phi$). The momentum kernels must therefore *not* apply the grounded factor again — unlike the SSA basal stencils, which do it internally. This asymmetry is deliberate and is the thing most likely to be mis-edited later; `tests/diva_residual_test.py` pins it.

### 5.3 Dual numbers for the sliding law

`get_diva_drag_coeff` (`cuda/stress.cu`) returns $c(U)$ as a `DualFloat`, so $c'(U)$ — and hence $f'=(cU)'$ by the product rule — falls out of the *same* evaluation. The same helper therefore serves the closure Newton, the block Jacobian, and (later) the adjoint, **for every sliding law, with no per-law hand derivation**. This required extending `DualFloat` with $\div$(dual,dual) and $\sqrt{\cdot}$(dual) — the only additions to the existing dual-number machinery.

### 5.4 Momentum residual

`residual_body<bool DIVA>` (`cuda/residuals.cu`) is shared by both schemes; `compute_residual` and `compute_residual_diva` are thin wrappers. `DIVA` is a compile-time flag, so each instantiation keeps only its own branch. The two differ in exactly two places:

1.  the $\eta$ tile is read from `eta_bar` instead of `populate_viscosity`;
2.  the basal term uses `get_tau_bx_diva_jac` / `get_tau_by_diva_jac` ($\tau_b=-\beta_{\mathrm{eff}}\bar u$, linear in velocity) instead of the sliding-law stencil.

### 5.5 The smoother

`vanka_smooth_body<bool DIVA>` and `build_5x5_vanka<bool DIVA, ...>` (`cuda/vanka.cu`) are shared the same way. The block owns $(u_l, u_r, v_t, v_b, H_c)$ — indices 0–4, row-major $J[5r+c]$ — and runs a damped local Newton with a $5\times5$ LU (Doolittle, no pivoting). Viscosity is frozen within the block; this is the level at which the DIVA operator is self-adjoint (§1).

### 5.6 The augmented $U_b$ unknown and its condensation

DIVA carries the basal speed as a **sixth local unknown** with the closure as its residual row, giving

$$\begin{bmatrix} A & \mathbf b \\ \mathbf c^{\mathsf T} & d \end{bmatrix}
\begin{bmatrix} \delta\mathbf x \\ \delta U_b\end{bmatrix}
=\begin{bmatrix} \mathbf r \\ r_{U_b}\end{bmatrix},
\qquad
\begin{aligned}
b_a &= \frac{\partial r_a}{\partial\beta_{\mathrm{eff}}}\cdot\frac{c'}{(1+cF_2)^2}\\
c_a &= -\frac{\partial \bar U}{\partial x_a}\\
d &= 1 + f'(U_b)F_2 \;\ge\; 1
\end{aligned}$$

Because $d$ is a **scalar and never singular**, $U_b$ is eliminated *analytically*:

$$\left(A - \frac{\mathbf b\,\mathbf c^{\mathsf T}}{d}\right)\delta\mathbf x
= \mathbf r - \frac{\mathbf b\, r_{U_b}}{d},
\qquad
\delta U_b = \frac{r_{U_b} - \mathbf c\cdot\delta\mathbf x}{d}.$$

This is **algebraically identical to solving the** $6\times6$ but leaves the hardcoded $5\times5$ layout and `lu_5x5_solve` untouched. It is what upgrades the secant drag that `build_5x5_vanka` assembled into the **tangent** of §2.7 — and note it is obtained *without ever deriving the tangent by hand*: the condensation performs the elimination numerically from two quantities the duals already give us.

Two documented approximations:

- $\mathbf c$ omits $\partial F_2/\partial H$, consistent with the lagged viscosity;
- $U_b$ is not written back from the block — `compute_diva_coeffs` re-solves the closure exactly before the next sweep, which is at least as good as one Newton step.

Sanity check: for a **linear** law $c'=0$, so $\mathbf b \equiv 0$ and the condensation is provably inert — secant and tangent coincide, as they must.

`lu_6x6_solve` exists in `cuda/vanka.cu` not for the hot path but as the **verification oracle**: `tests/diva_condensation_test.py` checks the condensed $5\times5$ against the explicit $6\times6$ on random systems.

### 5.7 Multigrid

Nothing in the FAS cycle changes. Each level diagnoses its own $\bar\eta, F_2,
\beta_{\mathrm{eff}}$ from its own restricted state, including for the coarse-grid operator evaluations $F_c(I u_h)$, so the coarse-grid correction is consistent. `u_b` is restricted with the rest of the state (giving the coarse closure a good warm start) and needs no prolongation, since it is diagnosed rather than corrected. The DIVA auxiliaries are all cell-local, so no new transfer operator is required.

------------------------------------------------------------------------

### 5.8 Operator symmetry: what the adjoint relies on

This is a real design constraint rather than a curiosity, so it is recorded here
explicitly. It was previously implicit in the code and untested.

**Where symmetry is and is not relied on.** GLIDE's adjoint is a hybrid:

| site | mechanism | needs symmetry |
|---|---|---|
| adjoint block solve (`vanka_smooth_adjoint`) | builds forward `J`, then **explicitly transposes** it (`J_T[r*5+c] = J[c*5+r]`) | no |
| adjoint multigrid transfers | reuses the forward operators | no -- multigrid only accelerates; the fixed point is set by the residual equation |
| basal / driving / flux / calving in `compute_vjp` | explicit scatter, `atomicAdd(adj_X, lambda_row * j.d_X)`, deposited at the **column** index | no |
| **viscous (membrane) term in `compute_vjp`** | λ-seeded forward JVP -- `populate_viscosity(..., lambda_u, lambda_v, ...)`, then `eta_c.d` fed through `apply_jvp`, deposited at the **row** index | **yes** |

So exactly one term computes `J λ` and uses it where `Jᵀ λ` is wanted. The tell is where
the result lands: the basal block scatters to column indices, the membrane block deposits
at its own row.

**Why the shortcut is there.** `∂η/∂u` is the worst transpose in the code to write by
hand: the shared `eta_local` tile couples a 3×3 cell neighbourhood, each `η` depends on
~8 velocity facets, and the shear terms contract four cells' viscosity derivatives at
once -- roughly 32 velocity degrees of freedom feeding a single row. Exploiting symmetry
lets the adjoint reuse the **forward data flow verbatim**, with λ substituted for the
perturbation. The honest transpose is perfectly feasible matrix-free (form
`w_c = (Gᵀλ)_c` per cell, then scatter `w_c · ∂η_c/∂u_j`), but it needs a second,
differently-shaped phase: an extra shared tile, an extra `__syncthreads()`, and a scatter
whose reach may exceed the current `halo = 1` tiling. The cost is code structure, not
flops.

**Measured** (central FD on the assembled residual, so the measured Jacobian contains
the full `∂η̄/∂u` and `∂β_eff/∂u` paths; relative asymmetry of `⟨Jx,y⟩` vs `⟨x,Jy⟩`):

| block | SSA | DIVA |
|---|---|---|
| momentum diagonal | 5.0e-6 | 7.3e-5 |
| off-diagonal `u`--`H` (positive control) | 9.8e-2 | 9.8e-2 |

The control confirms the test detects real asymmetry, so the diagonal figures are
meaningful. DIVA's momentum block is symmetric to three orders below a genuinely
asymmetric one. DIVA is ~15x less symmetric than SSA, which is plausibly a small real
asymmetry from the fixed-iteration closure not being exactly the gradient of anything;
expect that as a floor in FD gradient checks.

**What preserves symmetry.** The whole higher-order ladder (SSA, DIVA, MOLHO,
Blatter--Pattyn) is variationally derived, so the momentum block is symmetric by
construction. Also **every isotropic sliding law, neural networks included**: for
`tau_b = -c(s) u` with `s = |u|`,

$$
\frac{\partial \tau_{b,i}}{\partial u_j} = -\Big[c(s)\,\delta_{ij} + \frac{c'(s)}{s}u_i u_j\Big]
$$

and both terms are symmetric in `(i,j)` for *any* scalar `c`. The eigenvalues are
`c + c's = f'(s)` along the flow and `c` across it, so **definiteness** (not symmetry)
is what requires a monotone law -- the rate-weakening constraint documented in
`differentiable_sliding_laws.md`. Two separate properties: symmetry licenses the adjoint
shortcut, monotonicity licenses the solver.

Thermomechanical coupling does **not** threaten it: `B(T)` enters as a coefficient, so
within the momentum block `eta` is still an isotropic function of the strain invariant.
The new `∂r_u/∂T` and `∂r_T/∂u` blocks are off-diagonal and get explicit transposes, as
the `H` blocks already do.

**What would break it:** anisotropic drag (an NN emitting a 2x2 tensor, or drag not
antiparallel to `u`); thickness transport (already nonsymmetric, already explicit);
thermal or hydrological couplings (off-diagonal, handle explicitly).

**Consequence for the DIVA adjoint.** `η̄` depends on velocity through two paths, and
they are treated differently on purpose:

| path into `η̄` | stencil | mechanism | symmetry assumed |
|---|---|---|---|
| `ε̇²_mem(u)` | wide (3x3 cells) | λ-seeded JVP shortcut | yes -- but the *same* assumption SSA already makes, structurally guaranteed |
| closure, via `Ū_c` | **local, 4 facets** | explicit transpose scatter | **no** |

The closure path enters only through `Ū_c = |ū_c|`, which `compute_diva_coeffs` builds
from the cell's own four facets, so its transpose is a small local scatter. The DIVA
adjoint therefore introduces **no new symmetry assumption** beyond SSA's.

## 6. Verification status

| test | what it establishes |
|------------------------------------|------------------------------------|
| `ssa_regression_test.py` | SSA (both sliding laws) is **bit-identical** to the pre-DIVA reference. Run after every commit on this branch; it is what makes the shared-body refactors provably safe. |
| `diva_closure_test.py` | $\bar\eta$ collapses onto the analytic SSA viscosity with no shear; $F_2$ matches the quadrature to 1e-6; the Newton reproduces the linear closed form to 6 digits; the Coulomb closure residual is 2e-7; $\beta_{\mathrm{eff}}$ matches (G-41) and $\tau_b=\beta_{\mathrm{eff}}\bar U$ holds. |
| `diva_residual_test.py` | Handed the SSA coefficients, the DIVA residual reproduces the SSA residual **bit-for-bit** — pins the grounding/`water_drag` bookkeeping. |
| `diva_condensation_test.py` | The condensed $5\times5$ reproduces the full $6\times6$ to 1.1e-7; DIVA converges under regularized Coulomb. |
| `diva_solve_test.py` | End-to-end: DIVA converges as well as SSA ($1.0\times10^{-4}$ vs $9.9\times10^{-5}$), is 4.1 % faster than SSA (deformation on top of sliding), $0\le u_b\le\bar U$, $F_2>0$, deterministic, and agrees across multigrid depths to 2.2e-5. |

**Cost:** 0.81 s vs SSA's 0.73 s for ten V-cycles on a 128×128 5-level slab — about 11 % more, matching Goldberg's "trivially more expensive than SSA".

------------------------------------------------------------------------

## 7. Not yet done

- ~~**Complete `dJ/d(beta)`.**~~ **DONE.** `dJ/d(beta)` is now the cell-local product
`W_eta_c*d(eta_bar_c)/d(beta_c) + W_be_c*d(beta_eff_c)/d(beta_c)`, from a third dual
seeding of the closure (seeded with `grounded` so the result is w.r.t. the raw beta). It
agrees with finite differences at the FD floor -- best 2.2e-5, against an SSA control of
1.4e-4 -- and it *replaced* the facet-walking kernel with ~15 lines. The same expression
gives `u_c` and `m` by swapping which derivative field is used.

  Historical note worth keeping: with the frozen adjoint this read 9.6e-4 and with the
  exact adjoint 2.2e-2, because a wrong lambda had been partly cancelling the missing
  `eta_bar` path. Neither number was evidence about the gradient on its own.

- **`u_c` and `m` gradients.** Same shape as above. `u_c` and `m` are still plain floats
  in `diva_coeffs_cell`; templating them the way `beta` now is makes them seedable, and
  then each needs only its own pair of derivative fields plus one kernel line.

- ~~**Turn on the exact coefficient adjoint.**~~ The adjoint operator is now exact (dot-product identity
5.6e-7) and the adjoint solve converges, but the *parameter* sensitivity is not yet
complete: `get_diva_dbeta_eff_dbeta` covers only the `beta -> beta_eff` path, while beta
also moves `eta_bar` through `beta -> c -> tau_b -> shear term`. Against finite differences
`dJ/d(beta)` therefore reads 2.2e-2, having read 9.6e-4 with the frozen adjoint only
because a wrong lambda partly cancelled the missing term -- a good reminder that agreement
with FD is not evidence of correctness when two errors can offset.

The fix is simpler than the current kernel. `W_eta` and `W_be` already *are*
`lambda^T d(r)/d(coefficient)`, so the parameter gradient is a purely cell-local product
with no facet loop:

      dJ/d(beta)_c = W_eta_c * d(eta_bar_c)/d(beta_c) + W_be_c * d(beta_eff_c)/d(beta_c)

which needs one more dual seeding in compute_diva_derivs (seed beta) and replaces
compute_gradient_beta_diva entirely. The same shape then gives `u_c` and `m` for free.

- **Turn on the exact coefficient adjoint.** The four closure terms are implemented and
verified -- `AdjointOperators.diva_exact_coeff_adjoint = True` takes the adjoint identity
from 9.7e-2 to **5.0e-7**, i.e. round-off against an SSA control of 1.2e-7, with no
symmetry assumed. They are **on** by default. An earlier attempt appeared to stall the adjoint V-cycles at
1.1e-1; the cause was not the smoother but the Dirichlet rows -- the coefficient gather ran
after the main VJP kernel had replaced those rows with an identity, so what it added there
could never be reduced and sat in the residual as a floor. Skipping the Dirichlet facets in
the gather fixed it, and the adjoint now converges to 1.0e-6 exactly as the frozen version
does. The smoother is still the frozen block, which is fine: it is only a preconditioner,
exactly as in SSA where the VJP carries d(eta)/du and the smoother does not.

- **The adjoint.** `vanka_smooth_adjoint` is still SSA-only (it passes `nullptr` for the DIVA fields). This is the next piece of work and the condition under which the author endorsed the effort. The pieces are in place: `get_diva_dbeta_eff_du_b` already supplies $\partial\beta_{\mathrm{eff}}/\partial U_b$, and the parameter chain factor is $f_\theta(U_b)/(1+f'(U_b)F_2)$ — closed-form via the implicit function theorem at the converged $U_b$, so the adjoint never re-runs the per-cell Newton in reverse.
- **ISMIP-HOM validation -- required, not optional.** Everything verified so far establishes internal consistency and the SSA limit; nothing yet compares DIVA against an external reference. Goldberg runs experiment C and the nonlinear-sliding cases, and reproducing those figures is the acceptance gate for this branch. To be done once the adjoint is in, so the forward model and the gradients are validated together.
- **Thermomechanical** $B(z)$, which would use the vertical discretisation already here.
- **Quadrature order**: midpoint is first-cut; Goldberg's factor conventions are pinned but the rule itself has not been convergence-tested in $N_\sigma$.