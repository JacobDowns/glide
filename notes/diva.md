# Design doc: DIVA higher-order stress balance (augmented 6×6, dual-differentiated)

**Status:** proposal · **Branch:** `diva` (off `sliding-laws`) · **Scope:** `glide` core (CUDA + adjoint plumbing)

**Primary reference:** Goldberg (2011), *A variationally derived, depth-integrated approximation to a higher-order glaciological flow model*, J. Glaciol. 57(201) — PDF in `docs/`. Equation numbers below refer to it. Secondary: Arthern et al. (2015), Lipscomb et al. (2019, CISM).

## Summary

GLIDE solves the **SSA** (plug-flow) stress balance: horizontal velocity is depth-independent, so all resistance to the driving stress comes from membrane stretching + basal drag, with no vertical shear. This adds the **DIVA** approximation (depth-integrated viscosity approximation) as a **non-destructive, opt-in** higher-order option.

DIVA is the right first higher-order scheme for GLIDE because it **reuses SSA's exact 2D elliptic operator** — the primary unknowns stay $\bar u,\bar v$ (depth-averaged velocity) and $H$. Only two coefficients change *definition*:

1. $\bar\eta$ — the membrane viscosity becomes **depth-averaged and shear-aware** (the effective strain rate gains vertical-shear terms, so $\eta$ varies with depth);
2. $\beta_{\mathrm{eff}}$ — the basal drag the 2D solve feels becomes an **effective** friction that accounts for the shear partition between basal sliding and internal deformation.

Goldberg says this directly: *"a glacial flow model that reliably solves the SSA equations with a given sliding law can be easily modified to solve the hybrid momentum balance,"* and the discretized system is *"trivially more expensive"* than SSA with an identical sparsity pattern (eqs 43–44). That is exactly the fit we're exploiting.

**Governing principle (non-destructive):** an `approximation` selector defaults to `"ssa"` → the current code path, byte-for-byte. DIVA is selected explicitly; its extra field(s) and the vertical quadrature are only allocated/run when chosen. This mirrors the `sliding_law` selector added on the `sliding-laws` branch.

**Motivation.** (1) The deformation/sliding partition may materially change the sliding-law inversions this branch's parent (`sliding-laws`) is built for — an inverted $\beta$ under SSA absorbs shear that DIVA would attribute to deformation, contaminating interior friction. (2) DIVA yields a reconstructable **3D velocity field**, needed to couple the thermal solver for paleo runs.

## Variational structure and self-adjointness

DIVA is **derived from a variational principle** (Goldberg 2011): the approximation is made *to the action functional* $\mathcal L$ (eq 12), not to the equations, and the stress balance is recovered as its Euler–Lagrange equation. Goldberg states the consequence explicitly (p. 3):

> *"the resulting equation set is self-adjoint (i.e. ignoring dependence of viscosity on strain rate)."*

This resolves the concern Doug raised (DIVA "not obviously self-adjoint"): it **is** self-adjoint — at *exactly the frozen-viscosity level GLIDE's smoother already operates at* (§"How this grounds…"). The `∂η/∂u` term that the self-adjointness statement excludes is precisely the piece GLIDE already carries separately in the dual/adjoint path, not in the smoother. So the variational derivation guarantees a clean, symmetric linearized operator to build the adjoint from.

The sliding law enters $\mathcal L$ as a **basal dissipation potential** $\int_{\Gamma_b} F(u_b)\,d\Gamma$ (eq 12), with the traction its derivative, $\tau_b = f(u_b) = F'(u_b)$. Because *any* monotone law is the gradient of such a potential, **a nonlinear sliding law does not break the variational/self-adjoint structure** — it is just a different $F$. Goldberg treats linear and regularized-Coulomb laws on the same footing.

So the earlier worry about nonlinear laws is downgraded: the nonlinearity is a *computational* wrinkle (a per-cell drag equation, below), **not** a structural obstruction, and our augmented-$U_b$ + dual closure (§Design) makes adding laws mechanical. We still adopt Doug's discipline — **co-develop the adjoint with the forward model** — because it's good practice and keeps the closure differentiable, not because the adjoint is in doubt. (Doug is implementing MOLHO separately — a variationally-clean cousin; coordinate on shared vertical/thermal scaffolding.)

## What changes, physically (SSA → DIVA)

SSA effective strain rate (`viscosity.cu:110`, membrane only; $\eta \equiv$ Goldberg's $\nu$):

$$
\dot\varepsilon_e^2 = \dot\varepsilon_{xx}^2 + \dot\varepsilon_{yy}^2 + \dot\varepsilon_{xx}\dot\varepsilon_{yy} + \dot\varepsilon_{xy}^2 + \varepsilon_{\mathrm{reg}},
\qquad
\eta = \tfrac12\,B\,\dot\varepsilon_e^{(1-n)/n}\quad(n=3)
$$

DIVA adds vertical shear to the effective strain rate (eq 16):

$$
\dot\varepsilon_e^2 \;\mathrel{+}=\; \dot\varepsilon_{xz}^2 + \dot\varepsilon_{yz}^2,
\qquad
\dot\varepsilon_{xz}(z) = \frac{\tau_b\,(s-z)}{2\,\eta(z)\,H}
$$

The shear ansatz is that the shear stress is linear in depth, $\tau_{xz}(z) = \tau_b\,(s-z)/H$ (basal drag at the bed, zero at the surface; eq 31). This makes $\eta$ depend on $z$ → a **vertical discretization** is required. The membrane term uses the depth average $\bar\eta = \tfrac1H\int_b^s \eta\,dz$.

**Isothermal is enough for stage 1.** Even with $B$ uniform in $z$, $\eta(z)$ still varies with depth because $\dot\varepsilon_{xz}\propto(s-z)/H$. So the DIVA vertical-shear effect is captured with **no temperature field** — $B(z)$ (thermomechanical coupling; fork branch `thermal`) is a strictly later stage.

## The sliding closure — a per-cell drag equation

DIVA supplies a **kinematic** relation, independent of the sliding law, between the depth-averaged and basal velocities (integrating the shear ansatz twice; eqs 32–34):

$$
\bar{\mathbf u} = \mathbf u_b + \tau_b\,F_2,
\qquad
F_2 = \int_b^s \frac{1}{\eta}\Big(\frac{s-z}{H}\Big)^2 dz \;=\; \frac{\omega}{H}
$$

where $\omega = \int_b^s\!\int_b^z \frac{s-z'}{H\,\eta}\,dz'\,dz$ is Goldberg's kernel (eq 35); integration by parts gives $\omega = H F_2$, which **pins our convention with no stray factor**.

The **sliding law** supplies a second, independent relation at the bed, $\tau_b = f(u_b)$, with $f$ acting on the **basal** velocity (this branch's Weertman $\beta|u_b|^m$, regularized-Coulomb $\tau_{\max}|u_b|/(|u_b|+u_c)$, or an NN). Eliminating $\tau_b$:

$$
\bar{\mathbf u} = \mathbf u_b + f(u_b)\,F_2
\qquad\Longleftrightarrow\qquad
R(u_b) := u_b + f(u_b)\,F_2 - \bar{\mathbf u} = 0
$$

- **Linear** $f=\beta u_b$: closed form $u_b = \bar u/(1+\beta F_2)$, $\beta_{\mathrm{eff}} = \beta/(1+\beta F_2)$.
- **Nonlinear** $f$: a **per-cell root-find** for $u_b$ — Goldberg's eqs 38–39, *"solved at a location along the base independently of other locations… done column by column and thus easily parallelized."*

Two facts make the nonlinear case cheap and robust:

- **It's scalar.** Isotropic drag ($\tau_b\parallel u_b$) + isotropic viscosity ⇒ everything is collinear with $\bar{\mathbf u}$ (Goldberg: *"$\vec\tau$ will always be in the same direction as $(\bar u,\bar v)$"*). The direction is inherited and the unknown is the **basal speed** $U_b=|\mathbf u_b|$: solve $\bar U = U_b + f(U_b)\,F_2$, then $\mathbf u_b = (U_b/\bar U)\,\bar{\mathbf u}$.
- **It's well-posed.** Physical laws are monotone ($f'\ge0$) and $F_2>0$, so $R'(U_b)=1+f'(U_b)F_2>0$: unique root, Newton converges in ~2–4 warm-started iters.

Goldberg's effective drag (eq 41), with $c(U_b)\equiv f(U_b)/U_b$ the secant sliding coefficient and the bed-slope factor $\approx 1$:

$$
\beta_{\mathrm{eff}} = \frac{c(U_b)}{1 + c(U_b)\,F_2} \;=\; \frac{\tau_b}{\bar U}\;\ge 0,
\qquad
\tau_b = \beta_{\mathrm{eff}}\,\bar U
$$

which reduces to $\beta/(1+\beta F_2)$ for a linear law. **The no-sliding limit is handled gracefully:** as $U_b\to0$ (frozen bed), $\beta_{\mathrm{eff}}\to 1/F_2 = H/\omega$ (eq 40) — finite deformational resistance, where SSA would give $\beta_{\mathrm{eff}}=0$ and lose all interior traction. This is the whole point of DIVA over SSA in the interior.

### Secant vs. tangent (an implementation detail, not a landmine)

The **residual** uses the *secant* $\beta_{\mathrm{eff}}$ above; the **Jacobian and adjoint** need the *tangent*:

$$
\beta_{\mathrm{eff}}^{\mathrm{tangent}} = \frac{\partial\tau_b}{\partial\bar U} = \frac{f'(U_b)}{1 + f'(U_b)\,F_2}
$$

Equal for linear $f$; different for nonlinear $f$ (secant uses $c=f/U_b$, tangent uses $f'$). Both come out of the same closure, so producing both is free once we differentiate it — the point is just to not accidentally reuse the secant in the Jacobian. The parameter gradient carries the same denominator:

$$
\frac{\partial\tau_b}{\partial\theta} = \frac{f_\theta(U_b)}{1 + f'(U_b)\,F_2},
\qquad \theta \in \{\beta,\,m,\,\tau_{\max},\,u_c,\,\text{NN weights}\}
$$

All three are **closed-form via the implicit function theorem** at the converged $U_b$ and frozen $F_2$ — the adjoint never re-runs the per-cell Newton in reverse.

## How this grounds onto GLIDE's solver

GLIDE's nonlinear solve is a tripartite pattern, and DIVA slots into it rather than replacing it — mirroring Goldberg's own "iteration on viscosity" scheme (eqs 41–44: diagnose $\bar\eta,\omega,\beta_{\mathrm{eff}}$ from the iterate, solve the SSA-sparsity linear system, repeat):

- **Smoother (Vanka block):** viscosity is *frozen* — `build_5x5_vanka` (`vanka.cu:91`) assembles a local Jacobian with $\eta$ as a frozen tile, refreshed between sweeps. This is exactly the level at which Goldberg's self-adjointness holds. DIVA freezes $\beta_{\mathrm{eff}},\bar\eta$ here the same way.
- **Residual:** uses the true $\eta$ (and, for DIVA, the true secant $\beta_{\mathrm{eff}}$).
- **JVP / adjoint:** carries the exact coefficient–velocity coupling via the `DualFloat` dual-number path; $\partial\eta/\partial u$ (and, for DIVA, $\partial\beta_{\mathrm{eff}}/\partial u$) live here, *not* in the smoother. The adjoint solve transposes the local block (`lu_5x5_solve` on $J^{\mathsf T}$, `vanka.cu:800`).

The DIVA auxiliaries ($\eta_k$ over levels, $U_b$) are **cell-local** — no spatial coupling — so they are eliminated locally, never carried through the multigrid transfer operators.

## Design — augmented 6×6 block + dual-differentiated closure

Promote the basal speed $U_b$ to a **6th, cell-centered unknown** (peer of $H$), with one extra **local constraint row**:

$$
R_{U_b} = U_b + f(U_b)\,F_2 - |\bar{\mathbf u}| = 0
$$

The local Vanka system grows $5\times5 \to 6\times6$. The existing 5 unknowns are the four velocity facets + $H$; $U_b$ is the sixth, cell-centered like $H$, coupling to the four facet velocities exactly as $H$ does:

- $\partial(\text{momentum})/\partial U_b$ — the basal-drag rows depend on $U_b$ through $\beta_{\mathrm{eff}}$ (cell-centered, averaged to facets like $\beta$ is today).
- $\partial R_{U_b}/\partial(u,v\ \text{facets})$ — through $|\bar{\mathbf u}|$.
- $\partial R_{U_b}/\partial U_b = 1 + f'(U_b)\,F_2$ — the diagonal (always $>0$).

**Write $R_{U_b}$ in dual arithmetic.** Because the closure is *new code we own* (unlike the legacy hand-derived stress Jacobians), express $R_{U_b}$ and $\beta_{\mathrm{eff}}$ with `DualFloat` so their partials — for the 6×6 block **and** the transposed adjoint block — fall out automatically, **for every sliding law $f$**, amortizing the per-law derivation across the Weertman / Coulomb / NN zoo. This is the honest realization of "dualize the local law," and it is what makes the many-laws goal cheap.

Cost: extend `DualFloat` (`common.cu:4`, currently `+ − *  /scalar  __powf`) with $\div(\text{dual},\text{dual})$ and $\sqrt{\ }(\text{dual})$ overloads — a few lines each, one-time. Caveat: duals give the *local partials*; the cross-terms still have to be **wired into the 6×6 $J$** (and its transpose) by hand — dualization kills the derivation, not the assembly.

The LU solve extension is trivial: `lu_5x5_solve` (Doolittle, no pivot, `vanka.cu:5`) becomes `lu_6x6_solve` by adding one forward/back-substitution row. The block size is the *free* part.

## Vertical discretization

- $N_z$ sigma levels $\zeta_k=(s-z_k)/H\in[0,1]$ (default $N_z=8$), quadrature weights $w_k$.
- Integrals (convention pinned above via $\omega=HF_2$; cross-check Lipscomb 2019): $\bar\eta \approx \sum_k w_k\,\eta_k$ and $F_2 \approx \sum_k w_k\,\zeta_k^2/\eta_k$. $F_1$ only if the $u(z)$ reconstruction is needed (thermal coupling).
- **Storage stays 2D.** A single `compute_diva_coeffs` kernel loops $k=0..N_z$ *internally* (with a short inner $\eta_k$ fixed-point, since $\dot\varepsilon_{xz}\propto 1/\eta_k$), integrates, runs the $U_b$ closure, and writes 2D fields $\bar\eta$, $F_2$, $U_b$, $\beta_{\mathrm{eff}}$ (secant + tangent). No 3D $\eta$ array is stored. Genuine novelty is concentrated here — everything else is plumbing.

## Adjoint

Because the auxiliaries are cell-local and closed-form-differentiable, the adjoint is the transpose of the augmented 6×6 (dual-generated), with $\bar\eta$, $F_2$ frozen at their converged values — the same frozen-viscosity level at which Goldberg's operator is self-adjoint. The neglected $\partial F_2/\partial u$ cross-term is the only approximation; measure it with an FD check rather than assume it's zero. Parameter gradients reuse the existing dedicated contraction kernels (`compute_gradient_beta` etc., `operators.py:513`) with the closure chain factor $1/(1+f'F_2)$ applied; `model.backward` (`model.py:62`) gains the DIVA-mode gradients behind the same flag pattern as `compute_m_grad`/`compute_u_c_grad`.

## Staged plan

| stage | deliverable | files | risk |
|----------------|-------------------------|----------------|----------------|
| **0** | `approximation` selector; SSA path byte-identical (regression vs. saved SSA solve) | `grid.py`, `operators.py`, `multigrid.py` | low |
| **1** | isothermal DIVA **forward + adjoint together**: `compute_diva_coeffs` (vertical quadrature + $U_b$ closure), $\bar\eta$/$\beta_{\mathrm{eff}}$ into the residual, 6×6 block, dual `DualFloat` ops, transposed adjoint, FD checks | `viscosity.cu`, `stress.cu`, `vanka.cu`, `common.cu`, `residuals.cu`, `grid.py`, `multigrid.py`, `operators.py`, `torch.py`, `tests/` | high (core smoother) |
| **2** | general sliding laws under the closure (Weertman-$m$ / Coulomb / NN) — verify the functor reuse + per-law $f, f', f_\theta$ | `stress.cu` / law functor | moderate |
| **3** (opt) | thermomechanical $B(z)$ | + `thermal` branch | later |

**Beachhead option.** If the 6×6 core surgery feels too broad to start, do a **segregated beachhead** first (this is literally Goldberg's own outer scheme, eqs 41–44): keep the existing frozen-coefficient 5×5 smoother, add only `compute_diva_coeffs` producing $\bar\eta,\beta_{\mathrm{eff}}^{\mathrm{secant}},\beta_{\mathrm{eff}}^{\mathrm{tangent}}$ as frozen fields (zero new unknowns), and take the adjoint via the IFT tangent (the known `u_c`/`m` drill). This proves the physics + closure + adjoint with minimal blast radius; then refactor to the augmented 6×6 once validated. The augmented form is the destination (mechanical per-law adjoint); the segregated form is the lower-risk on-ramp. Either way the forward and adjoint land together.

## Test plan

Extend the FD harness (`tests/grad_*`, `tests/jvp_test.py`, `tests/adjoint_test.py`): FD checks on $\partial\tau_b/\partial\theta$ through the closure for each law, and on the 6×6 block via a coupled residual/JVP consistency check. Validate the forward against: (a) **reduces to SSA** as $F_2\to0$ (fast sliding / floating), (b) **reduces to SIA** in the no-sliding, frozen-bed limit ($\beta_{\mathrm{eff}}\to1/F_2$), (c) **ISMIP-HOM** — Goldberg runs experiments C (sliding) and the nonlinear-sliding-law cases; reproduce his figures as the acceptance target. Gate on the `approximation="ssa"` default reproducing current results bit-for-bit.

## Open decisions (for the first session)

1. $N_z$ default (proposing 8) and per-level recompute vs. finest-only + restrict.
2. Augmented-6×6 from the start vs. segregated beachhead first.
3. ~~Exact $F_n$ factor conventions~~ — **pinned** via $\omega=HF_2$ (Goldberg eq 35); cross-check against Lipscomb 2019 during coding.

## Rollout

Land stage 0 (selector + SSA-identical guard), then stage 1 as a single forward+adjoint unit. Keep SSA as the reference oracle in the $F_2\to0$ limit. Share the vertical-quadrature / $u(z)$-reconstruction scaffolding with Doug's MOLHO work. Push `diva` to `fork`; PR into `origin` only after the adjoint FD checks pass.
