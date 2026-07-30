# DIVA in GLIDE: mathematics and numerical method

**Branch:** `diva` (off `sliding-laws`) · **Status:** forward model, exact adjoint, all three sliding-parameter gradients, and both nested closure solves implemented and verified; ISMIP-HOM validation outstanding · **Companion:** `notes/diva.md` (the design proposal and staged plan)

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
U_b <- stored u_b, clamped to [0, U_bar]            (warm start; the root is unique)
repeat until converged, cap 20:                     NEWTON on F(U_b) = 0
    tau_b = c(U_b)*U_b
    for k in 0..N_sigma-1:                          the vertical quadrature
        eta_k <- min(shear-free, shear-dominated) asymptote     (both overestimate)
        repeat until converged, cap 10:             NEWTON on eta = G(eta)
            s = k_shear/eta_k^2;  E = eps_mem^2 + eps_reg + s
            G = 0.5*B*E^((1-n)/2n);  G' = -2p*(s/E)*(G/eta_k)
            eta_k -= (eta_k - G)/(1 - G')
        accumulate eta_bar, F2, and dF2/dtau_b
    (breaks only right after a quadrature, so the outputs match the final U_b)
    F  = U_b + c(U_b)*U_b*F2 - U_bar
    F' = 1 + f'*F2 + f*f'*dF2/dtau_b                the FULL slope -- see 5.2.0
    U_b = max(U_b - F/F', 0)
write eta_bar, F2, u_b, beta_eff = c/(1 + c*F2)
```

Note the quadrature sits INSIDE the Newton loop, because $F_2$ depends on $U_b$. There is no
outer "coupling" iteration; an earlier version had one and it 2-cycled (§5.2.0).

Design choices worth reviewing:

- **Adaptive iteration counts with caps, not fixed counts.** See §5.2.3 -- fixed counts sized by a scaling argument are what let both closure defects hide, and a count validated at $n=3$ was silently wrong at $n=4$.
- **Only interior (non-halo) threads write.** `u_b` is read in place as the warm start, so each cell must have exactly one writer or the result would depend on block scheduling. This is what makes the kernel deterministic.
- **Grounding is folded into `beta_eff` here** (via $\beta\cdot\phi$). The momentum kernels must therefore *not* apply the grounded factor again — unlike the SSA basal stencils, which do it internally. This asymmetry is deliberate and is the thing most likely to be mis-edited later; `tests/diva_residual_test.py` pins it.

### 5.2.0 The kernel as a root-finding problem

The clearest way to read `diva_coeffs_cell`. The SSA-shaped operator (Goldberg eqs 43–44) needs two
coefficients per cell, $\bar\eta$ and $\beta_{\mathrm{eff}}$, and the momentum solve can only offer
$\bar U$. The kernel inverts that. **There is one genuinely free scalar** — the partition of $\bar U$
into sliding and internal deformation — and one equation fixing it:

$$F(U_b) \;=\; U_b \;+\; f(U_b)\,F_2\big(f(U_b)\big) \;-\; \bar U \;=\; 0$$

Constitutive, $\tau_b = f(U_b) = c(U_b)U_b$:

$$c(U) = \begin{cases}\beta\varphi\,\big(U^2+u_{\mathrm{reg}}\big)^{\frac{m-1}{2}} + w_d & \text{Weertman}\\[6pt]\dfrac{\beta\varphi}{\sqrt{U^2+u_{\mathrm{reg}}}+u_c} + w_d & \text{regularized Coulomb}\end{cases}$$

Shear integral, dependent on $U_b$ only through $\tau_b$:

$$F_2(\tau_b) \;=\; H\!\int_0^1 \frac{\zeta^2}{\eta(\zeta;\tau_b)}\,\mathrm{d}\zeta \;\approx\; H\sum_{k=1}^{N_\sigma} w\,\frac{\zeta_k^2}{\eta(\zeta_k;\tau_b)},\qquad \zeta_k=\frac{k-\tfrac12}{N_\sigma},\;\; w=\frac1{N_\sigma}$$

and a nested root problem per level, which is why $F_2$ is not a closed form (§5.2.1):

$$G\big(\eta;\zeta,\tau_b\big) \;=\; \eta - \tfrac12 B\Big[\dot\varepsilon^2_{\mathrm{mem}} + \varepsilon_{\mathrm{reg}} + \Big(\frac{\tau_b\zeta}{2\eta}\Big)^{\!2}\Big]^{p} = 0, \qquad p=\frac{1-n}{2n}$$

Both outputs are then formulas:

$$\bar\eta = \sum_k w\,\eta(\zeta_k;\tau_b), \qquad \beta_{\mathrm{eff}} = \frac{c(U_b)}{1+c(U_b)F_2} \quad\text{(G-41)}$$

**The derivative is the whole story.**

$$F'(U_b) \;=\; \underbrace{1 + f'(U_b)F_2}_{\displaystyle R'(U_b)} \;+\; \underbrace{f(U_b)f'(U_b)\frac{\mathrm{d}F_2}{\mathrm{d}\tau_b}}_{\text{omitted by the code}}$$

Every term is non-negative ($f,f'\ge0$, $F_2>0$, and $\mathrm{d}F_2/\mathrm{d}\tau_b>0$ because more
drag means more shear means thinner ice), so $F'\ge 1$: **$F$ is strictly increasing, the root is
unique, and full Newton is unconditionally well conditioned.** The problem is benign.

**This is what the code now does.** It did not always: an earlier version ran block Gauss–Seidel,
with the closure Newton using $R'$ (holding $F_2$ frozen) and a `coupling_iters` loop refreshing
$F_2$ around it. That scheme's gain is exactly

$$\Phi'(\tau_b) \;=\; 1 - \frac{F'}{R'} \qquad\Longrightarrow\qquad |\Phi'|>1 \iff F' > 2R'$$

(verified against bisection-found roots to $10^{-11}$; see §5.2.1a for the table). So the mechanism
is simply that the iteration **under-estimates the slope of a monotone increasing function**, and
Newton with an under-estimated slope overshoots — by more than $2\times$, it oscillates. Our
configurations sat at $F'/R' = 1.013$, the omitted term being 1.4% of the total, which is why
nothing had ever visibly misbehaved -- stable by accident rather than by construction.

**Resolved** by using $F'$: true Newton, gain zero, quadratic, and `coupling_iters` gone rather
than wrapped in anything. $\mathrm{d}F_2/\mathrm{d}\tau_b$ is accumulated in the $\sigma$ loop from
the implicit function theorem on the converged level root, taken from primals as any Newton
denominator may be (verified against a finite difference of the exact quadrature: 2e-9).

Verified against the true root found by **bisection** — never by fixed-point iteration, which is
the thing that was under suspicion — from a single COLD call, in the regimes that used to 2-cycle:

| $\beta$ | $\bar\eta$ err | $F_2$ err | $U_b$ err | $\|F\|/\bar U$ |
|--------:|---------------:|----------:|----------:|---------------:|
| 0.02 | 1.1e-07 | 2.2e-07 | 1.7e-08 | 2.1e-08 |
| 0.1  | 3.4e-08 | 5.0e-07 | 3.9e-08 | 9.3e-08 |
| 0.5  | 2.8e-08 | 1.1e-07 | 8.3e-09 | 2.3e-08 |
| 2.0  | 4.2e-07 | 3.9e-07 | 2.1e-07 | 5.9e-07 |

and repeated calls now agree exactly, where the block iteration sat in a stable 2-cycle. Cost:
DIVA went from ~11% more expensive than SSA to 12.5%, and to **13.8%** once the iteration counts
became adaptive (§5.2.3).

One trap worth recording. `newton_iters` was first left at 4, and the result *looked* converged --
repeated calls were idempotent -- while sitting at a non-root, $U_b$ 53% high at $\beta=2$. That
is what a too-short Newton looks like from outside: the warm start is the previous answer, so
successive calls reproduce it. Idempotence alone is necessary, not sufficient; `diva_closure_test`
check 5 compares against the true root from COLD for exactly this reason.

### 5.2.0a The closure residual, and a correction about "two residuals"

Implemented alongside the Newton fix, and it corrects something stated too strongly earlier in
this note's history.

**The claim was: DIVA has two residuals and the solver monitors one, so it can report
convergence with the closure unsatisfied. That is wrong.** `compute_residual` calls
`compute_diva_coeffs` before evaluating (`operators.py`), so the reported $|r_u|,|r_v|,|r_H|$
are the residuals of the FULL DIVA system with coefficients consistent with the current
velocity — not of a frozen-coefficient surrogate. The refresh-before-residual ordering already
couples the two halves.

What the closure residual

$$r_{U_b} = U_b + f(U_b)F_2 - |\bar U|$$

does add is worth having, but it is two other things:

1. **Coefficient drift.** $\bar\eta$ and $\beta_{\mathrm{eff}}$ ARE frozen inside a V-cycle,
   so measuring $r_{U_b}$ *before* the refresh says how far they drifted out of consistency
   while the smoother worked, i.e. whether the segregated refresh keeps up. Measured *after* the
   refresh it is zero by construction and says nothing — a trap worth noting, since that is
   where it naturally lands if you add it to the reporting line without thinking.
2. **A standing guard that the closure solve converges at all.** This is what regressed silently
   before: the block iteration 2-cycled, so `u_b` and $F_2$ were simply wrong, and no test or
   diagnostic noticed.

Measured on the 128² 5-level slab after the Newton fix: $|r_{U_b}|/|\bar U|$ = 3.4e-6 after the
first V-cycle and ~2e-8 thereafter. So the coefficients track the velocity to round-off and the
segregated refresh is comfortably keeping up in this configuration — which is the evidence the
earlier §5.2.1a caveat asked for, at least here.

Reported separately and scaled by $|\bar U|$, never folded into the combined norm: $r_{U_b}$ is a
velocity residual and $r_u$ a momentum one, and mixing incommensurable units in one norm is
precisely the defect in `notes/open_questions.md` Q6.

### 5.2.1 The per-level viscosity solve, and why Picard was not enough

The innermost loop above solves a genuine scalar fixed point. At depth $\zeta$, Glen's law and
the DIVA shear ansatz are two relations in $(\eta, \dot\varepsilon_{xz})$:

$$\eta = \tfrac12 B\big[\underbrace{\dot\varepsilon^2_{\mathrm{mem}} + \varepsilon_{\mathrm{reg}}}_{A} + \dot\varepsilon_{xz}^2\big]^{p},
\qquad p \equiv \frac{1-n}{2n} < 0,
\qquad \dot\varepsilon_{xz} = \frac{\tau_b\zeta}{2\eta}$$

Eliminating the shear rate leaves $\eta$ on both sides:

$$\eta = \tfrac12 B\Big[A + \frac{k}{\eta^2}\Big]^{p} \equiv G(\eta),
\qquad k \equiv \Big(\frac{\tau_b\zeta}{2}\Big)^2$$

**The contraction rate.** Differentiating and evaluating at the root ($G(\eta)=\eta$),

$$G'(\eta) = -2p\,\frac{s}{E}\,\frac{G(\eta)}{\eta}
\;\;\xrightarrow[\text{at the root}]{}\;\;
2|p|\frac{s}{E},
\qquad s \equiv \frac{k}{\eta^2},\; E \equiv A + s$$

Since $s \le E$ by construction, the Picard iteration $\eta \leftarrow G(\eta)$ is a contraction with

$$|G'| \le 2|p| = \frac{n-1}{n} = \tfrac23 \ \text{for Glen } n=3$$

**independent of $B$, $H$, $\tau_b$ and $\zeta$.** And because $s>0$ with $p<0$, the shear-free
starting value $\eta_0 = \tfrac12 BA^{p}$ *overestimates*, so the iteration descends monotonically.
That looks like a licence to fix the trip count at 3 — and it is not, because the bound is only
attained as $s/E \to 1$, which is precisely the shear-dominated regime DIVA exists to capture.
Three sweeps of a rate-$2/3$ contraction starting hundreds of times away from the root is not
convergence.

**Measured.** With the defaults ($\varepsilon_{\mathrm{reg}} = 10^{-6}$, $n_\sigma = 8$,
$\dot\varepsilon^2_{\mathrm{mem}} \approx 6\times10^{-9}$ for the slab tests, $H = 1000$ m), 3 sweeps
against a converged reference. $\tau_b$ is in code units, i.e. divided by $\rho g$, so 100 kPa
$\approx 11$:

| $\tau_b$ | $s/E$ | error in $\bar\eta$ | **error in $F_2$** |
|---------:|------:|--------------------:|-------------------:|
| 1.0 | 0.16 | 0.0% | 0.0% |
| 4.4 *(our test configurations)* | 0.98 | 4.6% | **15.9%** |
| 11.0 *(~100 kPa)* | 1.00 | 19.4% | **53.0%** |
| 30.0 | 1.00 | 33.7% | **74.5%** |

$F_2$ suffers most because it weights deep levels by $\zeta^2$ and the near-bed levels are the
least converged (35% error in $\eta$ at $\zeta=1$, $\tau_b=4.4$; **17** sweeps are needed there for
$10^{-3}$). $F_2$ too low means deformation underestimated, which drives DIVA back toward the SSA
answer — the exact effect it exists to correct.

**Why no test caught it.** `diva_closure_test` check 1, the strong one, sets $\tau_b = 0$, where
$s=0$ and Picard is exact in one step. Checks 2 and 3 validate the *Newton* against the
**computed** $F_2$, so they pass whatever $F_2$ happens to be. Nothing else compares DIVA to an
external reference — which is what ISMIP-HOM is for, and this is the kind of thing it would have
surfaced.

Note this was never a defect in the *adjoint*: the dual numbers differentiate the implemented map
exactly, so the dot-product identity and the parameter gradients remain correct statements about
the code as written. What was wrong was the fidelity of the implemented forward map to the DIVA
equations.

**The fix: Newton, not Picard.** For $n=3$ the fixed point is in fact a *depressed cubic* with an
exact root. Substituting $p=-\tfrac13$ and clearing denominators:

$$\eta = \tfrac12 B\Big(A+\frac{k}{\eta^2}\Big)^{-1/3}
\;\Longrightarrow\;
B^3 = 8A\eta^3 + 8k\eta
\;\Longrightarrow\;
8A\,\eta^3 + 8k\,\eta - B^3 = 0$$

No $\eta^2$ term, and $A,k,B^3>0$, so by Descartes exactly one positive real root — available in
closed form via Cardano. **We do not use it.** Cardano's two cube-root terms are
$\sqrt[3]{h\pm\sqrt{D}}$ with $D \approx (a_1/3)^3$ when the shear dominates, so they nearly cancel
in exactly the regime of interest. Verified in float32 against a float64 reference:

| $\tau_b$ | Cardano (float32) | Newton, 3 iters (float32) |
|---------:|------------------:|--------------------------:|
| 11.0 | 2.7e-6 | 2.0e-7 |
| 30.0 | 9.0e-5 | 0.0 |
| 100.0 | **1.9e-3** | 3.5e-7 |

So the implementation applies **Newton to $\Phi(\eta) = \eta - G(\eta)$**:

$$\eta \leftarrow \eta - \frac{\eta - G(\eta)}{1 - G'(\eta)},
\qquad G'(\eta) = 2|p|\,\frac{s}{E}\,\frac{G(\eta)}{\eta}$$

Three properties make this the right choice over both Picard and Cardano:

- **Well conditioned by construction.** $G' \in (0, 2|p|) \subset (0,1)$, so $1-G' \in (1/3, 1)$ and
  the Newton denominator is never near zero. No safeguarding needed.
- **General in $n$.** The expression for $G'$ holds for any $p$, so unlike the closed form there is
  no `n == 3` branch.
- **No new dual overloads.** It uses only $\times, +, \div$ and `__powf`, all of which `DualFloat`
  already has, so the derivative rides along exactly as before.

The starting guess matters, and takes the smaller of the two asymptotic limits — both of which
overestimate, since dropping either positive term in the cubic inflates $\eta$:

$$\eta_A = \tfrac12 BA^{p} \quad (\text{shear-free}),
\qquad \eta_k = \big(\tfrac12 Bk^{p}\big)^{n} \quad (\text{shear-dominated})$$

With $\min(\eta_A,\eta_k)$, **3 Newton iterations reach float32 round-off across the whole range**
($\le 2.6\times10^{-7}$ up to $\tau_b=100$) — the same trip count the Picard loop used. Starting from
$\eta_A$ alone would need 6.

### 5.2.1a RESOLVED: the outer coupling loop was unstable at high basal drag

**Historical.** The loop this describes no longer exists — §5.2.0 replaced it with true Newton on
$F(U_b)=0$, whose gain is zero. Kept because the analysis is the reason for that change, because it
records how the diagnosis was and was not done, and because the same failure mode can reappear in
any segregated scheme.

Fixing the per-level solve exposed a problem in the loop *around* it. Measured, not conjectured.

**The system.** Per cell the closure has $N+1$ unknowns — the viscosity at each $\sigma$ level and
the basal speed — with $N+1$ equations:

$$(E_k)\quad \eta_k = \tfrac12 B\Big[\dot\varepsilon^2_{\mathrm{mem}} + \varepsilon_{\mathrm{reg}} + \Big(\frac{\tau_b\zeta_k}{2\eta_k}\Big)^{\!2}\Big]^{p}, \qquad k = 1\dots N$$

$$(C)\quad U_b + f(U_b)\,F_2 = \bar U$$

with $\tau_b = f(U_b)$, $F_2 = H\sum_k w\,\zeta_k^2/\eta_k$ and $\bar\eta = \sum_k w\,\eta_k$ all
*derived*. The coupling structure is what makes the algorithm possible: the $\eta_k$ interact only
through the scalar $\tau_b$, and $U_b$ sees the $\eta_k$ only through the scalar $F_2$. So the
whole system reduces to **one scalar fixed point**, $\tau_b = \Phi(\tau_b)$, where $\Phi$ is
"solve every $E_k$, form $F_2$, solve $C$, return $f(U_b)$". The outer loop is plain Picard on it.

**Why it can oscillate.** Every link has a fixed sign, from the physics:

$$\tau_b \uparrow \Rightarrow \dot\varepsilon_{xz}\uparrow \Rightarrow \eta\downarrow \Rightarrow F_2\uparrow,
\qquad F_2\uparrow \Rightarrow U_b\downarrow \ (\bar U \text{ fixed}),
\qquad U_b\downarrow \Rightarrow \tau_b'\downarrow$$

so $\Phi'(\tau_b) < 0$ **always** — the loop is negative feedback. It converges (alternating about
the root) when $|\Phi'|<1$ and settles into a **2-cycle** when $|\Phi'|>1$. Physically it is a
competition between sliding and deformation for a fixed budget $\bar U$: guess high drag and the
model reports lots of deformation, so little sliding is needed, so drag is low; guess low drag and
the reverse. It only damps if the gain is under one.

**Measured.** The root must be found by **bisection** on $\tau_b - \Phi(\tau_b)$, which is
guaranteed since $\Phi$ decreasing makes it strictly increasing. (An earlier attempt used
fixed-point iteration to locate the root and differentiated around whatever the 400th sweep
returned. That was invalid — the iteration is the thing under suspicion, and it was oscillating,
so it was never at a fixed point. The numbers below replace it.)

| configuration | $\beta$ | true $\tau_b^*$ | $\Phi'$ | |
|---|---:|---:|---:|---|
| closure test ($B=1$, soft) | 0.02 | 0.160 | −0.256 | converges |
| closure test ($B=1$, soft) | 0.1 | 0.263 | **−1.382** | oscillates |
| closure test ($B=1$, soft) | 2.0 | 0.290 | **−1.890** | oscillates |
| **realistic $B$, our slab tests** | 0.11 | 2.12 | **−0.013** | converges hard |
| realistic $B$, high drag | 1.0 | 7.54 | **−1.210** | oscillates |
| realistic $B$, thick, slow, high drag | 5.0 | 3.30 | −0.866 | converges |
| realistic $B$, soft membrane | 1.0 | 7.85 | **−1.645** | oscillates |

It is **not** an artefact of the soft-ice test configuration: the *local* map is unstable at
realistic stiffness whenever the basal drag is high. That $\tau_b^*=7.54$ is ~68 kPa with a 7.5/20
sliding–deformation split, an ordinary Greenland condition.

Directly observed in the kernel: in the soft configuration at $\beta=0.1$, repeated
`compute_diva_coeffs` calls (warm-starting `u_b` from the previous, as the solver does) leave
`u_b` alternating between $\approx 0.71$ and $\approx 5.75$, and $F_2$ by a factor of ~18, with no
sign of damping by call 12.

**IMPORTANT CAVEAT: this analysis holds $\bar U$ FIXED.** In the solver it is not — $\bar U$ responds
to $\beta_{\mathrm{eff}}$ through the momentum solve, an additional feedback path excluded here. So
$|\Phi'|>1$ establishes that *the local, $\bar U$-frozen map* 2-cycles; it does **not** establish that
the coupled solver oscillates. Settling that needs a stability analysis of the full system, which
has not been done.

**And GLIDE's local loop is a deviation from Goldberg, not a reproduction of it.** His scheme
(paper, following eq 42) is a *single* fixed-point iteration over the velocity field: from
$\bar u^{(i)}$ diagnose $\nu^{(i)}, \omega^{(i)}, \beta^{(i)}_{\mathrm{eff}}$, solve the linear 2-D
system (43–44) for $\bar u^{(i+1)}$, and then — in his words — "$\tau_x^{(i+1)}$ is set to
$\beta^{(i)}_{\mathrm{eff}}\bar u^{(i+1)}$, and $u_z^{(i+1)}$ is found from Equation (31) using
$\nu^{(i)}_{\mathrm{(hy)}}$". So $\tau$ and $u_z$ are **lagged across the outer loop** and the local
diagnosis is a single pass. There is no local sub-iteration to oscillate. `coupling_iters` was ours,
and is now gone (§5.2.0) -- so on this point we have converged back onto Goldberg's structure, with
the local closure solved properly rather than swept.

This also inverts part of the earlier recommendation: *more* coupling sweeps make the between-call
behaviour worse, not better, since three sweeps carry gain $(\Phi')^3 = -2.6$ where one carries
$-1.38$. What survives is the Newton argument — a robustly solved local closure hands the outer
iteration a well-defined function of $\bar U$ whatever the gain, which beats any Picard sweep count
and is what Goldberg's single lagged pass is implicitly relying on being well behaved.

**Why nothing had broken.** Our configurations sat at $\Phi' = -0.013$, three orders of magnitude
inside the local stability boundary, so every DIVA solve in the suite converged. The margin was real
but it was luck rather than design — which is why this was fixed before ISMIP-HOM rather than after,
those experiments being deliberately deformation-dominated. Jake's call, and the right one: no
configuration can be called validated while the closure is stable only by accident.

**The fix, as taken.** Superseded by the cleaner form in §5.2.0 — Newton on $F(U_b)=0$ directly,
which removes the outer loop instead of wrapping it — but the argument that got there was: since
$\Phi'<0$ always, $g(\tau_b) = \tau_b - \Phi(\tau_b)$ has
$g' = 1 + |\Phi'| \ge 1$: **unconditionally well conditioned**. Newton on $g$ converges in every
row of that table, quadratically — the same move as §5.2.1, one level out. Fixed under-relaxation
would also stabilise it but is the wrong trade: $\omega = 0.5$ rescues $\Phi'=-1.9$ while
*degrading* our present regime from 0.013 to 0.49. And $\Phi'$ can be had the way the closure
Newton already gets $f'$ — a primal-only dual evaluation, valid because the converged sensitivity
does not depend on the step size used to reach it — so it needs no nested duals.

Implemented in the cleaner form of §5.2.0. Note that `diva_closure_test` check 5 had to change
with it: it previously replicated the kernel's own coupling structure, which made it a
self-consistency check that would have PASSED this 2-cycle. It now targets the true root by
bisection, and check 6 asserts idempotence.

### 5.2.1b Two places where we knowingly differ from Goldberg

- **The bed-slope factor.** Goldberg carries $m = \sqrt{1 + b_x^2 + b_y^2}$ through his eqs (38)–(41);
  GLIDE has no such factor, i.e. assumes small bed slopes. Worth quantifying before ISMIP-HOM,
  whose topographic experiments deliberately impose slopes. Note the name collision: Goldberg's $m$
  is geometric, ours is the Weertman exponent.
- **A typo in the paper — CONFIRMED, and we follow the correct one.** Eq (40) gives the frozen-bed
  limit as $\tau_x = (H/\omega)\bar u$, i.e. $\beta_{\mathrm{eff}} = H/\omega = 1/F_2$; the text
  introducing (42) gives $H/(2\omega)$, a factor of 2 apart. Jake confirmed both appear in the
  typeset PDF, so it is not an extraction artefact.

  (40) is the correct one, and it is derivable rather than a matter of preference: eq (34) reads
  $u|_{z=b} = \bar u - \tau_x\omega/H$, so a frozen bed ($u|_{z=b}=0$) gives $\tau_x = H\bar u/\omega$
  directly. (42) cannot be reconciled with (34).

  We implement (40): $\beta_{\mathrm{eff}} = c/(1+cF_2) \to 1/F_2 = H/\omega$ as $c\to\infty$, pinned
  by `diva_closure_test` check 1. So nothing to change — but worth knowing before comparing any
  frozen-bed result against a published figure, in case the figure used (42).

### 5.2.3 Adaptive iteration, and the three things it took to get right

Both nested solves now run to a tolerance with a cap, rather than a fixed count. The motivation is
Jake's, and it is the standing directive applied to iteration counts: a fixed count is an
assumption that holds in the regimes you tested, a tolerance is a guarantee. Both defects above
were silent, and an adaptive loop would have exposed each of them immediately.

Three things had to be right, and only the first was obvious.

**1. Non-convergence must be observable.** An adaptive loop that silently caps is *worse* than a
fixed count, because it looks converged. So `diva_coeffs_cell` returns a per-cell `cap_flags`
bitmask, `compute_diva_coeffs` writes it, and the solver reports any nonzero count next to
$|r_{U_b}|$. `diva_closure_test` asserts it is zero across its whole matrix. This earned its keep
within minutes of being added: it immediately flagged that at $n=4$ the per-level solve was
hitting its cap in every cell.

**2. Terminate on stagnation, not only on a tolerance.** These kernels use `__powf` under
`--use_fast_math`, so $G(\eta)$ carries a few ulp of error and the Newton correction does not go
to zero -- it enters a limit cycle. Measured at $n=4$, the relative step alternates
2.1e-7 / 4.3e-7 indefinitely. A pure tolerance therefore has to be tuned *above* a floor that is
itself parameter-dependent, which is the same kind of tested-regime number the whole exercise is
meant to eliminate. Accepting "the correction stopped decreasing" is floor-agnostic. It is armed
only once the correction is already below 1e-4 relative, so an early non-monotone step from a poor
start cannot trip it.

**3. The criterion is on the value, but the ADJOINT needs the derivative converged too.** This was
the flagged risk that turned out to be real. Breaking the instant the primal criterion fires
leaves the seeded `.d` one step stale, and the DIVA JVP degraded against finite differences from
2.3e-4 to **2.3e-3** while the SSA control sat unchanged at 2.33e-4 -- the localisation that says
it is DIVA's derivative, not the harness.

The fix is one extra iteration after the criterion fires, and it is exact rather than a safety
margin. The Newton map $N$ has $N'(\text{root}) = 0$, so once the value sits at the root a further
step leaves it there while replacing the derivative with $\mathrm{d}(\text{root})/\mathrm{d}(\text{seed})$
exactly. With it, JVP vs FD is back to 2.352e-4 and the dot-product identity to 3.75e-7.

**Cost, and a prediction that was wrong.** I predicted adaptivity would take DIVA from 12.5% over
SSA back to ~11%, reasoning that the closure loop would drop from 8 iterations to 1--2 when
warm-started (which it does -- $|r_{U_b}| \approx 2$e-8 between refreshes means the warm start is
essentially at the root). It went to **13.8%** instead. The reason is instructive: the $\eta$ loop
now does *more* work than its fixed 3, because 3 was under-converged. The saving on one loop was
more than offset by the other loop finally doing the work it should always have done. The old
number was partly cheap because it was wrong.

Measured worst cases, now known rather than assumed: the deepest $\sigma$ level at $n=4$ needs 7--8
$\eta$ iterations (the old fixed 3 left it at **2.8e-3** relative error), and the cold-start closure
Newton needs more than 12 at $n=4$, $\beta=2$. Typical counts are far lower -- most levels exit at
2--5, and a warm-started closure exits at 1--2.

### 5.2.2 Methodology note: how this was found

Worth recording, because the finding was a by-product rather than the object of a search, and the
sequence generalises.

1. **A claim was made in prose** — a comment asserting that 3 sweeps suffice, resting on the
   contraction bound $2|p| = 2/3$.
2. **The claim was checked numerically before being repeated.** A 20-line NumPy replication of the
   iteration confirmed the algebra (predicted vs observed contraction ratios matched, monotone
   descent held, the bound was attained) — and in doing so swept parameters more widely than any
   test did.
3. **An outlier in that sweep was chased rather than dismissed.** One random draw showed 233% error
   after 3 sweeps. The tempting reading is "unphysical parameters".
4. **The regime was re-tested with the code's real defaults**, not random ones: `eps_reg` read from
   `grid.py`, $\tau_b$ converted to code units from a physical basal stress. The error survived, and
   sat squarely in the range our own tests run at.
5. **The effect was propagated to the quantities that matter.** Per-level $\eta$ error is not the
   headline; $\bar\eta$ and $F_2$ are what the momentum balance sees, and $F_2$ turned out to be
   hit ~3x harder by the $\zeta^2$ weighting.
6. **The candidate fix was itself checked in float32 before being adopted**, which is what rejected
   Cardano — it is exact in float64 and loses three digits in float32 in the regime of interest.

A seventh step belongs on the list, and it is the one this session got wrong twice: **when the
scaffolding built to measure something is itself an iterative scheme, verify that it converged
before trusting what it reports.** The loop-gain measurement in 5.2.1a was invalid for exactly
that reason -- 400 Picard sweeps were assumed to be a fixed point and were an oscillation. The
same error, in a different costume, produced the retracted Q6.

The transferable part: steps 2 and 6. An analytic bound is a statement about the limit, not about
the truncation actually shipped, and a closed form is a statement about exact arithmetic, not about
float32. Both needed a measurement to become claims about this code. See also
`notes/open_questions.md` Q6, where reasoning from a plausible mechanism instead of measuring first
produced a retracted entry.

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

This is a real design constraint rather than a curiosity, so it is recorded here explicitly. It was previously implicit in the code and untested.

**Where symmetry is and is not relied on.** GLIDE's adjoint is a hybrid:

| site | mechanism | needs symmetry |
|------------------------|------------------------|------------------------|
| adjoint block solve (`vanka_smooth_adjoint`) | builds forward `J`, then **explicitly transposes** it (`J_T[r*5+c] = J[c*5+r]`) | no |
| adjoint multigrid transfers | reuses the forward operators | no -- multigrid only accelerates; the fixed point is set by the residual equation |
| basal / driving / flux / calving in `compute_vjp` | explicit scatter, `atomicAdd(adj_X, lambda_row * j.d_X)`, deposited at the **column** index | no |
| **viscous (membrane) term in `compute_vjp`** | λ-seeded forward JVP -- `populate_viscosity(..., lambda_u, lambda_v, ...)`, then `eta_c.d` fed through `apply_jvp`, deposited at the **row** index | **yes** |

So exactly one term computes `J λ` and uses it where `Jᵀ λ` is wanted. The tell is where the result lands: the basal block scatters to column indices, the membrane block deposits at its own row.

**Why the shortcut is there.** `∂η/∂u` is the worst transpose in the code to write by hand: the shared `eta_local` tile couples a 3×3 cell neighbourhood, each `η` depends on \~8 velocity facets, and the shear terms contract four cells' viscosity derivatives at once -- roughly 32 velocity degrees of freedom feeding a single row. Exploiting symmetry lets the adjoint reuse the **forward data flow verbatim**, with λ substituted for the perturbation. The honest transpose is perfectly feasible matrix-free (form `w_c = (Gᵀλ)_c` per cell, then scatter `w_c · ∂η_c/∂u_j`), but it needs a second, differently-shaped phase: an extra shared tile, an extra `__syncthreads()`, and a scatter whose reach may exceed the current `halo = 1` tiling. The cost is code structure, not flops.

**Measured** (central FD on the assembled residual, so the measured Jacobian contains the full `∂η̄/∂u` and `∂β_eff/∂u` paths; relative asymmetry of `⟨Jx,y⟩` vs `⟨x,Jy⟩`):

| block                                    | SSA    | DIVA   |
|------------------------------------------|--------|--------|
| momentum diagonal                        | 5.0e-6 | 7.3e-5 |
| off-diagonal `u`--`H` (positive control) | 9.8e-2 | 9.8e-2 |

The control confirms the test detects real asymmetry, so the diagonal figures are meaningful. DIVA's momentum block is symmetric to three orders below a genuinely asymmetric one. DIVA is \~15x less symmetric than SSA, which is plausibly a small real asymmetry from the fixed-iteration closure not being exactly the gradient of anything; expect that as a floor in FD gradient checks.

**What preserves symmetry.** The whole higher-order ladder (SSA, DIVA, MOLHO, Blatter--Pattyn) is variationally derived, so the momentum block is symmetric by construction. Also **every isotropic sliding law, neural networks included**: for `tau_b = -c(s) u` with `s = |u|`,

$$\frac{\partial \tau_{b,i}}{\partial u_j} = -\Big[c(s)\,\delta_{ij} + \frac{c'(s)}{s}u_i u_j\Big]$$

and both terms are symmetric in `(i,j)` for *any* scalar `c`. The eigenvalues are `c + c's = f'(s)` along the flow and `c` across it, so **definiteness** (not symmetry) is what requires a monotone law -- the rate-weakening constraint documented in `differentiable_sliding_laws.md`. Two separate properties: symmetry licenses the adjoint shortcut, monotonicity licenses the solver.

Thermomechanical coupling does **not** threaten it: `B(T)` enters as a coefficient, so within the momentum block `eta` is still an isotropic function of the strain invariant. The new `∂r_u/∂T` and `∂r_T/∂u` blocks are off-diagonal and get explicit transposes, as the `H` blocks already do.

**What would break it:** anisotropic drag (an NN emitting a 2x2 tensor, or drag not antiparallel to `u`); thickness transport (already nonsymmetric, already explicit); thermal or hydrological couplings (off-diagonal, handle explicitly).

**Consequence for the DIVA adjoint.** `η̄` depends on velocity through two paths, and they are treated differently on purpose:

| path into `η̄` | stencil | mechanism | symmetry assumed |
|------------------|------------------|------------------|------------------|
| `ε̇²_mem(u)` | wide (3x3 cells) | λ-seeded JVP shortcut | yes -- but the *same* assumption SSA already makes, structurally guaranteed |
| closure, via `Ū_c` | **local, 4 facets** | explicit transpose scatter | **no** |

The closure path enters only through `Ū_c = |ū_c|`, which `compute_diva_coeffs` builds from the cell's own four facets, so its transpose is a small local scatter. The DIVA adjoint therefore introduces **no new symmetry assumption** beyond SSA's.

### 5.9 How the scheme compares to Goldberg (2011) and Arthern et al. (2015)

Both published DIVA implementations use the **same outer scheme**: Picard on the momentum
equations with $\bar\eta$ and $\beta_{\mathrm{eff}}$ lagged, re-diagnosed between solves. Arthern
converges it on the momentum residual "expressed as a fraction of the norm of $f$", which is what
our reported $|r|/|r_0|$ is. Nothing we do differs there.

Everything that differs is in the two **implicit local problems**, and this is where the papers are
thin — Jake's observation, and largely right, though Arthern is much more explicit than Goldberg.

**The implicit viscosity.** $\eta$ appears on both sides, because the vertical shear strain rate
$\dot\varepsilon_{xz} = \tau_b\zeta/(2\eta)$ depends on it. Arthern states the problem outright:

> "The reason that equation (3) is an implicit definition for viscosity $\eta$ is that the strain
> rates for vertical shear themselves depend on viscosity."

and gives the resolution in one sentence:

> "...given estimates of $s, h, B, \tau_{bx}, \tau_{by}, \partial_x\bar u, \dots$ the viscosity
> $\eta$ can be found by **solving a cubic equation** obtained by substituting equations (5) and (4)
> into equation (3) and rearranging. Integration over depth is then carried out by numerical
> quadrature."

That cubic is exactly the one derived independently in §5.2.1, $8A\eta^3 + 8k\eta - B^3 = 0$ — a
useful confirmation of the algebra, arrived at from the other direction. **Goldberg does not solve
it at all**: he lags it, computing $u_z^{(i+1)}$ from his eq (31) using $\nu^{(i)}$, so the
viscosity advances one Picard step per momentum iteration and is never locally converged.

**The sliding closure.** Note what Arthern's sentence takes as *given*: $\tau_{bx},\tau_{by}$. So the
cubic is solved with $\tau_b$ **lagged** — the viscosity and the closure are not resolved against
each other. Goldberg's eqs (38)–(39) are the local root find, and he is explicit that it is
cell-local ("solved at a location along the base independently of other locations"), but in his
actual scheme $\tau$ is likewise set after the momentum solve from the previous $\beta_{\mathrm{eff}}$.

|  | implicit $\eta(\zeta)$ | closure for $U_b$, $\tau_b$ | outer |
|---|---|---|---|
| **Goldberg 2011** | lagged, one Picard step per momentum iteration | lagged | Picard on momentum |
| **Arthern 2015** | solved exactly: cubic for $n=3$, then quadrature | lagged (given to the cubic) | Picard, residual tolerance |
| **GLIDE (here)** | solved by Newton to tolerance | solved by Newton to tolerance, **jointly** with $\eta$ | multigrid + refresh; both residuals reported |

So we are strictly more converged locally than either, and the joint resolution is the part neither
paper does: our quadrature sits *inside* the closure Newton, so $\eta$, $F_2$ and $U_b$ are mutually
consistent before the coefficients are handed to the momentum solve. §5.2.0 is why that turned out
to matter — lagging $F_2$ against the closure is precisely the block Gauss–Seidel that 2-cycles at
high basal drag.

Two smaller divergences, both deliberate:

- **We use Newton on the cubic rather than Cardano**, despite having the closed form. Arthern
  presumably works in double precision, where Cardano is exact (verified: 2e-12). In float32 its two
  cube roots nearly cancel when the shear dominates and it loses three digits — 1.9e-3 at
  $\tau_b = 100$ against Newton's 3.5e-7 (§5.2.1). Newton is also $n$-general where the cubic is not.
- **We solve to a tolerance with stagnation detection, not a fixed sweep count** (§5.2.3). Neither
  paper says how many local iterations it takes, which is the detail that would have saved us the
  most time.

**The adjoint is where the divergence is largest.** Arthern does not compute a discrete adjoint at
all. He uses the Kohn–Vogelius functional with a Neumann/Dirichlet pair (Arthern & Gudmundsson
2010): solve the forward problem twice under different upper-surface boundary conditions and the
gradient falls out of the difference. That buys a gradient without an adjoint, at the cost of being
frankly approximate — his own words on the viscosity in the Dirichlet solve:

> "We do not recompute viscosity $\eta(z)$ but simply reuse the viscosity field that was computed
> for the Neumann solution. This choice is somewhat heuristic but practically convenient."

and on the stiffness update: *"Strictly, this applies only for a linear rheology. Nevertheless, at
each iteration, we updated the ice stiffness coefficient $B$ heuristically as follows..."*

That is entirely reasonable for a fixed-point inversion scheme, and it is the opposite of what we
need. Our exact coefficient adjoint — dot-product identity 3.75e-7, no symmetry assumed, every
closure path differentiated — exists because the gradient has to be consumed by a general optimizer,
and because a learned sliding law needs a derivative that is exact by construction rather than by
regime.

**Worth noting for the sliding-law work.** Arthern's headline conclusion is that
*"no simple sliding law adequately represents basal shear stress as a function of sliding speed"* --
his recovered basal drag varies by factors exceeding $10^{10}$ and resists any $\tau_b(u_b)$ fit.
That is an argument from the observational side for the learned-closure direction, from an author
with no stake in it.

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

## 7. Status and what remains

Done and verified (see `tests/` for each):

- **The forward model.** Converges as well as SSA, **13.8%** more expensive, consistent across multigrid depths. SSA is bit-identical with `stress_balance = 0`.

- **The closure solves properly.** Both nested root problems are Newton with the full slope: the per-level viscosity (§5.2.1) and the $U_b$ partition (§5.2.0). Verified against roots found by bisection from a cold start, agreement 5e-7, in the regimes where the previous block iteration 2-cycled. `diva_closure_test` checks 5 and 6 pin it; the closure residual $|r_{U_b}|/|\bar U|$ is reported every V-cycle (§5.2.0a) and reads ~2e-8.

- **The exact coefficient adjoint.** `AdjointOperators.diva_exact_coeff_adjoint = True`, on by default. Dot-product identity $\langle J^T\lambda, x\rangle$ vs $\langle\lambda, Jx\rangle$ = **5.6e-7** against an SSA control of 1.2e-7, with **no symmetry assumed anywhere** in the DIVA path. `vanka_smooth_adjoint` is templated on DIVA and uses the uncondensed block; the smoother stays frozen-coefficient, which is fine because it is only a preconditioner -- exactly as in SSA, where the VJP carries $\partial\eta/\partial u$ and the smoother does not.

  An earlier attempt appeared to stall the adjoint V-cycles at 1.1e-1. The cause was not the smoother but the **Dirichlet rows**: the coefficient gather ran after the main VJP kernel had replaced those rows with an identity, so what it added there could never be reduced and sat in the residual as a floor. Flat *and* omega-independent is the signature of that, as against a weak preconditioner, which responds to omega.

- **All three sliding-parameter gradients**, $\partial J/\partial\beta$, $\partial J/\partial u_c$ and $\partial J/\partial m$, as one cell-local expression:

```         
  dJ/d(p)_c = W_eta_c * d(eta_bar_c)/d(p_c) + W_be_c * d(beta_eff_c)/d(p_c)
```

  `W_eta` and `W_be`, filled by `vjp_body`, already *are* $\lambda^T\,\partial r/\partial(\text{coefficient})$ summed over every row that touches the cell, so no facet loop is involved and nothing in the expression knows which parameter $p$ is -- the identity of $p$ lives entirely in which pair of derivative fields is passed. One kernel serves $\beta$ and $u_c$; a reducing variant serves the global $m$. Every input a gradient could be wanted for is templated in `get_diva_c_of_U`, so adding a parameter means seeding it and reading `.d`, with no second derivation to keep in sync. Contrast SSA, which needs a separate ~90-line facet-walking kernel per parameter because there the parameters enter the momentum stencils directly rather than through a state-dependent coefficient.

  Against finite differences (best of a bracketed step sweep):

  | | SSA control | DIVA |
  |---|---|---|
  | $\partial J/\partial\beta$ | 1.4e-4 | **2.2e-5** |
  | $\partial J/\partial u_c$ | 2.6e-4 | **1.6e-4** |
  | $\partial J/\partial m$ | 9.5e-5 | **5.5e-5** |

  DIVA is at or better than the exact-adjoint SSA control throughout, i.e. all six are limited by the finite-difference reference rather than by the adjoint.

  Two implementation details that are easy to get wrong. The $\beta$ seeding perturbs by `grounded`, not 1, because the closure is handed $\beta\varphi$ -- otherwise the result is $\partial/\partial(\beta\varphi)$ rather than the $\partial/\partial\beta$ an inversion controls. And $m$ needs `__powf(dual, dual)`: it sits in the *exponent*, so $d(x^p) = x^p(\tfrac{p}{x}dx + \log x\,dp)$ and the existing `(dual, float)` overload cannot supply the second term.

- **Methodological note, worth keeping.** $\partial J/\partial\beta$ read 9.6e-4 with a *wrong* (frozen) adjoint and 2.2e-2 with the right one, because a wrong $\lambda$ was partly cancelling the missing `eta_bar` path. Neither number was evidence about the gradient on its own. Relatedly, no single FD step is trustworthy here: two-sided truncation falls like $\varepsilon^2$ while float32 round-off grows like $1/\varepsilon$, so the tests sweep $\varepsilon$ to bracket the crossover and print the whole curve. $\partial J/\partial u_c$ under DIVA reads 1.6e-4 at $\varepsilon = 2$ and 1.4e-3 at $\varepsilon = 1$; judging on one step would have manufactured a defect that is not there.

Remaining, in order:

- **ISMIP-HOM validation -- required, not optional.** Everything verified so far establishes internal consistency and the SSA limit; nothing yet compares DIVA against an external reference. Goldberg runs experiment C and the nonlinear-sliding cases, and reproducing those figures is the acceptance gate for this branch. The forward model and the gradients are both in place now, so they can be validated together.

- **Quadrature-order convergence in** $N_\sigma$. The default is 8 with midpoint quadrature; nothing yet establishes what that costs against, say, 32.

- **Thermomechanical** $B(z)$, which would use the vertical discretisation already here.

- **Quadrature order**: midpoint is first-cut; Goldberg's factor conventions are pinned but the rule itself has not been convergence-tested in $N_\sigma$.