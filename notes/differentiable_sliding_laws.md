# Design doc: extensible, fully-differentiable sliding laws

**Status:** proposal · **Branch:** `sliding-laws` (off canonical `main`) · **Scope:** `glide`
core (CUDA + adjoint plumbing)

## Summary

GLIDE's basal sliding law is hard-coded to a single Weertman power form, and the
adjoint exposes a gradient only w.r.t. the coefficient field `beta`. This adds, as a
**non-destructive, opt-in** extension:

1. a gradient w.r.t. the rate exponent `m` (so it can be *calibrated*, not just swept);
2. a small *sliding-law interface* so additional physically-parameterized laws (Budd,
   regularized-Coulomb / Schoof) are first-class, fully-differentiable citizens whose
   spatial parameter fields can be inverted for — or predicted by a neural network —
   exactly like `beta` is today.

**Governing principle (non-destructive):** the default code path stays byte-for-byte
unchanged. A `sliding_law` selector defaults to `"weertman"` → the current kernel
branch, verbatim. Existing configs, the rest of the stack, and the student VMs behave
identically. New laws are selected explicitly; their extra parameter fields are only
allocated/used when chosen; new adjoint kernels are added *alongside* the existing ones.

The headline payoff: a velocity-dependent friction law where **the NN predicts static
spatial parameter fields and the kernel owns the `|u|` dependence**. This eliminates the
outer Picard/Anderson loop a velocity-dependent law otherwise needs, and yields an
*exact* adjoint instead of the frozen-coefficient approximation the phase-3 prototype used.

## Motivation (empirical, from the phase-3 prototype)

A velocity-dependent inversion was prototyped *outside* the kernel (NN-predicted drag
`C(|u|, features)`, lagged through Picard/Anderson, reusing the existing `beta`-adjoint
at the converged state). It works, but the leave-one-basin-out study surfaced two limits
that a kernel-resident law removes:

1. **The outer loop is the weak link at extrapolation.** Holding out a fast NE basin
   (Mouginot 223), the flexible regularized-Coulomb law's *forward solve failed to
   converge* — Anderson oscillated at 4–10 % residual and the reported held-out error
   depended on where iteration stopped. Replacing it with a **rigid power law solved
   natively** made the same basin converge cleanly. The pathology was the outer
   iteration, not the physics.
2. **Pure Weertman is a poor vehicle for velocity dependence.** A sweep over a *global*
   exponent `m` showed `m=1` (linear) is robust but `m<1` is **freeze-prone**: the
   effective drag coefficient `C = beta·|u|^(m-1)` diverges as `|u|→0` (rate-weakening),
   so no `beta` cap simultaneously prevents fast-basin freeze and supplies interior drag.
   Robust velocity dependence needs a **bounded-drag** law (regularized-Coulomb,
   `C = tau_max/(|u|+u_c)`, which saturates) — which the power law can't express and the
   Picard prototype couldn't solve robustly. So it has to live in the kernel.

The native rigid law also *beat* the outer-loop flexible law on mean held-out error at a
fraction of the cost. The lesson isn't "velocity dependence doesn't help" — it's "put a
*bounded-drag* velocity dependence inside the solver."

## Current state of the code (canonical GLIDE)

- **The velocity Jacobian already exists and is already used.** The basal-stress Jacobian
  (`glide/cuda/stress.cu`, ~line 270) computes `jac.res` (the stress), `jac.d_u` (the rate
  dependence, line 271), and `jac.d_beta_l/d_beta_r` (lines 276–277). `d_u` drives the
  Newton/Vanka smoother *and*, via the transpose, the adjoint solve. So a velocity-dependent
  law is **already differentiated exactly at the fixed point**; there is no need to expose
  `dJ/du` (a state, not a control).
- **The adjoint is cleanly factored.** `model.backward()` solves the adjoint state
  (law-agnostic — it only needs `d_u`/`d_v`). Each invertible parameter then has a
  *dedicated contraction kernel*: `compute_gradient_beta` (`glide/operators.py:513`), with
  siblings `compute_gradient_bed/H_prev/smb` (lines 544/576/579). `GlideStep.backward`
  (`glide/torch.py:35`) reads `sliding.beta.grad` and returns it. **Adding a new invertible
  parameter is therefore additive** — a new `compute_gradient_X` kernel + one line in
  `torch.py` — provided the forward Jacobian stays correct.
- **`m` is a global scalar `Constant`** (`glide/grid.py:90`), broadcast into the per-stencil
  `s.m`. Settable today via `mg.sliding.m.set(value)` (no edit needed for a *global* exponent),
  but it has no gradient and cannot vary in space.
- **A forward-mode dual-number framework exists** (`DualFloat`, `apply_jvp`, `get_*_dual` in
  `stress.cu`) — useful for deriving new JVP terms without hand algebra.
- **`GlideStep` now returns 4 values** — `u, v, H, mask` (`torch.py:32`); `backward` takes
  `(gu, gv, gH, gM)` (line 35) and returns grads for `H_prev, bed, beta, smb`. The mask is
  non-differentiable. (Downstream code should unpack `u, v, *_`.)
- **Finite-difference gradient-test harness exists**: `tests/adjoint_test.py`,
  `tests/grad_test.py`, `tests/jvp_test.py`. Every new derivative must ship a check here.

## What's missing

1. `d_m` — no exponent gradient (Jacobian term + contraction kernel + autograd exposure).
2. Law-form flexibility — the law is inlined as one Weertman form; no way to express
   Budd / regularized-Coulomb, nor to invert for *their* parameter fields.

## Two flavors of "general", and their plumbing

1. **Kernel-resident parametric families** (Weertman / Budd / regularized-Coulomb): the NN
   predicts *static* spatial parameter fields, the kernel owns `|u|`. Native, Picard-free,
   exact adjoint. **This is the flagship.**
2. **A fully arbitrary NN drag** `C(|u|, features)`: the NN can't run inside the CUDA kernel,
   so this still needs the Picard/UDE loop — but the kernel can give it a **clean external
   drag-coefficient hook** (an explicit per-cell `C` input with the correct `d_u` for the
   Newton coupling) instead of the phase-3 `m=1` β-slot *hack*.

## Proposed design — a sliding-law interface

Refactor the inlined law into a functor with a fixed contract, dispatched on the
`sliding_law` selector (compile-time template or runtime enum; the Weertman branch is the
current code verbatim):

```cpp
struct SlidingLaw {
    __device__ float value (float unorm_sq, Params p) const;  // basal stress
    __device__ float d_u   (float unorm_sq, Params p) const;  // d(tau_b)/d(u)  (have, Weertman)
    __device__ ParamsGrad d_par(float unorm_sq, Params p) const; // d(tau_b)/d(each param)
};
```

Concrete laws (~10–20 lines each): `Weertman{beta, m}` (default; `d_par = {d_beta, d_m}`),
`Budd{beta, m, N}`, `RegularizedCoulomb{tau_max, u_c}` (Schoof; monotone stress by
construction; the prototype's law, made native). Prefer to obtain `d_u`/`d_par` via the
existing dual-number machinery rather than hand-deriving each entry.

**Stability constraint (physical, not numerical):** a rate-weakening law
(`d|tau|/d|u| < 0`) makes the Newton Jacobian indefinite — the ice-stream multiplicity
problem — so it destabilizes the *native* solve too, not just Picard. Default to
monotone-stress forms (Weertman `m>0`, regularized-Coulomb) and document the requirement.
This is exactly what the phase-3 study found the data wants anyway.

## Tiered plan

| tier | deliverable | files | risk |
|------|-------------|-------|------|
| **1** | invertible global `m` | `stress.cu` (`d_m` term), `operators.py` (`compute_gradient_m`), `grid.py`/`multigrid.py` (`m` carries a grad), `torch.py` (expose `m.grad`), `tests/grad_test.py` (FD check) | low — forward unchanged |
| **2** | `SlidingLaw` interface + Budd + regularized-Coulomb with parameter *fields* + per-param gradient kernels + `sliding_law` selector | `stress.cu`, `grid.py`, `multigrid.py`, `operators.py`, `torch.py`, all three test files | moderate — touches core residual (default branch preserved) |
| **3** | spatially-varying `m(x,y)` (`Constant` → `Field`) | + all kernel signatures carrying `m` | higher |

### Tier-1 `d_m` derivation (for reference)

For the sliding stress `tau_b^slide = -beta_eff · (|u|^2 + u_reg)^{(m-1)/2} · u`
(the `beta`-dependent part of `jac.res`; the `water_drag` term has no `m`-dependence):

```
d(tau_b^slide)/dm = tau_b^slide · (1/2) · ln(|u|^2 + u_reg)
                  = -beta_eff · unorm_sq_pow · u · 0.5 · logf(unorm_sq + u_reg)
```

`compute_gradient_m` contracts this with `lambda_u`/`lambda_v` and reduces to the single
scalar `m.grad` (sum over grounded cells). Because `m` is a scalar `Constant`, the reduction
is over the whole domain.

## Test plan

Extend the FD harness so every new derivative is checked against a central difference on a
small grid: `d_m` (Tier 1); `d_tau_max`, `d_u_c`, `d_N`, and each law's `d_u` (Tier 2). Add a
coupled check that a native velocity-dependent inversion recovers a known synthetic parameter
field (port the prototype's `phase3/synthetic_test.py` to the native path). Gate the PR on these
passing, and confirm the `sliding_law="weertman"` default reproduces current results bit-for-bit.

## Rollout

Land Tier 1 first — small, self-contained, exercises the whole pattern (new gradient kernel
+ autograd exposure + FD test). Then Tier 2 (the regularized-Coulomb family) builds on the
proven path. Keep the phase-3 Picard prototype (`phase3/`) as the reference oracle the native
path is validated against. Push `sliding-laws` to `fork`; PR into `origin` (`glide-ism/glide`).
