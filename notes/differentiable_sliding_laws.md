# Design doc: extensible, fully-differentiable sliding laws

**Status:** proposal · **Scope:** `glide` core (CUDA + adjoint plumbing) · **Author:** (draft)

## Summary

GLIDE's basal sliding law is hard-coded to a single Weertman power form, and the
adjoint exposes a gradient only w.r.t. the coefficient field `beta`. This proposal
adds (1) a gradient w.r.t. the rate exponent `m`, and (2) a small *sliding-law
interface* so additional physically-parameterized laws (Budd, regularized-Coulomb
/ Schoof) become first-class, fully-differentiable citizens whose spatial
parameter fields can be inverted for — or predicted by a neural network — exactly
like `beta` is today.

The headline payoff: a velocity-dependent friction law where **the NN predicts
static spatial parameter fields and the kernel owns the `|u|` dependence**. This
eliminates the outer Picard/Anderson loop a velocity-dependent law otherwise needs,
and yields an *exact* adjoint instead of a frozen-coefficient approximation.

## Motivation (empirical)

A prototype velocity-dependent inversion was built *outside* the kernel (NN-predicted
drag `C(|u|, features)`, lagged through a Picard/Anderson outer loop, reusing the
existing `beta`-adjoint at the converged state). It works, but the leave-one-basin-out
generalization study surfaced two limitations that a kernel-resident law removes:

1. **The outer loop is the weak link at extrapolation.** Holding out a fast NE basin
   (Mouginot 223), the flexible regularized-Coulomb law's *forward solve failed to
   converge* — Anderson oscillated at 4–10 % residual indefinitely, and the reported
   held-out error depended on where iteration stopped (35–62 m/yr across runs).
   Replacing it with a **rigid power law solved natively** (`beta(features)` static,
   GLIDE's own nonlinear solve handling `|u|^(m-1)`) made the *same* basin converge
   cleanly to ~25–28 m/yr. The pathology was the outer iteration, not the physics.
2. **The exponent is not invertible.** A sweep over a *global* `m` (1.0 / 0.5 / 0.333)
   showed generalization depends strongly on it (mild `m≈0.5–1` generalizes best;
   canonical `m=1/3` over-amplifies in fast basins). Today `m` can only be swept by
   hand because the adjoint has no `d_m`. A gradient would let it be calibrated.

The native rigid law also *beat* the outer-loop flexible law on mean held-out error
at a fraction of the cost. So the lesson is not "velocity dependence doesn't help" —
it's "put the velocity dependence inside the solver." This doc is how.

## Current state of the code

- **The velocity Jacobian already exists and is already used.** `get_tau_bx_jac`
  (`glide/cuda/stress.cu:254`) computes `d_u`, `d_v_*` (the rate dependence) — these
  drive the Newton/Vanka smoother *and*, via the transpose, the adjoint solve. So a
  velocity-dependent law is **already differentiated exactly at the fixed point**;
  there is no need to expose `dJ/du` (a state, not a control).
- **The adjoint is cleanly factored.** `model.backward()` solves the adjoint state
  `lambda` (law-agnostic — it only needs `d_u`/`d_v`). Each invertible parameter then
  has a *dedicated contraction kernel*: `compute_gradient_beta`
  (`glide/operators.py:526`) contracts `lambda` with `d(res)/d(beta)`; siblings exist
  for `bed`, etc. `GlideStep.backward` (`glide/torch.py:54`) reads `sliding.beta.grad`
  and returns it. **Adding a new invertible parameter is therefore additive** — a new
  `compute_gradient_X` kernel + one line in `torch.py` — provided the forward Jacobian
  stays correct.
- **`m` is a global scalar `Constant`** (`glide/grid.py:77`), broadcast into the
  per-stencil `s.m`. Settable today via `mg.sliding.m.set(value)` (no edit needed for a
  global exponent), but it has no gradient and cannot vary in space.
- **A forward-mode dual-number framework exists** (`DualFloat`, `get_tau_bx_dual`,
  `apply_jvp` in `stress.cu`) — useful for deriving new JVP terms without hand algebra.
- **A finite-difference gradient-test harness exists**: `tests/adjoint_test.py`,
  `tests/jvp_test.py`, `tests/grad_test.py`. Every new derivative must ship a check here.

## What's missing

1. `d_m` — no exponent gradient (Jacobian + contraction kernel + autograd exposure).
2. Law-form flexibility — the law is inlined as one Weertman form; no way to express
   Budd / regularized-Coulomb, nor to invert for *their* parameter fields.

## Proposed design

### A sliding-law interface in CUDA

Refactor the inlined law in `get_tau_bx_jac`/`get_tau_by_jac` into a functor with a
fixed contract, templated into the residual/Jacobian kernels (compile-time dispatch,
zero overhead; a runtime enum is an acceptable alternative):

```cpp
struct SlidingLaw {
    // basal stress magnitude factor and its derivatives, given speed^2 and params
    __device__ float       value (float unorm_sq, Params p) const;   // C such that tau_b = -(C + water_drag) u
    __device__ float       d_u   (float unorm_sq, Params p) const;   // d(tau_b)/d(u)   (already have for Weertman)
    __device__ ParamsGrad  d_par (float unorm_sq, Params p) const;   // d(tau_b)/d(each invertible param)
};
```

Concrete laws (each ~10–20 lines):
- `Weertman{ beta, m }` — current behavior; `d_par = {d_beta, d_m}`.
- `Budd{ beta, m, N }` — effective-pressure weighting.
- `RegularizedCoulomb{ tau_max, u_c }` (Schoof) — `|tau| = tau_max·|u|/(|u|+u_c)`,
  monotone stress by construction. This is the prototype's law, made native.

Prefer to obtain `d_u` and `d_par` through the **existing dual-number machinery**
(seed a dual per parameter, read the derivative) rather than hand-deriving each entry —
fewer ways to be wrong in a differentiable core.

### The NN/inversion contract

A law's parameters become `Field`s (like `beta`) in `grid.Sliding` / the multigrid
managers. The NN predicts them from features as *static* fields; the native solve
supplies the velocity dependence; each field gets a `compute_gradient_<param>` kernel
that contracts the already-solved `lambda`. No outer loop, exact gradient. This is the
phase-2 mechanism (`NN → beta`) generalized to richer physics (`NN → {tau_max, u_c}`).

### Stability constraint (physical, not numerical)

A rate-weakening law (`d|tau|/d|u| < 0`) makes the Newton Jacobian indefinite — the
ice-stream multiplicity problem — so it destabilizes the *native* solve too, not just
Picard. The interface should default to monotone-stress forms (Weertman `m>0`,
regularized-Coulomb), and the docs should state the requirement. This is exactly what
the prototype's generalization study found the data wants anyway.

## Tiered plan

| tier | deliverable | files touched | risk |
|------|-------------|---------------|------|
| **1** | invertible global `m` | `stress.cu` (`d_m` term), `operators.py` (`compute_gradient_m`), `grid.py`/`multigrid.py` (`m` carries a grad), `torch.py` (expose `m.grad`), `tests/grad_test.py` (FD check) | low — forward unchanged |
| **2** | `SlidingLaw` interface + Budd + regularized-Coulomb with parameter `Field`s and per-param gradient kernels | `stress.cu` (residual + Jacobian refactor), `grid.py`, `multigrid.py`, `operators.py`, `torch.py`, all three test files | moderate — touches core residual |
| **3** | spatially-varying `m(x,y)` (`Constant` → `Field`) | + all kernel signatures carrying `m` | higher |

### Tier-1 `d_m` derivation (for reference)

For `tau_b^slide = -beta · (|u|^2 + u_reg)^{(m-1)/2} · u`,

```
d(tau_b^slide)/dm = tau_b^slide · (1/2) · ln(|u|^2 + u_reg).
```

`compute_gradient_m` contracts this with `lambda_u`/`lambda_v` and reduces to the
single scalar `m.grad` (sum over grounded cells).

## Test plan

Extend the FD harness so every new derivative is checked against a central difference
on a small grid: `d_m` (Tier 1); `d_tau_max`, `d_u_c`, `d_N`, and each law's `d_u`
(Tier 2). Add a coupled check that a native velocity-dependent inversion recovers a
known synthetic parameter field (the prototype's synthetic-recovery test, ported to
the native path). Gate the PR on these passing.

## Rollout

Land Tier 1 first — small, self-contained, exercises the whole pattern (new gradient
kernel + autograd exposure + FD test) so Tier 2 builds on a proven path. Keep the
prototype (`phase3/invert_ude.py`, Picard) as the reference oracle the native path is
validated against.
