# Open questions for GLIDE (found while implementing DIVA)

Things noticed on the `diva` branch that concern **existing, upstream** GLIDE code rather
than DIVA itself. Nothing here has been changed, because each would alter SSA behaviour
and belongs in its own discussion. Recorded so they are not lost.

---

## Q1. The adjoint V-cycle uses the FAS form for a linear system

**Where:** `FASAdjointSolver.vcycle`, `glide/multigrid.py`.

**What.** The adjoint system `Jᵀλ = f` is *linear* -- `J` depends only on the frozen
forward state, which is why `vanka_smooth_adjoint` has no Newton loop (contrast the
forward smoother's `while (k<newton_steps && rnorm>tol)`). But the V-cycle still uses the
Full Approximation Scheme: it restricts `λ` itself, saves it in `w_lambda_*`, applies the
coarse operator to the restricted `λ`, forms the tau-corrected right-hand side, and then
recovers the correction by subtracting the saved copy:

```python
next_level.f_u[:,:] = next_level.vjp_u[:,:] - next_level.r_u[:,:]      # A_c(I λ_h) − I r_h
...
next_level.scratch.z_lambda_u[:,:] = next_level.lambda_u.data[:,:] - next_level.w_lambda_u[:,:]
```

For a linear operator this is *algebraically equivalent* to the plain correction scheme.
Writing `v_c = I λ_h + e_c` and using linearity, the `A_c(I λ_h)` terms cancel and leave
`A_c e_c = −I r_h`, which is what a correction scheme solves directly (zero coarse initial
guess, `f_c = −I r_h`, prolong `λ_c` as the correction).

**Cost is not the issue.** The extra work is three restrictions of `λ`, three copies,
three subtractions, and one extra coarse `compute_vjp` per level. Summed over the
hierarchy the extra operator applications come to about `1/4 + 1/16 + 1/64 + ... ≈ 0.33`
finest-grid equivalents per V-cycle, against roughly 231 finest-equivalent smoothing
sweeps with the `tests/grad_test.py` settings -- about **0.1%**. Negligible.

**The concern is float32 cancellation.** The correction is recovered as a difference of
two nearly equal, large quantities (`λ_c` and `w_c`, both approximately `I λ_h`). Late in
convergence the coarse solve barely moves `λ_c`, so the difference loses relative
precision: in float32 (~7 digits) a correction of size `1e-4·|λ|` retains ~3 digits, and
`1e-6·|λ|` retains essentially none. A correction scheme has no such cancellation,
because `e_c` is built from zero and its accuracy is relative to its own magnitude.

Note the asymmetry with the forward solver: there the operator is genuinely nonlinear, so
FAS is required and the cancellation is a price that must be paid. In the adjoint it is
avoidable.

**Hypothesis, and evidence against it so far.** This *could* limit how far the adjoint
solve converges. However, on the `tests/diva_adjoint_test.py` configuration the SSA
adjoint reaches `9.5e-7` in 3 V-cycles and the DIVA adjoint `1.1e-6` in 20, i.e. both
attain the requested `1e-6`. So the cancellation is a theoretical concern that has **not
been demonstrated to bite**; the plateau originally observed in `tests/grad_test.py` is
more plausibly the under-converged *forward* solve there (see Q3). Recorded because the
argument is structural and the fix is cheap, not because a failure has been attributed
to it.

**Proposed test** (small, contained): switch the adjoint `vcycle` to the correction
scheme -- `f_c = −I r_h`, zero the coarse `λ`, prolong `λ_c` directly -- and see whether
the adjoint residual floor drops. If it does, it is an accuracy improvement for *every*
gradient GLIDE produces, well beyond DIVA. If it does not, the current form is vindicated
and the FAS shape is worth keeping for code symmetry with the forward solver.

---

## Q2. Guard bounds in the `compute_gradient_*` kernels look like they drop one cell

**Where:** `glide/cuda/grad.cu` -- `compute_gradient_beta` (lines ~64-65, ~89-90),
`compute_gradient_bed` (~328-329, ~346-347), and `compute_gradient_u_c` (~159-160,
~184-185, added on the `sliding-laws` branch, which mirrored the existing pattern).

**What.** Each kernel loops over momentum rows (facets) and scatters `λ_row × ∂r/∂param`
to the two adjacent cells:

```cuda
if (j>0     )  {atomicAdd(&grad_beta[i * nx + j - 1], lambda_u_l * j_tau_bx.d_beta_l);}
if (j<(nx-1))  {atomicAdd(&grad_beta[i * nx + j]    , lambda_u_l * j_tau_bx.d_beta_r);}
```

The left guard is clear: cell `j-1` exists only when `j > 0`.

The right guard is the question. For a u-facet `j`, the right-hand cell is `j`, which is a
valid cell index for `j <= nx-1`; the guard needed to stay in bounds is therefore
`j < nx` (facets run to `j = nx`, per `has_u`). The code uses `j < nx-1`, which also
excludes `j = nx-1` -- an interior facet with a real momentum row, since only facets `0`
and `nx` carry Dirichlet rows (`if (j == 0 || j == nx) ru_l = get_vfacet(u,...)`).

If that is unintended, then the **last column of `dJ/dβ` (and `dJ/d(bed)`, `dJ/du_c`) is
systematically missing one of its four contributions**, and likewise the last row via the
`i < ny-1` guard in the v-block.

**Why it might be deliberate.** The pattern is applied consistently across all three
kernels and both directions, which does not look accidental. Plausible intent: excluding
boundary-adjacent cells from inversion, or suppressing an edge artefact.

**Question for Doug:** is the `nx-1` / `ny-1` bound intentional, or should it be
`nx` / `ny`? Worth resolving before the DIVA parameter gradients are written, since they
will otherwise inherit whichever convention is chosen.

---

## Q3. `tests/grad_test.py` cannot detect regressions

**Where:** `glide/tests/grad_test.py:96`.

The perturbation direction is drawn unseeded:

```python
beta_pert = cp.random.randn(*grid.sliding.beta.data.shape, dtype=cp.float32)
```

so every run tests a different directional derivative and the reported relative error
varies widely -- observed 1.6% to 26% across runs on *unmodified upstream `main`*, with
the FD estimate itself changing sign (`+9515`, `−2562`, `+2937`). The forward solve in
that configuration also plateaus near `|r|/|r0| ≈ 3e-2` rather than reaching its `1e-3`
tolerance, so the finite-difference reference is noise-dominated.

This is not a defect in the adjoint -- it is a test that cannot serve as a regression
oracle. `tests/ssa_regression_test.py` was added on the `diva` branch for that purpose
(cold-started, converged, bit-reproducible). Left as-is deliberately; a seeded direction
plus a converged configuration would make `grad_test.py` a usable gradient check, but
that is a change to an upstream test.

------------------------------------------------------------------------

## Q4. `prolongate_hfacet` dispatched to the **v**facet kernels — FIXED on this branch

`Multigrid.prolongate_hfacet` (`glide/multigrid.py`) looks up
`prolongate_vfacet_injection` / `prolongate_vfacet_bilinear` rather than the `hfacet`
kernels of the same names. Those exist in `cuda/transfer.cu`, are correct, and are
**never called**.

This is not a naming quibble: the two kernels index different grids. `vfacet` treats the
array as `(ny, nx+1)` and reads the coarse field with row stride `nx_coarse+1`; `hfacet`
treats it as `(ny+1, nx)` with stride `nx_coarse`. The v-velocity correction is therefore
prolongated with the u-grid's strides, which shears it by one element per row. For a
square grid the *element count* happens to match (`ny*(nx+1) == (ny+1)*nx`), so nothing
goes out of bounds and nothing crashes -- which is presumably why it has survived.

Direct check, prolongating a random coarse v field both ways on a 9x8 -> 17x16 transfer:

    max |vfacet_kernel - hfacet_kernel| = 2.06   (field values are O(1))
    270 of 272 elements differ by more than 1e-6

Both call sites are live: the forward FAS V-cycle and the adjoint V-cycle both prolongate
the `z_v` correction this way.

**Impact.** Invisible in the configurations our tests use, because they run a strong fine
smoother (`finest_steps = 150`, `post_steps = 50`) that does nearly all the work -- fixing
it changes convergence there by under 5%, and the *converged answer* is unaffected either
way, since FAS coarse corrections only accelerate and the fine smoother still drives the
true residual down. It matters when multigrid is actually load-bearing. With the smoother
turned down (`pre/post/finest = 2`, `newton steps = 5`) on the same 128^2 problem:

    V-cycle 6    |r|/|r0|     |r_u|      |r_v|
    current       6.31e-4    8.02e-2    1.14e-1
    fixed         4.12e-4    6.85e-2    5.88e-2

so the v residual converges about **2x** faster once its correction is right. Note the
signature in the current version: `|r_v|` sits consistently *above* `|r_u|` and the gap
widens, whereas with the fix the two components contract at the same rate -- which is what
a symmetric problem should do.

**Fixed** on the `diva` branch. Before doing so it was checked three independent ways,
because "the original might have been correct in some matched way" is a real possibility
with staggered grids and the convergence evidence alone was not conclusive:

1. *Strides.* `prolongate_hfacet` receives a `(ny+1, nx)` field, whose coarse row stride is
   `nx_coarse`. The vfacet kernel reads with `nx_coarse+1`.
2. *Linear exactness, solver-independent.* Bilinear prolongation must reproduce a linear
   field exactly. On a linear ramp over the hfacet grid, interior error was
   **0.000000e+00** with the hfacet kernel and **exactly 1.0** with the vfacet kernel --
   i.e. a half-cell offset in y, which is precisely the staggering difference (an hfacet
   value sits at `(j, i+0.5)`, a vfacet value at `(j+0.5, i)`).
3. *Consistency with the restriction.* `restrict_hfacet` is exact for a linear field on the
   `(ny+1, nx)` layout (error 0.000000e+00), so it was always written for the correct
   staggering -- it was NOT matched to the buggy prolongation. Restriction and prolongation
   were an inconsistent pair; the round trip restrict -> prolongate is now exact.

The converged solution is unchanged, as expected for a coarse-grid correction: driven to 30
V-cycles, both versions reach the same true momentum residual (2.09e-3 vs 2.12e-3) and their
states agree to 2e-4 in v and 3e-5 in u.

`tests/ssa_reference.json` was regenerated. The fingerprint moved by 3.9% in v (and only
1e-6 in u, 1e-9 in H, which is the expected signature of a change to the v prolongation).
That is larger than it sounds: the regression configuration stops at 15 V-cycles and stalls
short of its tolerance (Q6), so the fingerprint records an ITERATE, not a solution, and a
different iteration path lands on a different iterate. The fingerprint remains a valid
change-detector, which is its job; it is not evidence about the answer.

------------------------------------------------------------------------

## Q5. The Vanka smoother's `relaxation` option is a no-op

`vanka_smooth_body` (`cuda/vanka.cu`) takes `float relaxation`, then unconditionally
overwrites it inside the Newton loop:

```cuda
    relaxation = 0.5f;
```

So `vanka_options.newton_options.relaxation` has no effect. Currently unobservable -- the
Python default is also `0.5` and every caller in `tests/` sets `0.5` -- but the knob is
dead, and anyone who reaches for it to stabilise a hard solve will see no change and
conclude that under-relaxation does not help.

Either the line is a debugging leftover, or it is deliberate and the parameter should be
removed from the signature and the config so it stops advertising a control that does not
exist. Left as-is because changing it alters SSA behaviour for any caller passing something
other than 0.5.


------------------------------------------------------------------------

## Q6. The reported residual norm mixes units, so `|r|/|r0|` understates convergence

**This entry replaces an earlier claim that the forward solve stalls an order of magnitude
short of its tolerance. That was wrong.** The solve converges to float32 round-off. What
plateaus is the reported *number*, for reasons that are entirely in the norm.

`FASCDSolver.solve` reports

```python
    |r| = sqrt(norm(r_u)**2 + norm(r_v)**2 + norm(r_H)**2)      # cp.linalg.norm, raw L2
```

Two problems, both raised by Jake before this was measured.

**1. The three components have different units.** `r_u`, `r_v` are momentum residuals; `r_H`
is mass (thickness per time). Adding their squares makes the total track whichever has the
larger raw magnitude, regardless of how converged either is. On the standard 128^2 slab, at
the point where the reported ratio plateaus:

    row         |r|        scale of that row      relative      float32 floor for the norm
    momentum   1.56e-3     2.23e+02  (|r_u| at t=0)   7.0e-6         2.7e-5
    mass       2.25e-2     1.28e+05  (|H/dt|)         1.8e-7         1.5e-2

The momentum rows are *below* a round-off estimate for their own norm. The mass row sits at
1.5x its floor. Both are converged as far as float32 permits. `|r_H|` is large in absolute
terms only because `H ~ 1000` and `dt = 1`, so one ulp per cell over 16384 cells already
gives ~1.5e-2.

**2. `|r0|` is pure x-momentum in any cold-started scenario.** These tests start from
`u = v = 0` with `H = H_prev`, so at t=0

    |r_u| = 2.23e+02        (the driving stress)
    |r_v| = 0.000e+00       (no slope in y, v = 0)
    |r_H| = 0.000e+00       (H = H_prev and div(uH) = 0)

`r_H` starts at *exactly zero* and can only grow once the velocities become nonzero. So
dividing the later mixed norm by this `|r0|` is measuring the mass row's round-off floor in
units of the momentum row's initial driving stress:

    1.5e-2 / 2.23e+02  =  6.7e-5     vs the observed plateau of ~1.0e-4.

That is the whole "stall". There is nothing to diagnose in the solver.

**3. Not normalised by grid size.** `cp.linalg.norm` over an (ny, nx) array grows like
`sqrt(N)` for a fixed pointwise residual. `|r|/|r0|` is unaffected (both scale together), but
`absolute_tolerance` means something different at every resolution, and absolute residuals are
not comparable across the hierarchy.

**Suggested fix**, if this is worth changing upstream: report each row separately against a
scale of its own -- momentum against `|r_u(t=0)|` or the driving-stress norm, mass against
`|H/dt|` -- and use an RMS rather than a raw L2 so the numbers are resolution-independent.
As it stands, a converged solve reports `1e-4` and looks like a failure, which is how this
was misread here in the first place.

------------------------------------------------------------------------

## Q7. What actually limits the finite-difference gradient checks

Worth writing down because it was mis-attributed twice: first to the adjoint, then to a
non-existent solver stall.

The floor is float32 error in the *state*, and it is measurable. Solving the same problem four
ways -- all converged to round-off, varying multigrid depth (5 and 3 levels), V-cycle count
(15 and 40) and smoother strength -- gives

    J = 8.89718281e+04, 8.89718359e+04, 8.89715703e+04, 8.89717188e+04

a spread of **2.66e-1**, i.e. 3.0e-6 of J. The FD numerator at the step the test uses is
`2*eps*dJ/dbeta.d ~ 79.3`, so the implied floor on relative FD agreement is

    2.66e-1 / 79.3  ~  3.3e-3

Measured agreements sit at or just under that (SSA `dJ/d(beta)` 1.4e-3), which is expected:
the `+eps` and `-eps` solves use identical settings, so their state errors partially correlate
and cancel better than the worst-case spread implies.

Two things this rules out. It is **not** the objective's summation: accumulating J in float64
changes the agreement from 1.365e-3 to 1.330e-3, i.e. not at all. And it is **not** solver
convergence, per Q6.

Consequence: an SSA `dJ/d(beta)` reading of 1.4e-4, as seen before the Q4 fix, was *better
than the floor* -- a fortunate cancellation, not a reproducible baseline. That is why it moved
to 1.4e-3 when the iteration path changed, with no adjoint code touched. Absolute FD bounds on
this problem should therefore be set near 3e-3, or the comparison should be made against the
exact-adjoint SSA control, which measures the shared floor directly.
`tests/diva_gradient_test.py` does the latter.
