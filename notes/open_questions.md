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

## Q4. `prolongate_hfacet` dispatches to the **v**facet kernels

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

Not fixed here: it is upstream code, unrelated to DIVA, and correcting it changes SSA
output so it belongs in its own change with its own before/after numbers (the ones above).

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
