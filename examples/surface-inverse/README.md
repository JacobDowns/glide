# Inverting for basal traction from *surface* velocity

GLIDE's inversions have always compared observed surface velocity against the model's
**depth-averaged** velocity. That is exact under SSA, where the two are the same field.
Under DIVA they are not: they differ by the vertical shear,

$$\frac{u_s}{|\bar{\mathbf u}|}
=\frac{u_b+\tau_b\mathcal I_1}{u_b+\tau_b\mathcal I_2},$$

which is 1 wherever sliding dominates and rises as deformation takes over. A uniform slab
with a frozen bed gives $\mathcal I_1/\mathcal I_2=(n+2)/(n+1)=1.25$ at $n=3$ — a useful
reference value, but **not an upper bound**: the ratio follows the actual viscosity
profile. Measured over observed ice on Greenland at full resolution, median **1.038**,
90th percentile 1.088, max **1.43**, with 35% of cells above 1.05.

Fitting $|\bar{\mathbf u}|$ to surface data therefore asks the inversion to absorb the
shear into $\beta$, biasing the recovered traction exactly where deformation matters.
`forward_greenland_diva/` shows this directly: running a β that was inverted under SSA
through the DIVA forward model over-predicts surface speed in a broad band around the
whole margin, because that β was fitted to make $|\bar{\mathbf u}|$ match observations
that the surface velocity exceeds.

These examples invert against the surface velocity instead.

| File | What it does |
|---|---|
| `twin_inverse.py` | Synthetic twin: recover a **known** β from DIVA surface observations, and run the same inversion against the depth average for comparison. Minutes. |
| `greenland_surface_inverse.py` | The Greenland case, structured exactly like `examples/greenland/greenland_inverse.py`, with DIVA and the surface objective. |
| `common.py` | The surface-velocity vector, and the elastic-net regularizer. |
| `plotting.py` | Figures: β, velocity fit, shear ratio, convergence. |

```bash
uv run python examples/surface-inverse/twin_inverse.py
uv run python examples/surface-inverse/greenland_surface_inverse.py --coarsest 5 --finest 2
```

## How the surface objective reaches β

The closure returns a surface *speed*; its direction is inherited from the depth-averaged
flow. The model counterpart of an observed velocity vector is therefore composed in torch,

```python
u_surf = u_s * (u_c, v_c) / |(u_c, v_c)|
```

so autograd carries the derivative through both factors. The gradient then has two pieces
that a depth-averaged objective does not:

$$\frac{dJ}{d\beta}=\underbrace{\frac{\partial J}{\partial\beta}\bigg|_{\rm explicit}}_{\rm new}
+\lambda^T\frac{\partial r}{\partial\beta},
\qquad
\left(\frac{\partial r}{\partial x}\right)^{T}\lambda=-\underbrace{\frac{\partial J}{\partial x}}_{\rm new\ scatter}$$

- $\partial J/\partial x$ is scattered from cells to velocity facets through the stored
  $\partial u_s/\partial\dot\varepsilon^2_{\rm mem}$ and $\partial u_s/\partial\bar U$;
- $\partial J/\partial\beta|_{\rm explicit}$ is new outright — $u_s$ depends on $\beta$
  *directly* through the column closure, which $\bar{\mathbf u}$ never did.

Both are handled inside `GlideStep` when `return_u_s=True`. The derivation is in
[`notes/diva_adjoint_map.md`](../../notes/diva_adjoint_map.md) §7 and the kernels are
verified by `tests/diva_surface_gradient_test.py`.

## Two things that will bite

**The adjoint tolerance must be reachable.** `GlideStep` zeros every gradient when the
adjoint solver reports non-convergence, and the solver reports non-convergence whenever it
spends its full V-cycle budget. The float32 residual floor is grid-dependent — about
`7e-7` relative at 64², `1.3e-6` at 128² — so a tolerance below it guarantees a zero
gradient from a solve that had in fact converged perfectly well after two V-cycles.

This bites harder than it sounds, and it bit this example first. Downstream it looks like a
stalled optimizer, and if the objective carries a regularization term — which contributes
its gradient through torch rather than through `GlideStep` — the *total* gradient stays
nonzero, so a guard testing `grad != 0` sees nothing wrong while the optimizer quietly
minimizes the regularizer alone. `J_reg` falling while `J_data` climbs is the signature.

`GlideStep` now emits a `RuntimeWarning` and sets `model.last_backward_converged` when this
happens; check **that flag**, not the gradient, as both scripts do.

**Thickness is now part of the differentiated state.** The closure is differentiated with
respect to $H$ as well as velocity, so objectives may involve thickness and the gradient
remains exact for the coupled $(u,v,H)$ block
([`notes/diva_adjoint_map.md`](../../notes/diva_adjoint_map.md) §12). Auxiliary fields --
notably the grounded fraction $\varphi(H)$ -- are still held fixed, so a configuration
where grounding-line migration matters is outside what these gradients cover.

A practical note if you build a thickness objective: its finite-difference floor is far
coarser than a velocity objective's (~1e-2 against ~3e-4), because $H$ responds to $\beta$
only through one step of flux divergence. Sweep the step size rather than trusting a single
one -- `tests/diva_surface_torch_test.py` records the measured bracket.
