"""dJ/d(beta) through the TORCH wrapper, for an objective on surface velocity.

tests/diva_surface_gradient_test.py checks the kernels and the raw multigrid API.  This
one checks the layer an inversion actually drives: GlideStep with return_u_s=True, where
the surface cotangent has to travel

    torch autograd -> GlideStep.backward -> model.backward(dJdu_s=...)
        -> diva_surface_misfit_rhs        (cell cotangent -> facet right-hand side)
        -> FAS adjoint solve
        -> compute_gradient_beta          (lambda^T dr/dbeta)
        -> diva_surface_param_gradient    (the EXPLICIT du_s/dbeta term)

Two things can go wrong here that the kernel-level test cannot see: the explicit term
could be dropped or double-counted as it is added to the reduced gradient, and the
surface right-hand side could overwrite the depth-averaged one instead of accumulating
with it (diva_surface_misfit_rhs FILLS f_u/f_v).  Both show up as a finite-difference
gap in the objective the inversion is actually minimizing.

THE SSA CONTROL IS THE CALIBRATION.  It runs the same harness on a depth-averaged
objective, where the whole surface path is inactive, and it is subject to the same
float32 finite-difference floor.  Judging DIVA against an absolute bound instead would
be judging the FD noise (see notes/open_questions.md Q7).

The objective is the surface velocity VECTOR, u_s * unit(ubar) -- the composition an
inversion uses, so the direction factor is differentiated too.

A third case checks an objective on THICKNESS, J = 1/2 sum w*(H - H_obs)^2.  That one
needs no new plumbing on the torch side -- H has always been a returned state variable --
but under DIVA it is only correct once the closure is differentiated with respect to H,
because beta moves H through

    beta -> u,v -> flux divergence -> H

while H moves the residual back through the closure

    H -> I1, I2 -> U_b, tau_b -> eta_bar, beta_eff -> r ,

and the adjoint has to carry both.  It is run at a longer dt so the thickness has time to
respond to the velocity at all.

    uv run python tests/diva_surface_torch_test.py
"""
import contextlib
import io

import cupy as cp
import numpy as np
import torch

from glide.model import IceDynamics
from glide.torch import glide_step

ny = nx = 64
dx = 2000.0
dt = cp.float32(1.0)
n_levels = 4
RHO_I, GRAV = 917.0, 9.81
EPS_SWEEP = (1e-3, 3e-4, 1e-4)
# The thickness objective needs a different bracket entirely.  H responds to beta only
# through one step of flux divergence, so the signal is far smaller relative to the
# solver's own noise: sweeping down to 1e-4 lands well past the round-off wall (rel 4.4e-1
# at 3e-4), while 3e-2 is already into truncation error (1.0e-1).  Measured minimum is at
# 3e-3, where the SSA control reaches 7.5e-3 -- an order of magnitude coarser than the
# velocity objective's 3e-4, and that is the floor this case is judged against.
EPS_SWEEP_H = (1e-2, 3e-3, 1e-3)
SEED = 12345


def build(stress_balance):
    model = IceDynamics(n_levels=n_levels, ny=ny, nx=nx, dx=dx)
    mg = model.mg
    x = cp.arange(nx, dtype=cp.float32) * dx
    X, _ = cp.meshgrid(x, cp.arange(ny, dtype=cp.float32) * dx)
    srf = 1000.0 * cp.ones((ny, nx), cp.float32) - cp.tan(cp.deg2rad(0.1)) * X + 10000
    bed = srf - 1000.0

    mg.geometry.bed.set(bed)
    mg.rheology.B.set(cp.full((ny, nx), (1e-16 ** -(1. / 3)) / (RHO_I * GRAV), cp.float32))
    mg.state.H.set(srf - bed)
    mg.state.H_prev.set(srf - bed)
    mg.sliding.m.set(1.0)
    mg.sliding.u_reg.set(1.0)
    mg.rheology.stress_balance.set(stress_balance)
    mg.rheology.n_sigma.set(8.0)

    # 1e-6 relative is reachable in ~2 V-cycles; the float32 floor is ~7e-7.  A tolerance
    # below the floor makes the solver spend every cycle and report NOT converged, on
    # which GlideStep zeros the gradient -- which would make this test pass a zero.
    for s in (model.forward_solver, model.adjoint_solver):
        s.vanka_options.omega.set(0.5)
        s.fas_options.set(coarsest_steps=200, pre_steps=10, post_steps=50,
                          finest_steps=150, maximum_vcycles=20,
                          relative_tolerance=1e-6, absolute_tolerance=1e-9,
                          report_norms=False)
    model.forward_solver.vanka_options.newton_options.relaxation.set(0.5)
    model.forward_solver.vanka_options.newton_options.steps.set(30)
    model.adjoint_solver.vanka_options.newton_options.ssa_damping.set(cp.float32(0.1))
    return model


def reference_beta():
    x = cp.arange(nx, dtype=cp.float32) * dx
    X, Y = cp.meshgrid(x, cp.arange(ny, dtype=cp.float32) * dx)
    L = 20000.0
    return (1000 * cp.sin(2 * cp.pi * X / L) * cp.sin(2 * cp.pi * Y / L) + 1000) / (RHO_I * GRAV)


def velocity(model, beta_t, surface, step=None):
    """Cell-centred model velocity: the surface vector under DIVA, else the depth average."""
    g = model.mg[0]
    H_prev = torch.tensor(g.state.H_prev.data)
    bed = torch.tensor(g.geometry.bed.data)
    smb = torch.tensor(g.forcing.smb.data)
    with contextlib.redirect_stdout(io.StringIO()):
        # `surface` is True (surface vector), False (depth average) or 'H' (thickness);
        # only the first needs u_s returned.
        out = glide_step(cp.float32(0.0), step if step is not None else dt,
                         model, 0, H_prev, bed, beta_t, smb,
                         return_u_s=(surface is True))
    if surface == 'H':
        return out[2]
    u, v = out[0], out[1]
    u_c = 0.5 * (u[:, 1:] + u[:, :-1])
    v_c = 0.5 * (v[1:] + v[:-1])
    if not surface:
        return u_c, v_c
    spd = torch.sqrt(u_c ** 2 + v_c ** 2 + 1e-12)
    return out[4] * u_c / spd, out[4] * v_c / spd


def objective(model, beta_t, obs, w, surface, step=None):
    out = velocity(model, beta_t, surface, step)
    if surface == 'H':
        return 0.5 * (w * (out - obs[0]) ** 2).sum()
    ux, vy = out
    return 0.5 * (w * ((ux - obs[0]) ** 2 + (vy - obs[1]) ** 2)).sum()


def run(stress_balance, surface, tag, step=None):
    model = build(stress_balance)
    w = torch.zeros((ny, nx), device='cuda')
    w[6:-6, 6:-6] = 1.0

    bt = reference_beta()
    _o = velocity(model, torch.tensor(bt), surface, step)
    obs = [_o.detach()] if surface == 'H' else [o.detach() for o in _o]

    beta0 = torch.tensor(bt * 1.3, requires_grad=True)
    J = objective(model, beta0, obs, w, surface, step)
    J.backward()
    g = beta0.grad.detach().cpu().numpy()

    assert np.abs(g).max() > 0.0, (
        "the gradient is identically zero -- GlideStep zeroes it when the adjoint "
        "reports non-convergence, so check that the FAS tolerance is above the float32 "
        "residual floor rather than looking for a bug in the surface path")

    rng = np.random.RandomState(SEED)
    d = np.zeros((ny, nx), np.float32)
    d[6:-6, 6:-6] = rng.randn(ny - 12, nx - 12)
    gvp = float((g * d).sum())

    print(f"{tag}\n  J0 = {float(J.detach()):.6e}   adjoint g.d = {gvp:+.6e}")
    best = np.inf
    for eps in (EPS_SWEEP_H if surface == 'H' else EPS_SWEEP):
        pert = torch.tensor(cp.asarray(d) * eps)
        b = torch.tensor(bt * 1.3)
        Jp = float(objective(model, b + pert, obs, w, surface, step).detach())
        Jm = float(objective(model, b - pert, obs, w, surface, step).detach())
        fd = (Jp - Jm) / (2 * eps)
        rel = abs(fd - gvp) / max(abs(fd), 1e-30)
        print(f"  eps={eps:<8g} FD = {fd:+.6e}  rel = {rel:.3e}")
        best = min(best, rel)
    print(f"  best rel = {best:.3e}\n")
    return best


def main():
    # SSA control: the surface path is inactive, so this measures the harness and the
    # float32 FD floor, not the code under test.
    rel_ssa = run(0.0, False, "SSA control (depth-averaged objective, surface path inactive)")
    assert rel_ssa < 1e-2, f"harness broken: SSA gradient vs FD is {rel_ssa:.3e}"

    rel_diva = run(1.0, True, "DIVA (surface-velocity objective through GlideStep)")
    assert rel_diva < 1e-2, f"DIVA surface gradient is grossly wrong: {rel_diva:.3e}"

    bound = max(5.0 * rel_ssa, 3e-3)
    assert rel_diva < bound, (
        f"DIVA surface gradient is materially worse than the SSA control: "
        f"{rel_diva:.3e} vs {rel_ssa:.3e} (bound {bound:.3e}). Suspect the explicit "
        f"du_s/dbeta term or the accumulation of the surface right-hand side.")

    # ---- objectives on THICKNESS ----
    # dt = 10 so the thickness actually moves; at dt = 1 the response to beta is small
    # enough that the finite difference is mostly round-off.  dt = 50 is worse, not
    # better -- the adjoint stops converging there and GlideStep zeroes the gradient.
    relH_ssa = run(0.0, 'H', "SSA control (thickness objective, no closure H path)", step=cp.float32(10.0))
    assert relH_ssa < 3e-2, f"harness broken: SSA thickness gradient vs FD is {relH_ssa:.3e}"

    relH_diva = run(1.0, 'H', "DIVA (thickness objective, closure H path under test)",
                    step=cp.float32(10.0))
    assert relH_diva < 3e-2, f"DIVA thickness gradient is grossly wrong: {relH_diva:.3e}"
    boundH = max(3.0 * relH_ssa, 2e-2)
    assert relH_diva < boundH, (
        f"DIVA thickness gradient is materially worse than the SSA control: "
        f"{relH_diva:.3e} vs {relH_ssa:.3e} (bound {boundH:.3e}). Suspect the closure's "
        f"H path -- deta_dH/dbe_dH into the thickness cotangent.")

    print(f"summary: velocity  SSA {rel_ssa:.3e}   DIVA surface {rel_diva:.3e}")
    print(f"         thickness SSA {relH_ssa:.3e}   DIVA {relH_diva:.3e}")
    print("OK: the torch surface-velocity and thickness gradients agree with finite differences")


if __name__ == '__main__':
    main()
