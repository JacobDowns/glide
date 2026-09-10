"""dJ/d(beta) through the torch wrapper for a DIVA SURFACE-velocity objective, on the unified branch.

Exercises the layer an inversion drives: glide_step(return_u_s=True), where the surface cotangent
travels torch autograd -> GlideStep.backward -> model.backward(dJdu_s=...) -> diva_surface_misfit_rhs
(cell cotangent -> facet RHS) -> FAS adjoint -> compute_gradient_beta -> diva_surface_param_gradient
(the explicit du_s/dbeta term).  Two things this catches that a kernel test cannot: the explicit term
dropped or double-counted, and the surface RHS overwriting the depth-averaged one instead of
accumulating.  The SSA control runs the same harness on a depth-averaged objective (surface path
inactive) and calibrates the float32 FD floor.  A third case checks a THICKNESS objective (the
closure's H path).  Adapted from the fork's diva_surface_torch_test.py to the unified 7-tuple
GlideStep (u_s is out[6], H is out[4]) and stress_scheme.

    python tests/diva_surface_test.py
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
EPS_SWEEP_H = (1e-2, 3e-3, 1e-3)
SEED = 12345


def build(scheme):
    model = IceDynamics(n_levels=n_levels, ny=ny, nx=nx, dx=dx, stress_scheme=scheme)
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
    if scheme == 'diva':
        mg.rheology.n_sigma.set(8.0)
        # leave eps_reg_shear at its 1e-12 default, matching the fork's diva_surface_torch_test.py
    for s in (model.forward_solver, model.adjoint_solver):
        s.vanka_options.omega.set(0.5)
        s.fas_options.set(coarsest_steps=200, pre_steps=10, post_steps=50,
                          finest_steps=150, maximum_vcycles=20,
                          relative_tolerance=1e-6, absolute_tolerance=1e-9,
                          report_norms=False)
    model.forward_solver.vanka_options.newton_options.relaxation.set(0.5)
    model.forward_solver.vanka_options.newton_options.steps.set(30)
    model.adjoint_solver.vanka_options.newton_options.momentum_damping.set(cp.float32(0.1))
    return model


def reference_beta():
    x = cp.arange(nx, dtype=cp.float32) * dx
    X, Y = cp.meshgrid(x, cp.arange(ny, dtype=cp.float32) * dx)
    L = 20000.0
    return (1000 * cp.sin(2 * cp.pi * X / L) * cp.sin(2 * cp.pi * Y / L) + 1000) / (RHO_I * GRAV)


def velocity(model, beta_t, surface, step=None):
    """Cell-centred model velocity: surface vector under DIVA, else depth average; 'H' -> thickness."""
    g = model.mg[0]
    H_prev = torch.tensor(g.state.H_prev.data)
    bed = torch.tensor(g.geometry.bed.data)
    smb = torch.tensor(g.forcing.smb.data)
    with contextlib.redirect_stdout(io.StringIO()):
        out = glide_step(cp.float32(0.0), step if step is not None else dt,
                         model, 0, H_prev, bed, beta_t, smb,
                         return_u_s=(surface is True))
    if surface == 'H':
        return out[4]                                   # diva-mi 7-tuple: H is out[4]
    u, v = out[0], out[1]
    u_c = 0.5 * (u[:, 1:] + u[:, :-1])
    v_c = 0.5 * (v[1:] + v[:-1])
    if not surface:
        return u_c, v_c
    spd = torch.sqrt(u_c ** 2 + v_c ** 2 + 1e-12)
    return out[6] * u_c / spd, out[6] * v_c / spd        # u_s is out[6]


def objective(model, beta_t, obs, w, surface, step=None):
    out = velocity(model, beta_t, surface, step)
    if surface == 'H':
        return 0.5 * (w * (out - obs[0]) ** 2).sum()
    ux, vy = out
    return 0.5 * (w * ((ux - obs[0]) ** 2 + (vy - obs[1]) ** 2)).sum()


def run(scheme, surface, tag, step=None):
    model = build(scheme)
    w = torch.zeros((ny, nx), device='cuda')
    w[6:-6, 6:-6] = 1.0
    bt = reference_beta()
    _o = velocity(model, torch.tensor(bt), surface, step)
    obs = [_o.detach()] if surface == 'H' else [o.detach() for o in _o]

    beta0 = torch.tensor(bt * 1.3, requires_grad=True)
    J = objective(model, beta0, obs, w, surface, step)
    J.backward()
    g = beta0.grad.detach().cpu().numpy()
    assert np.abs(g).max() > 0.0, "gradient identically zero -- adjoint likely reported non-convergence"

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
    rel_ssa = run('ssa', False, "SSA control (depth-averaged objective, surface path inactive)")
    assert rel_ssa < 1e-2, f"harness broken: SSA gradient vs FD is {rel_ssa:.3e}"
    rel_diva = run('diva', True, "DIVA (surface-velocity objective through GlideStep)")
    # The surface objective is nonlinear (u_s * unit(ubar)), so its FD sweep is noisy: the MIN over
    # eps is the adjoint accuracy (~3e-3 here), while larger eps carry truncation error.  Bound is set
    # to catch a gross wiring error -- a dropped/double-counted explicit du_s/dbeta term or a surface
    # RHS that overwrites rather than accumulates would push this to O(1e-1), far above 1.5e-2.
    assert rel_diva < 1.5e-2, (f"DIVA surface gradient is wrong: {rel_diva:.3e} (SSA control "
                               f"{rel_ssa:.3e}) -- suspect the explicit du_s/dbeta term or the "
                               f"surface RHS accumulation in diva_surface_misfit_rhs.")

    relH_ssa = run('ssa', 'H', "SSA control (thickness objective, no closure H path)", step=cp.float32(10.0))
    assert relH_ssa < 3e-2, f"harness broken: SSA thickness gradient vs FD is {relH_ssa:.3e}"
    relH_diva = run('diva', 'H', "DIVA (thickness objective, closure H path under test)", step=cp.float32(10.0))
    # H responds to beta through one step of flux divergence, so its FD signal is small relative to
    # solver noise -- a coarse floor (~1e-2).  Bound catches a dropped closure-H path (deta_dH/dbe_dH),
    # which would leave DIVA far above the SSA control.
    assert relH_diva < 4e-2, (f"DIVA thickness gradient is wrong: {relH_diva:.3e} (SSA control "
                              f"{relH_ssa:.3e}) -- suspect the closure's H path (deta_dH/dbe_dH).")

    print(f"summary: velocity  SSA {rel_ssa:.3e}   DIVA surface {rel_diva:.3e}")
    print(f"         thickness SSA {relH_ssa:.3e}   DIVA {relH_diva:.3e}")
    print("OK: the torch surface-velocity and thickness gradients agree with finite differences")


if __name__ == '__main__':
    main()
