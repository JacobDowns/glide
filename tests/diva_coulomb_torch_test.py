"""End-to-end torch autograd for a DIVA Coulomb (u_c) inversion.

The other gradient tests check the adjoint math directly (diva_gradient_test calls
compute_gradient_u_c; diva_derivs_test finite-differences the closure derivatives).  This one
checks the PLUMBING the user actually drives: u_c passed as a differentiable input to
glide_step, the loss backpropagated through torch, and the gradient returned in u_c.grad --
GlideStep.forward setting the model's u_c, model.backward(compute_uc_grad=True) reducing the
adjoint onto it, and GlideStep.backward routing it into the right autograd slot.

Setup is inversion-shaped: synthetic surface observations from a reference u_c, then the
gradient of the velocity misfit at a perturbed u_c, compared to a finite difference of the same
loss along a seeded direction.  DIVA + regularized Coulomb law (u_c is a DIVA-only field here).

    uv run python tests/diva_coulomb_torch_test.py
"""
import contextlib
import io

import cupy as cp
import numpy as np
import pyproj
import torch

from glide.model import IceDynamics
from glide.torch import glide_step

ny = nx = 64
dx = 2000.0
L = 20000.0
RHO_I, GRAV = 917.0, 9.81
dt = cp.float32(1.0)
t = cp.float32(0.0)
SEED = 7
U_C_TRUE, U_C_EVAL = 100.0, 130.0
TOL = 1e-2                       # FD-limited, like the other gradient checks


def build():
    model = IceDynamics(n_levels=3, ny=ny, nx=nx, dx=dx, x0=0.0, y0=0.0,
                        crs=pyproj.CRS("EPSG:3413"), stress_scheme='diva')
    mg = model.mg
    x = cp.arange(nx, dtype=cp.float32) * dx
    y = cp.arange(ny, dtype=cp.float32) * dx
    X, Y = cp.meshgrid(x, y)
    srf = 1000.0 - cp.tan(cp.deg2rad(0.1)) * X + 10000
    bed = srf - 1000.0
    mg.geometry.bed.set(bed); mg.geometry.depth.set(cp.maximum(-bed, 0))
    mg.state.H.set(srf - bed); mg.state.H_prev.set(srf - bed)
    mg.rheology.B.set(cp.full((ny, nx), (1e-16 ** -(1. / 3)) / (RHO_I * GRAV), cp.float32))
    mg.rheology.n.set(3.0)
    for lvl in mg.levels:
        lvl.rheology.n_sigma.set(cp.float32(8.0))
    mg.sliding.sliding_law.set(1.0)                                  # regularized Coulomb
    # In Coulomb mode beta is tau_max; scale it for a comparable drag at these speeds.
    tau_max = (1000 * cp.sin(2 * cp.pi * X / L) * cp.sin(2 * cp.pi * Y / L) + 1000) / (RHO_I * GRAV) * 120.0
    mg.sliding.beta.set(tau_max); mg.sliding.m.set(1.0); mg.sliding.u_reg.set(1.0)
    mg.sliding.u_c.set(cp.full((ny, nx), U_C_TRUE, dtype=cp.float32))
    for solver, md in ((model.forward_solver, 0.1), (model.adjoint_solver, 0.01)):
        solver.fas_options.set(coarsest_steps=200, pre_steps=10, post_steps=50, finest_steps=150,
                               relative_tolerance=1e-7, absolute_tolerance=1e-7,
                               maximum_vcycles=30, report_norms=False)
        solver.vanka_options.omega.set(cp.float32(0.5))
        solver.vanka_options.newton_options.momentum_damping.set(cp.float32(md))
    model.forward_solver.vanka_options.newton_options.relaxation.set(cp.float32(0.5))
    model.forward_solver.vanka_options.newton_options.steps.set(30)
    return model, mg


def main():
    model, mg = build()
    H_prev = torch.tensor(mg[0].state.H_prev.data, device='cuda')
    bed = torch.tensor(mg[0].geometry.bed.data, device='cuda')
    beta = torch.tensor(mg[0].sliding.beta.data, device='cuda')
    smb = torch.zeros((ny, nx), device='cuda')
    w = np.zeros((ny, nx), np.float32); w[6:-6, 6:-6] = 1.0
    wu = torch.tensor(cp.asarray(np.pad(w, ((0, 0), (0, 1)), mode='edge')), device='cuda')
    wv = torch.tensor(cp.asarray(np.pad(w, ((0, 1), (0, 0)), mode='edge')), device='cuda')

    def velocity(uc_np, want_grad=False):
        uc = torch.tensor(cp.asarray(uc_np), device='cuda', requires_grad=want_grad)
        with contextlib.redirect_stdout(io.StringIO()):
            u, v, ud, vd, H, mask = glide_step(t, dt, model, 0, H_prev, bed, beta, smb, u_c=uc)
        return uc, u, v

    # synthetic surface observations from the reference u_c
    _, u_obs, v_obs = velocity(np.full((ny, nx), U_C_TRUE, np.float32))
    u_obs, v_obs = u_obs.detach().clone(), v_obs.detach().clone()

    def loss(u, v):
        return 0.5 * (((u - u_obs) * wu) ** 2).sum() + 0.5 * (((v - v_obs) * wv) ** 2).sum()

    # autograd gradient of the misfit at a perturbed u_c
    uc_eval = np.full((ny, nx), U_C_EVAL, np.float32)
    uc, u, v = velocity(uc_eval, want_grad=True)
    loss(u, v).backward()
    g_adj = cp.asnumpy(cp.asarray(uc.grad.detach()))

    rng = np.random.RandomState(SEED)
    d = (rng.randn(ny, nx) * w).astype(np.float32)
    gvp_adj = float(np.sum(g_adj * d))

    def Jval(uc_np):
        _, u, v = velocity(uc_np)
        return float(loss(u.detach(), v.detach()))

    print(f"DIVA Coulomb u_c torch-autograd gradient vs finite differences:")
    best = np.inf
    for eps in (3.0, 1.0, 0.3):
        fd = (Jval(uc_eval + eps * d) - Jval(uc_eval - eps * d)) / (2 * eps)
        rel = abs(fd - gvp_adj) / max(abs(fd), 1e-30)
        print(f"  eps={eps:<5g}  FD = {fd:+.6e}  autograd = {gvp_adj:+.6e}  rel = {rel:.3e}")
        best = min(best, rel)
    assert np.any(g_adj != 0.0), "u_c.grad is identically zero -- the Coulomb gradient did not flow"
    assert best < TOL, f"torch autograd u_c gradient disagrees with FD ({best:.3e})"
    print(f"\nOK: the DIVA Coulomb u_c gradient flows through torch autograd (best rel {best:.2e})")


if __name__ == '__main__':
    main()
