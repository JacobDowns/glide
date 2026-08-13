"""Twin experiment: recover a known beta from DIVA SURFACE velocity observations.

Structured exactly like examples/greenland/greenland_inverse.py -- a torch optimization
loop around GlideStep, run coarse-to-fine over the multigrid hierarchy with elastic-net
regularization on log(beta) -- but with two differences that are the point of this
example:

  1. DIVA is on, so the surface velocity and the depth average are different fields.
  2. The objective is built on the SURFACE velocity, u_s * unit(ubar), which is what
     satellite observations actually measure.  Gradients reach beta through the
     closure's surface transpose AND through its explicit parameter dependence -- see
     notes/diva_adjoint_map.md 7.3.

Because the truth is known here, the run answers a question the Greenland case cannot:
does the recovered beta match the beta that generated the data?  It also runs the same
inversion against the DEPTH-AVERAGED velocity (what GLIDE inversions did before the
surface path existed) so the bias from fitting the wrong quantity is measurable rather
than asserted.

    uv run python examples/surface-inverse/twin_inverse.py
    uv run python examples/surface-inverse/twin_inverse.py --epochs 40 --no-compare
"""
import argparse
import contextlib
import io
import os
import time

import cupy as cp
import numpy as np
import torch

from glide.model import IceDynamics
from glide.torch import glide_step

from common import surface_velocity, depth_averaged_velocity, elastic_net
import plotting

RHO_I, GRAV = 917.0, 9.81


# ---------------------------------------------------------------- configuration ----
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n', type=int, default=128, help='grid size (square)')
    p.add_argument('--dx', type=float, default=2000.0)
    p.add_argument('--levels', type=int, default=4, help='multigrid levels')
    p.add_argument('--coarsest', type=int, default=2,
                   help='coarsest level to start the inversion on')
    p.add_argument('--epochs', type=int, default=60, help='iterations per level')
    p.add_argument('--lr', type=float, default=2e-2)
    p.add_argument('--noise', type=float, default=0.0,
                   help='relative Gaussian noise added to the observations')
    p.add_argument('--outdir', default='inverse_twin')
    p.add_argument('--no-compare', action='store_true',
                   help='skip the depth-averaged-objective comparison run')
    return p.parse_args()


# --------------------------------------------------------------------- the model ----
def build_model(args, stress_balance=1.0):
    """A slab on a slope with a bumpy bed -- enough shear for DIVA to matter."""
    ny = nx = args.n
    dx = args.dx
    model = IceDynamics(n_levels=args.levels, ny=ny, nx=nx, dx=dx)
    mg = model.mg

    x = cp.arange(nx, dtype=cp.float32) * dx
    y = cp.arange(ny, dtype=cp.float32) * dx
    X, Y = cp.meshgrid(x, y)
    L = nx * dx

    # Gentle surface slope, with a broad bedrock swell so the thickness varies.
    #
    # The ice is deliberately THICK.  Deformation grows far faster with thickness than
    # sliding does, and the surface-to-depth-averaged ratio approaches the uniform-slab
    # value I1/I2 = (n+2)/(n+1) = 1.25 at n = 3 as the bed freezes (not a hard bound --
    # real profiles reach 1.43 on Greenland).  Thin
    # ice puts that ratio near 1, where DIVA and SSA agree and this example would have
    # nothing to show.  At ~1800 m the slow interior sits near the shear-dominated end
    # while the streak still slides, so both regimes appear in one domain.
    srf = 2600.0 - cp.tan(cp.deg2rad(0.25)) * X
    bed = srf - (1800.0 + 400.0 * cp.sin(2 * cp.pi * X / L) * cp.sin(2 * cp.pi * Y / L))

    B = cp.full((ny, nx), (1e-16 ** -(1. / 3)) / (RHO_I * GRAV), cp.float32)

    mg.geometry.bed.set(bed)
    mg.geometry.depth.set(cp.maximum(-bed, 0))
    mg.rheology.B.set(B)
    mg.rheology.n.set(3.0)
    mg.rheology.eps_reg.set(1e-6)
    mg.state.H.set(srf - bed)
    mg.state.H_prev.set(srf - bed)

    mg.sliding.m.set(1.0)
    mg.sliding.u_reg.set(1.0)
    mg.sliding.water_drag.set(1e-4)

    # DIVA
    mg.rheology.stress_balance.set(stress_balance)
    mg.rheology.n_sigma.set(8.0)
    mg.rheology.eps_reg_shear.set(1e-12)

    # The float32 residual floor is grid dependent -- ~7e-7 relative at 64^2, ~1.3e-6 at
    # 128^2 -- and a tolerance below it makes the solver spend every V-cycle and report NOT
    # converged, on which GlideStep zeros the gradient.  1e-5 clears the floor at every
    # size used here and is still reached in about two cycles.
    for s in (model.forward_solver, model.adjoint_solver):
        s.vanka_options.omega.set(0.5)
        s.fas_options.set(coarsest_steps=200, pre_steps=10, post_steps=50,
                          finest_steps=100, maximum_vcycles=20,
                          relative_tolerance=1e-5, absolute_tolerance=1e-7,
                          report_norms=False)
    model.forward_solver.vanka_options.newton_options.relaxation.set(0.5)
    model.forward_solver.vanka_options.newton_options.steps.set(30)
    model.adjoint_solver.vanka_options.newton_options.ssa_damping.set(cp.float32(0.1))
    return model


def beta_true(args):
    """Smooth background traction with a slippery streak through the middle."""
    ny = nx = args.n
    dx = args.dx
    x = cp.arange(nx, dtype=cp.float32) * dx
    y = cp.arange(ny, dtype=cp.float32) * dx
    X, Y = cp.meshgrid(x, y)
    L = nx * dx

    background = 1400.0 + 900.0 * cp.sin(2 * cp.pi * X / L) * cp.sin(2 * cp.pi * Y / L)
    streak = 1.0 - 0.92 * cp.exp(-((Y - 0.5 * L) ** 2) / (2 * (0.055 * L) ** 2))
    return (background * streak / (RHO_I * GRAV)).astype(cp.float32)


def run_forward(model, level, beta_t, want_surface):
    """One GlideStep, returning the cell-centred velocity the objective compares."""
    g = model.mg[level]
    H_prev = torch.tensor(g.state.H_prev.data)
    bed = torch.tensor(g.geometry.bed.data)
    smb = torch.tensor(g.forcing.smb.data)
    with contextlib.redirect_stdout(io.StringIO()):
        out = glide_step(cp.float32(0.0), cp.float32(1.0), model, level,
                         H_prev, bed, beta_t, smb, return_u_s=want_surface)
    u, v, H, mask = out[:4]
    if want_surface:
        return surface_velocity(u, v, out[4])
    return depth_averaged_velocity(u, v)


# ---------------------------------------------------------------- the inversion ----
def invert(model, args, obs_levels, use_surface, tag):
    """Coarse-to-fine RMSprop on log(beta), exactly the Greenland example's schedule."""
    history = {}
    log_beta = torch.log(torch.tensor(
        model.mg[args.coarsest].sliding.beta.data, device='cuda'))

    for level in range(args.coarsest, -1, -1):
        log_beta.requires_grad_()
        optimizer = torch.optim.RMSprop([log_beta], lr=args.lr)
        dx = model.mg[level].dx
        obs_x, obs_y = obs_levels[level]
        hist = {'J': [], 'J_data': [], 'J_reg': []}

        for j in range(args.epochs):
            optimizer.zero_grad()
            beta = torch.exp(log_beta)

            mx, my = run_forward(model, level, beta, use_surface)

            # Masked L1, as in the Greenland example: robust to the outliers that
            # real velocity mosaics carry.
            J_data = (abs(mx - obs_x)).mean() + (abs(my - obs_y)).mean()
            J_l2, J_tv = elastic_net(log_beta, dx)
            J = J_data + J_l2 + J_tv

            J.backward()
            if not torch.isfinite(log_beta.grad).all():
                raise RuntimeError(f'non-finite gradient at level {level}, iter {j}')
            # Test the MODEL's convergence flag, not whether the total gradient is zero:
            # the regularizer contributes its gradient through torch, so a zeroed model
            # gradient still leaves a nonzero total and the optimizer would quietly
            # minimize the regularizer alone (which is exactly what happened once).
            if not getattr(model, 'last_backward_converged', True):
                raise RuntimeError(
                    f'[{tag}] adjoint did not converge at level {level}, iter {j}; the '
                    f'gradient was zeroed. Raise fas_options.relative_tolerance above the '
                    f'float32 residual floor for this grid size.')
            optimizer.step()

            hist['J'].append(float(J.detach()))
            hist['J_data'].append(float(J_data.detach()))
            hist['J_reg'].append(float((J_l2 + J_tv).detach()))
            if j % 10 == 0 or j == args.epochs - 1:
                print(f'  [{tag}] level {level} iter {j:3d}/{args.epochs} | '
                      f'J {float(J.detach()):9.3f}  J_data {float(J_data.detach()):9.3f}  '
                      f'J_reg {float((J_l2 + J_tv).detach()):8.4f}')

        history[level] = hist
        if level > 0:
            log_beta = torch.tensor(model.mg.prolongate_cell(
                cp.asarray(log_beta.detach()), method='bilinear'))

    return torch.exp(log_beta.detach()), history


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(0)

    print('building the twin model (DIVA)...')
    model = build_model(args, stress_balance=1.0)
    b_true = beta_true(args)

    # ---- observations: a DIVA forward run at the true beta ----
    print('generating synthetic surface-velocity observations...')
    obs_x, obs_y = (o.detach() for o in
                    run_forward(model, 0, torch.tensor(b_true), True))
    if args.noise > 0:
        g = torch.Generator(device='cuda').manual_seed(1)
        scale = args.noise * torch.sqrt(obs_x ** 2 + obs_y ** 2)
        obs_x = obs_x + scale * torch.randn(obs_x.shape, generator=g, device='cuda')
        obs_y = obs_y + scale * torch.randn(obs_y.shape, generator=g, device='cuda')

    # The shear ratio at the truth: how much surface and depth-averaged flow differ.
    u_s_true = model.mg[0].state.u_s.data.copy()
    u, v = model.mg[0].state.u.data, model.mg[0].state.v.data
    ubar_true = cp.hypot(0.5 * (u[:, :-1] + u[:, 1:]), 0.5 * (v[:-1, :] + v[1:, :]))
    print(f'  surface/depth-averaged speed ratio: '
          f'median {float(cp.median(u_s_true / cp.maximum(ubar_true, 1e-6))):.3f}, '
          f'max {float((u_s_true / cp.maximum(ubar_true, 1e-6)).max()):.3f}')

    # Restrict the observations onto every level, as the Greenland example does.
    obs_levels = [(obs_x, obs_y)]
    for _ in range(1, args.levels):
        obs_levels.append((torch.as_tensor(model.mg.restrict_cell(cp.asarray(obs_levels[-1][0]))),
                           torch.as_tensor(model.mg.restrict_cell(cp.asarray(obs_levels[-1][1])))))

    # ---- the starting guess: uniform, deliberately wrong ----
    b_init = cp.full(b_true.shape, float(cp.median(b_true)), cp.float32)

    def reset(m):
        m.mg.sliding.beta.set(b_init)
        m.mg.state.H.set(m.mg[0].state.H_prev.data)

    print('\n=== inversion A: SURFACE-velocity objective (u_s) ===')
    reset(model)
    t0 = time.time()
    beta_surf, hist_surf = invert(model, args, obs_levels, True, 'surface')
    print(f'  done in {time.time() - t0:.1f} s')

    results = {'surface': (beta_surf, hist_surf)}

    if not args.no_compare:
        print('\n=== inversion B: DEPTH-AVERAGED objective (the pre-existing path) ===')
        print('    same observations, same schedule -- fitting |ubar| to surface data')
        reset(model)
        t0 = time.time()
        beta_bar, hist_bar = invert(model, args, obs_levels, False, 'depth-avg')
        print(f'  done in {time.time() - t0:.1f} s')
        results['depth-averaged'] = (beta_bar, hist_bar)

    # ---- final forward run at the recovered beta, for the fit plots ----
    mx, my = (o.detach() for o in run_forward(model, 0, beta_surf, True))
    mod_speed = torch.sqrt(mx ** 2 + my ** 2)
    obs_speed = torch.sqrt(obs_x ** 2 + obs_y ** 2)
    u_s_rec = model.mg[0].state.u_s.data.copy()
    u_b_rec = model.mg[0].state.u_b.data.copy()
    u, v = model.mg[0].state.u.data, model.mg[0].state.v.data
    ubar_rec = cp.hypot(0.5 * (u[:, :-1] + u[:, 1:]), 0.5 * (v[:-1, :] + v[1:, :]))

    # ---- scoring ----
    print('\n=== recovery against the known truth ===')
    bt = cp.asarray(b_true)
    interior = np.zeros(bt.shape, bool)
    interior[6:-6, 6:-6] = True
    print(f'  {"objective":>16} {"median |log10 ratio|":>22} {"speed RMSE":>13}')
    for name, (b, _) in results.items():
        r = np.abs(np.log10(np.maximum(cp.asnumpy(cp.asarray(b)), 1e-12) /
                            np.maximum(cp.asnumpy(bt), 1e-12)))[interior]
        sx, sy = (o.detach() for o in run_forward(model, 0, b, True))
        rmse = float(torch.sqrt(((torch.sqrt(sx ** 2 + sy ** 2) - obs_speed) ** 2).mean()))
        print(f'  {name:>16} {np.median(r):22.4f} {rmse:13.3f}')

    # ---- figures ----
    print('\nwriting figures...')
    out = args.outdir
    paths = [
        plotting.plot_beta(b_init, beta_surf, b_true, path=f'{out}/beta_surface.png'),
        plotting.plot_velocity(obs_speed, mod_speed, path=f'{out}/velocity.png'),
        plotting.plot_sliding_fraction(u_b_rec, ubar_rec, path=f'{out}/sliding_fraction.png'),
        plotting.plot_history(hist_surf, path=f'{out}/convergence.png'),
    ]
    if not args.no_compare:
        paths.append(plotting.plot_beta(b_init, results['depth-averaged'][0], b_true,
                                        path=f'{out}/beta_depth_averaged.png'))
    for p in paths:
        print(f'  {p}')


if __name__ == '__main__':
    main()
