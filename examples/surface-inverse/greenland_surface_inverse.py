"""Greenland inverse simulation against SURFACE velocity observations, under DIVA.

This is examples/greenland/greenland_inverse.py with one physical change: the observed
velocity mosaic is compared against the model's SURFACE velocity rather than its
depth-averaged velocity.

Under SSA the two are the same field, so the distinction never arose.  Under DIVA they
differ by the vertical shear.  Measured over observed ice on Greenland at full resolution:
median 1.038, 90th percentile 1.088, max 1.43, with 35% of cells above 1.05.  The uniform
slab with a frozen bed gives I1/I2 = (n+2)/(n+1) = 1.25 at n = 3, which is a reference
value and NOT a bound -- the ratio follows the actual viscosity profile.  Fitting |ubar| to surface data therefore asks the inversion
to absorb that difference into beta, biasing the recovered traction high exactly where
the ice is slow: the interior, which is most of the ice sheet.

The gradient reaches beta by two routes that a depth-averaged objective does not have:

    dJ/dbeta = dJ/dbeta|explicit + lambda^T dr/dbeta,   (dr/dx)^T lambda = -dJ/dx

  * dJ/dx is scattered from cells to velocity facets through the closure's stored
    du_s/d(eps_mem^2) and du_s/d(Ubar);
  * dJ/dbeta|explicit is new outright -- u_s depends on beta directly through the
    column closure, which ubar never did.

Both are handled inside GlideStep when return_u_s=True.  See notes/diva_adjoint_map.md
section 7 and tests/diva_surface_gradient_test.py.

    uv run python examples/surface-inverse/greenland_surface_inverse.py
    uv run python examples/surface-inverse/greenland_surface_inverse.py --coarsest 5 --epochs 50

NOTE: the closure is differentiated with respect to thickness as well as velocity
(notes/diva_adjoint_map.md section 12), so this gradient is exact for the coupled (u,v,H)
block.  Auxiliary fields are still held fixed -- in particular the grounded fraction
phi(H) -- so cases dominated by grounding-line migration are outside what it covers.
"""
import argparse
import contextlib
import io
import os
import time

import cupy as cp
import numpy as np
import pyproj
import torch
from scipy.ndimage import gaussian_filter

from glide.data import load_greenland_preprocessed

# The loader caches into ./data relative to the CWD, so running from a new directory
# re-downloads 165 MB.  Point every run at the workspace-level cache instead.
_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'data')
_CACHE = os.path.abspath(_CACHE) if os.path.isdir(
    os.path.abspath(_CACHE)) else None
from glide.io import VTIWriter
from glide.model import IceDynamics
from glide.torch import glide_step

from common import surface_velocity, depth_averaged_velocity, elastic_net
import plotting


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--levels', type=int, default=6)
    p.add_argument('--coarsest', type=int, default=5,
                   help='coarsest level to start the inversion on')
    p.add_argument('--finest', type=int, default=0,
                   help='finest level to finish on (raise it for a cheaper run)')
    p.add_argument('--epochs', type=int, default=50, help='iterations per level')
    p.add_argument('--lr', type=float, default=1e-2)
    p.add_argument('--dt', type=float, default=10.0)
    p.add_argument('--depth-averaged', action='store_true',
                   help='fit |ubar| instead of u_s, i.e. the pre-existing behaviour')
    p.add_argument('--ssa', action='store_true',
                   help='run SSA instead of DIVA (implies --depth-averaged)')
    p.add_argument('--outdir', default='inverse_greenland_surface')
    p.add_argument('--vti', action='store_true', help='also write vti/pvd output')
    return p.parse_args()


def build_model(args, dataset):
    ny, nx, dx = dataset.ny, dataset.nx, dataset.dx
    model = IceDynamics(n_levels=args.levels, ny=ny, nx=nx, dx=dx,
                        x0=dataset.x[0].item(), y0=dataset.y[0].item(),
                        crs=pyproj.CRS("EPSG:3413"))
    mg = model.mg

    ### Initialize state
    thk = gaussian_filter(dataset.thickness.values, 1)
    mg.state.H.set(thk)
    mg.state.H_prev.set(thk)

    ### Initialize geometry
    bed = gaussian_filter(dataset.bed.values, 1)
    mg.geometry.bed.set(bed)
    mg.geometry.depth.set(np.maximum(-bed, 0))
    mg.geometry.sigmoid_c.set(0.1)
    mg.geometry.sigmoid_k.set(3.0)

    ### Initialize rheology
    B = cp.zeros((ny, nx), dtype=cp.float32)
    B.fill(1e-17 ** (-1.0 / 3.0) / (917 * 9.81))
    mg.rheology.B.set(B)
    mg.rheology.eps_reg.set(1e-6)
    mg.rheology.n.set(3.0)

    ### Stress balance: DIVA unless asked otherwise
    mg.rheology.stress_balance.set(0.0 if args.ssa else 1.0)
    mg.rheology.n_sigma.set(8.0)
    mg.rheology.eps_reg_shear.set(1e-12)

    ### Initialize sliding
    beta = cp.zeros((ny, nx), dtype=cp.float32)
    beta.fill(2.5)
    mg.sliding.beta.set(beta)
    mg.sliding.m.set(1. / 3.)
    mg.sliding.water_drag.set(1e-4)

    ### Initialize calving
    mg.calving.calving_rate.set(2000.0)

    ### Initialize forcing
    mg.forcing.smb.set(dataset.smb.values)

    ### Multigrid solver parameters
    # Relative 1e-2 with an absolute floor, as in the original example.  Do not push the
    # tolerances below the float32 residual floor (~1e-6 relative): the solver would then
    # spend every V-cycle, report NOT converged, and GlideStep would zero the gradient --
    # which looks like a stalled optimizer, not like a solver problem.
    model.forward_solver.fas_options.set(
        coarsest_steps=200, pre_steps=10,
        post_steps=150, finest_steps=0,
        relative_tolerance=1e-2, absolute_tolerance=10.0,
        report_norms=False)
    # post_steps=30, NOT the 150 this example used to inherit from the forward SSA
    # configuration.  A smoother is meant to kill high-frequency error in a handful of
    # sweeps; 150 is seven times this solver's own default and far past what it is for.
    # SSA is merely wasteful there, DIVA is not: its adjoint smoother acquires a growing
    # mode when linearised about the off-equilibrium states a V-cycle hands its coarse
    # levels, and that mode needs ~20-30 sweeps to emerge (so 10 pre-sweeps look fine
    # while 150 post-sweeps amplified the residual 8e6-fold at level 3 and took the solve
    # to NaN).  30 is both cheaper in total work and inside the safe window.
    #
    # vanka_config.smoother_growth_check is the actual safety net -- it stops a sweep that
    # is raising the residual -- so this value is now a performance choice rather than a
    # correctness one.  See notes/diva_numerics.md 5.3.
    #
    # maximum_vcycles defaults to 10, which a handful of inversion iterations were
    # exhausting -- and a capped adjoint is reported as NOT converged, on which GlideStep
    # zeroes the gradient and that step moves beta on the regularizer alone.
    model.adjoint_solver.fas_options.set(
        coarsest_steps=200, pre_steps=10,
        post_steps=30, finest_steps=0, maximum_vcycles=30,
        relative_tolerance=1e-2, absolute_tolerance=1e-5,  # the adjoint variable is
        report_norms=False)                                # small in magnitude
    return model


def run_forward(model, level, beta_t, dt, want_surface):
    g = model.mg[level]
    H_prev = torch.tensor(g.state.H_prev.data)
    bed = torch.tensor(g.geometry.bed.data)
    smb = torch.tensor(g.forcing.smb.data)
    with contextlib.redirect_stdout(io.StringIO()):
        out = glide_step(cp.float32(0.0), cp.float32(dt), model, level,
                         H_prev, bed, beta_t, smb, return_u_s=want_surface)
    u, v, H, mask = out[:4]
    if want_surface:
        return surface_velocity(u, v, out[4])
    return depth_averaged_velocity(u, v)


def main():
    args = parse_args()
    use_surface = not (args.depth_averaged or args.ssa)
    os.makedirs(args.outdir, exist_ok=True)

    print('loading the preprocessed Greenland dataset...')
    dataset = load_greenland_preprocessed(cache_dir=_CACHE)
    model = build_model(args, dataset)
    mg = model.mg
    print(f'  {dataset.ny} x {dataset.nx} at {dataset.dx} m, '
          f'{"DIVA" if not args.ssa else "SSA"}, '
          f'objective on {"surface" if use_surface else "depth-averaged"} velocity')

    ### Load velocity data ###
    u_obs = cp.array(dataset.vx.values, dtype=cp.float32)
    v_obs = cp.array(dataset.vy.values, dtype=cp.float32)

    # Build hierarchy of observations
    observation_levels = [(u_obs, v_obs)]
    for j in range(1, args.levels):
        observation_levels.append((mg.restrict_cell(observation_levels[-1][0]),
                                   mg.restrict_cell(observation_levels[-1][1])))

    t = cp.float32(0.0)   # Dummy time, which we don't use here
    dt = args.dt
    history = {}

    # Index of coarsest grid to start solving the inverse problem at
    log_beta = torch.log(torch.tensor(mg[args.coarsest].sliding.beta.data, device='cuda'))

    # Solve the inverse problem at progressively finer levels
    for level in range(args.coarsest, args.finest - 1, -1):
        log_beta.requires_grad_()

        u_obs_l, v_obs_l = (torch.as_tensor(o) for o in observation_levels[level])
        u_mask = abs(u_obs_l) > 0.01
        v_mask = abs(v_obs_l) > 0.01
        dx = mg[level].dx

        optimizer = torch.optim.RMSprop([log_beta], lr=args.lr)
        hist = {'J': [], 'J_data': [], 'J_reg': []}

        writer = None
        if args.vti:
            writer = VTIWriter(f'{args.outdir}/level_{level}/vti', base='greenland',
                               dx=dx, static_fields={'U_obs': [u_obs_l, v_obs_l]},
                               dynamic_fields={'beta': mg[level].sliding.beta,
                                               'U': [mg[level].state.u, mg[level].state.v],
                                               'u_s': mg[level].state.u_s})
            writer.initialize(mg[level])

        t0 = time.time()
        for j in range(args.epochs):
            optimizer.zero_grad()
            beta = torch.exp(log_beta)

            # The model counterpart of the observations: the surface velocity vector
            # under DIVA, the depth average otherwise.
            mx, my = run_forward(model, level, beta, dt, use_surface)

            # L1 Objective function, masked by valid data
            J_data = (abs(mx - u_obs_l) * u_mask).mean() + (abs(my - v_obs_l) * v_mask).mean()
            J_l2, J_tv = elastic_net(log_beta, dx)
            J = J_data + J_l2 + J_tv

            J.backward()
            # The model's own flag, not `grad == 0`: the regularizer keeps the total
            # gradient nonzero even when the model gradient has been zeroed.
            if not getattr(model, 'last_backward_converged', True):
                print(f'  WARNING level {level} iter {j}: adjoint did not converge, so '
                      f'this step used regularization gradient only')
            # Belt and braces. A non-finite gradient poisons log_beta on the very next
            # optimizer step and every later objective reads nan, which is a slow and
            # confusing way to find out; stop at the first one instead.
            if not torch.isfinite(log_beta.grad).all():
                raise RuntimeError(
                    f'non-finite gradient at level {level}, iter {j} '
                    f'(J={float(J.detach()):.4g}, beta range '
                    f'{float(beta.min()):.4g}-{float(beta.max()):.4g}). The adjoint solve '
                    f'is the usual source; rerun with '
                    f'adjoint_solver.fas_options.set(report_norms=True) to see its V-cycles.')
            optimizer.step()

            hist['J'].append(float(J.detach()))
            hist['J_data'].append(float(J_data.detach()))
            hist['J_reg'].append(float((J_l2 + J_tv).detach()))
            print(f"Level {level}, Iter. {j}/{args.epochs} | J: {float(J.detach()):.2f}, "
                  f"J_data: {float(J_data.detach()):.2f}, J_L1: {float(J_tv.detach()):.2f}, "
                  f"J_L2: {float(J_l2.detach()):.2f}")
            if writer is not None:
                writer.append(mg[level], time=j)
                writer.write_pvd()

        history[level] = hist
        print(f'  level {level} done in {time.time() - t0:.1f} s')

        os.makedirs(f'{args.outdir}/level_{level}', exist_ok=True)
        mg[level].sliding.beta.to_dataarray().to_netcdf(
            f'{args.outdir}/level_{level}/beta_opt.nc')

        if level > args.finest:
            log_beta = torch.tensor(mg.prolongate_cell(
                cp.asarray(log_beta.detach()), method='bilinear'))

    # ---------------------------------------------------------------- figures ----
    final = args.finest
    beta_opt = torch.exp(log_beta.detach())
    mx, my = (o.detach() for o in run_forward(model, final, beta_opt, dt, use_surface))
    mod_speed = torch.sqrt(mx ** 2 + my ** 2)
    u_obs_f, v_obs_f = (torch.as_tensor(o) for o in observation_levels[final])
    obs_speed = torch.sqrt(u_obs_f ** 2 + v_obs_f ** 2)
    valid = (obs_speed > 0.01).float()

    print('\nwriting figures...')
    out = args.outdir
    paths = [
        plotting.plot_beta(cp.full(cp.asarray(beta_opt).shape, 2.5, cp.float32),
                           beta_opt, mask=valid, path=f'{out}/beta.png'),
        plotting.plot_velocity(obs_speed, mod_speed, mask=valid,
                               path=f'{out}/velocity.png'),
        plotting.plot_history(history, path=f'{out}/convergence.png'),
    ]
    if not args.ssa:
        g = mg[final]
        u, v = g.state.u.data, g.state.v.data
        ubar = cp.hypot(0.5 * (u[:, :-1] + u[:, 1:]), 0.5 * (v[:-1, :] + v[1:, :]))
        paths.append(plotting.plot_sliding_fraction(g.state.u_b.data, ubar, mask=valid,
                                                   path=f'{out}/sliding_fraction.png'))
        r = cp.asnumpy(g.state.u_s.data / cp.maximum(ubar, 1e-6))
        m = cp.asnumpy(valid) > 0
        print(f'  surface/depth-averaged speed ratio over observed ice: '
              f'median {np.nanmedian(r[m]):.3f}, 99th pct {np.nanpercentile(r[m], 99):.3f}')
    for p in paths:
        print(f'  {p}')


if __name__ == '__main__':
    main()
