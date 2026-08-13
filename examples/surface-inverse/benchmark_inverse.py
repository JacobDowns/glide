"""Cost of a Greenland inversion iteration, by level and by objective.

One optimizer iteration is a forward solve plus an adjoint solve plus a reduction, and
they do not scale alike -- so a single "seconds per iteration" number hides what you would
want to know before planning a run. This separates:

  * forward vs backward, per level, so the adjoint's share is visible;
  * the three objectives -- DIVA on surface velocity, DIVA on the depth average, and SSA --
    so the price of the surface path is isolated from the price of DIVA itself.

The surface objective costs more than the depth-averaged one for a specific reason: its
adjoint right-hand side needs the closure derivatives (compute_diva_derivs runs six dual
seedings of the whole column solve), and it adds an explicit parameter term. Whether that
is a little or a lot is the question this answers.

Cost is measured at a FIXED V-cycle count with tolerances off, for both the forward and
the adjoint solve.  Timing to a tolerance measures the tolerance whenever a scheme fails to
reach it -- and the DIVA adjoint does fail to reach it at the finer levels, burning its
whole cap, which made a first version of this benchmark report a 15.6x DIVA/SSA ratio that
was almost entirely cap and not cost.  Convergence is therefore reported separately, as the
fraction of iterations whose adjoint actually converged.

Cold-start iterations are reported separately from warm ones, since the first iteration at
each level carries the coefficient refresh from a standing start.

    uv run python examples/surface-inverse/benchmark_inverse.py
    uv run python examples/surface-inverse/benchmark_inverse.py --levels-list 3 2 --iters 6
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
from glide.model import IceDynamics
from glide.torch import glide_step

_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'data')
_CACHE = os.path.abspath(_CACHE) if os.path.isdir(os.path.abspath(_CACHE)) else None

from common import surface_velocity, depth_averaged_velocity, elastic_net
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--levels-list', type=int, nargs='+', default=[4, 3, 2, 1])
    p.add_argument('--n-levels', type=int, default=6)
    p.add_argument('--iters', type=int, default=8, help='optimizer iterations timed per case')
    p.add_argument('--dt', type=float, default=10.0)
    p.add_argument('--lr', type=float, default=1e-2)
    p.add_argument('--fixed-cycles', type=int, default=4,
                   help='V-cycles per solve for the cost measurement (tolerances off)')
    p.add_argument('--outdir', default='benchmark_inverse')
    return p.parse_args()


def build(args, dataset, stress_balance):
    ny, nx, dx = dataset.ny, dataset.nx, dataset.dx
    model = IceDynamics(n_levels=args.n_levels, ny=ny, nx=nx, dx=dx,
                        x0=dataset.x[0].item(), y0=dataset.y[0].item(),
                        crs=pyproj.CRS("EPSG:3413"))
    mg = model.mg
    thk = gaussian_filter(dataset.thickness.values, 1)
    mg.state.H.set(thk); mg.state.H_prev.set(thk)
    bed = gaussian_filter(dataset.bed.values, 1)
    mg.geometry.bed.set(bed)
    mg.geometry.depth.set(np.maximum(-bed, 0))
    mg.geometry.sigmoid_c.set(0.1); mg.geometry.sigmoid_k.set(3.0)

    B = cp.zeros((ny, nx), dtype=cp.float32)
    B.fill(1e-17 ** (-1.0 / 3.0) / (917 * 9.81))
    mg.rheology.B.set(B)
    mg.rheology.eps_reg.set(1e-6); mg.rheology.n.set(3.0)
    mg.rheology.stress_balance.set(stress_balance)
    mg.rheology.n_sigma.set(8.0); mg.rheology.eps_reg_shear.set(1e-12)

    mg.sliding.beta.set(cp.full((ny, nx), 2.5, cp.float32))
    mg.sliding.m.set(1. / 3.); mg.sliding.water_drag.set(1e-4)
    mg.calving.calving_rate.set(2000.0)
    mg.forcing.smb.set(dataset.smb.values)

    return model


def set_solvers(model, fixed_cycles=None):
    """fixed_cycles=K forces exactly K V-cycles on both solves (tolerances off), which is
    the only way to time cost rather than tolerance."""
    if fixed_cycles is not None:
        for s in (model.forward_solver, model.adjoint_solver):
            s.fas_options.set(coarsest_steps=200, pre_steps=10, post_steps=150,
                              finest_steps=0, maximum_vcycles=fixed_cycles,
                              relative_tolerance=0.0, absolute_tolerance=0.0,
                              report_norms=False)
    else:
        model.forward_solver.fas_options.set(
            coarsest_steps=200, pre_steps=10, post_steps=150, finest_steps=0,
            maximum_vcycles=30, relative_tolerance=1e-2, absolute_tolerance=10.0,
            report_norms=False)
        model.adjoint_solver.fas_options.set(
            coarsest_steps=200, pre_steps=10, post_steps=150, finest_steps=0,
            maximum_vcycles=30, relative_tolerance=1e-2, absolute_tolerance=1e-5,
            report_norms=False)


def sync():
    cp.cuda.Stream.null.synchronize()


def timed_iteration(model, level, args, log_beta, obs, use_surface, ssa):
    """One optimizer iteration, timed as (forward, backward)."""
    g = model.mg[level]
    H_prev = torch.tensor(g.state.H_prev.data)
    bed = torch.tensor(g.geometry.bed.data)
    smb = torch.tensor(g.forcing.smb.data)
    beta = torch.exp(log_beta)

    sync(); t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        out = glide_step(cp.float32(0.0), cp.float32(args.dt), model, level,
                         H_prev, bed, beta, smb, return_u_s=use_surface)
    mx, my = (surface_velocity(out[0], out[1], out[4]) if use_surface
              else depth_averaged_velocity(out[0], out[1]))
    J_data = (abs(mx - obs[0])).mean() + (abs(my - obs[1])).mean()
    J_l2, J_tv = elastic_net(log_beta, g.dx)
    J = J_data + J_l2 + J_tv
    sync(); t_fwd = time.perf_counter() - t0

    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        J.backward()
    sync(); t_bwd = time.perf_counter() - t0
    return t_fwd, t_bwd, float(J.detach())


def run_case(args, dataset, level, tag, stress_balance, use_surface, fixed_cycles):
    model = build(args, dataset, stress_balance)
    set_solvers(model, fixed_cycles)
    mg = model.mg
    u_obs = cp.array(dataset.vx.values, dtype=cp.float32)
    v_obs = cp.array(dataset.vy.values, dtype=cp.float32)
    for _ in range(level):
        u_obs, v_obs = mg.restrict_cell(u_obs), mg.restrict_cell(v_obs)
    obs = (torch.as_tensor(u_obs), torch.as_tensor(v_obs))

    log_beta = torch.log(torch.tensor(mg[level].sliding.beta.data, device='cuda'))
    log_beta.requires_grad_()
    optimizer = torch.optim.RMSprop([log_beta], lr=args.lr)

    fwd, bwd, conv = [], [], []
    for _ in range(args.iters):
        optimizer.zero_grad()
        f, b, _ = timed_iteration(model, level, args, log_beta, obs, use_surface, stress_balance < 0.5)
        optimizer.step()
        fwd.append(f); bwd.append(b)
        conv.append(bool(getattr(model, 'last_backward_converged', True)))
    shape = tuple(mg[level].state.H.data.shape)
    del model
    cp.get_default_memory_pool().free_all_blocks()
    return dict(fwd=fwd, bwd=bwd, conv=conv, shape=shape)


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    dataset = load_greenland_preprocessed(cache_dir=_CACHE)
    print(f'Greenland {dataset.ny} x {dataset.nx} at {dataset.dx} m, '
          f'{args.iters} optimizer iterations per case, dt = {args.dt} a')
    print(f'cost at exactly {args.fixed_cycles} V-cycles per solve; "adj conv" is from a '
          f'separate pass at production tolerances\n')

    cases = (('DIVA surface', 1.0, True),
             ('DIVA depth-avg', 1.0, False),
             ('SSA', 0.0, False))
    results = {}
    hdr = (f'{"level":>6} {"shape":>13} {"objective":>15} {"cold s":>8} '
           f'{"warm fwd":>9} {"warm bwd":>9} {"warm tot":>9} {"bwd share":>10} {"adj conv":>9}')
    print(hdr); print('-' * len(hdr))
    for level in args.levels_list:
        for tag, sb, surf in cases:
            r = run_case(args, dataset, level, tag, sb, surf, args.fixed_cycles)
            rc = run_case(args, dataset, level, tag, sb, surf, None)   # convergence pass
            r['conv'] = rc['conv']
            results[(level, tag)] = r
            wf, wb = np.mean(r['fwd'][1:]), np.mean(r['bwd'][1:])
            cold = r['fwd'][0] + r['bwd'][0]
            print(f'{level:>6} {str(r["shape"]):>13} {tag:>15} {cold:8.3f} '
                  f'{wf:9.3f} {wb:9.3f} {wf+wb:9.3f} {100*wb/(wf+wb):9.1f}% '
                  f'{sum(r["conv"]):5d}/{len(r["conv"])}')
        base = results[(level, 'SSA')]
        b_tot = np.mean(base['fwd'][1:]) + np.mean(base['bwd'][1:])
        for tag, _, _ in cases[:2]:
            r = results[(level, tag)]
            tot = np.mean(r['fwd'][1:]) + np.mean(r['bwd'][1:])
            print(f'{"":>6} {"":>13} {tag+" / SSA":>15} {"":>8} {"":>9} {"":>9} '
                  f'{tot/b_tot:8.2f}x')
        print()

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    colors = {'DIVA surface': '#4c72b0', 'DIVA depth-avg': '#dd8452', 'SSA': '#333333'}
    cells = [np.prod(results[(l, 'SSA')]['shape']) for l in args.levels_list]
    for tag, _, _ in cases:
        tot = [np.mean(results[(l, tag)]['fwd'][1:]) + np.mean(results[(l, tag)]['bwd'][1:])
               for l in args.levels_list]
        share = [100 * np.mean(results[(l, tag)]['bwd'][1:]) /
                 (np.mean(results[(l, tag)]['fwd'][1:]) + np.mean(results[(l, tag)]['bwd'][1:]))
                 for l in args.levels_list]
        axes[0].plot(cells, tot, 'o-', color=colors[tag], label=tag)
        axes[1].plot(cells, share, 'o-', color=colors[tag], label=tag)
    axes[0].set_xscale('log'); axes[0].set_yscale('log')
    axes[0].set_xlabel('cells'); axes[0].set_ylabel('seconds / iteration')
    axes[0].set_title('cost per optimizer iteration'); axes[0].legend(fontsize=8)
    axes[1].set_xscale('log'); axes[1].set_xlabel('cells')
    axes[1].set_ylabel('% of iteration'); axes[1].set_title('adjoint share')
    p = f'{args.outdir}/inverse_benchmark.png'
    fig.savefig(p, bbox_inches='tight'); plt.close(fig)
    print(f'  {p}')


if __name__ == '__main__':
    main()
