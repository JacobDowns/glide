"""Forward DIVA against SSA over a series of time steps, on Greenland.

Answers three questions that are easy to conflate:

  1. **What does DIVA cost?**  Measured at a FIXED V-cycle count with the tolerances
     switched off, so it is pure per-cycle throughput and cannot be contaminated by one
     scheme meeting a tolerance sooner -- or, as happened on the first version of this
     benchmark, by SSA stalling just above the tolerance at level 2 and burning cycles
     while DIVA converged in one.  Timing a run to a tolerance that a scheme cannot reach
     measures the tolerance, not the scheme.
  2. **Does the cost hold up over a transient?**  A single cold-started solve is not
     representative -- later steps are warm-started from the previous one and are usually
     much cheaper.  This runs a real sequence of steps with the geometry evolving.
  3. **How far does each get?**  Reported separately from cost: the absolute residual
     after a fixed number of cycles (comparable across schemes, same units), and then with
     the production tolerances, the cycles used and whether it actually converged.  Note
     |r|/|r0| is NOT comparable between schemes -- their initial residuals differ.

  4. **Where do the two schemes actually diverge?**  Not "is DIVA different" -- it is, by
     construction -- but how the difference in surface speed and in ice volume develops as
     the transient runs.

The V-cycle counts come from parsing the solver's own report_norms output, which is the
only place they are exposed.

    uv run python examples/surface-inverse/benchmark_forward.py
    uv run python examples/surface-inverse/benchmark_forward.py --levels-list 3 2 --steps 5
"""
import argparse
import contextlib
import io
import os
import re
import time

import cupy as cp
import numpy as np
import pyproj
from scipy.ndimage import gaussian_filter

from glide.data import load_greenland_preprocessed
from glide.model import IceDynamics

_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'data')
_CACHE = os.path.abspath(_CACHE) if os.path.isdir(os.path.abspath(_CACHE)) else None

import plotting
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--levels-list', type=int, nargs='+', default=[4, 3, 2, 1],
                   help='multigrid levels to benchmark (0 = finest, 900 m)')
    p.add_argument('--n-levels', type=int, default=6)
    p.add_argument('--steps', type=int, default=10, help='time steps per run')
    p.add_argument('--dt', type=float, default=10.0)
    p.add_argument('--fixed-cycles', type=int, default=4,
                   help='V-cycles per step for the cost measurement (tolerances off)')
    p.add_argument('--outdir', default='benchmark_forward')
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


def set_solver(model, fixed_cycles=None):
    """fixed_cycles=K forces exactly K V-cycles (tolerances off) for a clean cost measure;
    None uses the production tolerances so convergence can be judged."""
    if fixed_cycles is not None:
        model.forward_solver.fas_options.set(
            coarsest_steps=200, pre_steps=10, post_steps=150, finest_steps=0,
            maximum_vcycles=fixed_cycles, relative_tolerance=0.0, absolute_tolerance=0.0,
            report_norms=True)
    else:
        model.forward_solver.fas_options.set(
            coarsest_steps=200, pre_steps=10, post_steps=150, finest_steps=0,
            maximum_vcycles=30, relative_tolerance=1e-2, absolute_tolerance=10.0,
            report_norms=True)      # the only place V-cycle counts are exposed


_VCYCLE = re.compile(r'V-cycle (\d+):')


def timed_step(model, level, dt):
    """One time step; returns (seconds, V-cycles used)."""
    buf = io.StringIO()
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        model.forward(cp.float32(0.0), cp.float32(dt), update_geometry=True)
    cp.cuda.Stream.null.synchronize()
    dt_s = time.perf_counter() - t0
    hits = _VCYCLE.findall(buf.getvalue())
    return dt_s, (int(hits[-1]) + 1 if hits else 0)


def diagnostics(model, level, stress_balance):
    g = model.mg[level]
    u, v = g.state.u.data, g.state.v.data
    ubar = cp.hypot(0.5 * (u[:, :-1] + u[:, 1:]), 0.5 * (v[:-1, :] + v[1:, :]))
    surf = g.state.u_s.data if stress_balance > 0.5 else ubar
    dx = g.dx
    return dict(volume=float(cp.sum(g.state.H.data)) * dx * dx * 1e-9,   # km^3
                max_speed=float(cp.nanmax(surf)),
                mean_speed=float(cp.nanmean(surf)))


def residual_norm(model, level, dt):
    ru, rv, rH = model.mg[level].forward_operators.compute_residual(
        cp.float32(dt), return_norms=True)
    return float(cp.sqrt(ru**2 + rv**2 + rH**2))


def run(args, dataset, level, stress_balance):
    # ---- cost: exactly FIXED_CYCLES V-cycles per step, tolerances off ----
    model = build(args, dataset, stress_balance)
    set_solver(model, fixed_cycles=args.fixed_cycles)
    model.set_top_level(level)
    times, resid = [], []
    for _ in range(args.steps):
        t, _ = timed_step(model, level, args.dt)
        times.append(t)
        resid.append(residual_norm(model, level, args.dt))
    shape = tuple(model.mg[level].state.H.data.shape)
    del model
    cp.get_default_memory_pool().free_all_blocks()

    # ---- convergence: production tolerances, on a fresh model ----
    model = build(args, dataset, stress_balance)
    set_solver(model, fixed_cycles=None)
    model.set_top_level(level)
    cycles, diag = [], []
    for _ in range(args.steps):
        _, c = timed_step(model, level, args.dt)
        cycles.append(c)
        diag.append(diagnostics(model, level, stress_balance))
    converged = [c < 30 for c in cycles]
    del model
    cp.get_default_memory_pool().free_all_blocks()
    return dict(times=times, resid=resid, cycles=cycles, converged=converged,
                diag=diag, shape=shape)


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    dataset = load_greenland_preprocessed(cache_dir=_CACHE)
    print(f'Greenland {dataset.ny} x {dataset.nx} at {dataset.dx} m, '
          f'{args.steps} steps of {args.dt} a, m = 1/3\n')

    results = {}
    print(f'COST -- exactly {args.fixed_cycles} V-cycles per step, tolerances off')
    hdr = (f'{"level":>6} {"shape":>13} {"scheme":>5} {"step 0":>9} {"steps 1+":>10} '
           f'{"per V-cycle":>12} {"|r| after":>11}')
    print(hdr); print('-' * len(hdr))
    for level in args.levels_list:
        for sb, name in ((0.0, 'SSA'), (1.0, 'DIVA')):
            r = run(args, dataset, level, sb)
            results[(level, name)] = r
            warm = r['times'][1:] or r['times']
            print(f'{level:>6} {str(r["shape"]):>13} {name:>5} {r["times"][0]:9.3f} '
                  f'{np.mean(warm):10.3f} {np.mean(warm)/args.fixed_cycles:12.4f} '
                  f'{np.mean(r["resid"][1:] or r["resid"]):11.3e}')
        rs, rd = results[(level, 'SSA')], results[(level, 'DIVA')]
        ws = np.mean(rs['times'][1:] or rs['times'])
        wd = np.mean(rd['times'][1:] or rd['times'])
        print(f'{"":>6} {"":>13} {"DIVA/SSA":>5} {"":>9} {wd/ws:10.2f}x')
        print()

    # Convergence is reported separately, and NOT as a ratio: the two schemes have
    # different initial residuals, so |r|/|r0| is not comparable between them.  What is
    # comparable is whether each met the production tolerance, and in how many cycles.
    print('CONVERGENCE -- production tolerances (rel 1e-2, abs 10.0, cap 30)')
    hdr2 = (f'{"level":>6} {"scheme":>5} {"V-cyc 0":>8} {"V-cyc 1+":>9} '
            f'{"steps converged":>16}')
    print(hdr2); print('-' * len(hdr2))
    for level in args.levels_list:
        for name in ('SSA', 'DIVA'):
            r = results[(level, name)]
            wcyc = r['cycles'][1:] or r['cycles']
            nconv = sum(r['converged'])
            print(f'{level:>6} {name:>5} {r["cycles"][0]:8d} {np.mean(wcyc):9.1f} '
                  f'{nconv:11d}/{len(r["converged"])}')
        print()

    # ---- how the two schemes diverge over the transient ----
    print(f'{"level":>6} {"volume drift DIVA-SSA":>24} {"mean surface speed":>28}')
    for level in args.levels_list:
        rs, rd = results[(level, 'SSA')], results[(level, 'DIVA')]
        v_s = rs['diag'][-1]['volume']; v_d = rd['diag'][-1]['volume']
        m_s = rs['diag'][-1]['mean_speed']; m_d = rd['diag'][-1]['mean_speed']
        print(f'{level:>6} {100*(v_d-v_s)/v_s:23.4f}% '
              f'  SSA {m_s:8.2f}  DIVA {m_d:8.2f}  ({100*(m_d-m_s)/m_s:+.2f}%)')

    # ---- figures ----
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
    for level in args.levels_list:
        for name, style in (('SSA', '--'), ('DIVA', '-')):
            r = results[(level, name)]
            axes[0].plot(r['times'], style, label=f'L{level} {name}')
            axes[1].plot(r['cycles'], style)
            axes[2].plot([d['mean_speed'] for d in r['diag']], style)
    axes[0].set_yscale('log'); axes[0].set_ylabel('seconds'); axes[0].set_title('wall-clock per step')
    axes[1].set_ylabel('V-cycles'); axes[1].set_title('V-cycles per step')
    axes[2].set_yscale('log'); axes[2].set_ylabel('m a$^{-1}$'); axes[2].set_title('mean surface speed')
    for a in axes:
        a.set_xlabel('time step')
    axes[0].legend(fontsize=7, ncol=2)
    p = f'{args.outdir}/forward_benchmark.png'
    fig.savefig(p, bbox_inches='tight'); plt.close(fig)
    print(f'\n  {p}')


if __name__ == '__main__':
    main()
