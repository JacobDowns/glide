"""Plot a saved beta field from any GLIDE Greenland inversion.

Reads a beta_opt.nc written by an inversion, runs ONE forward solve with it, and
produces the same figures the inversion scripts do.  Useful for looking at a result
after the fact -- including runs made before the plotting existed.

    # the SSA / depth-averaged inversion from examples/greenland/greenland_inverse.py
    uv run python examples/surface-inverse/plot_saved_beta.py \\
        --beta ../../inverse/level_0/beta_opt.nc --ssa --outdir inverse_greenland_june

    # a DIVA result, compared against surface velocity
    uv run python examples/surface-inverse/plot_saved_beta.py --beta path/to/beta_opt.nc
"""
import argparse
import contextlib
import io
import os

import cupy as cp
import numpy as np
import pyproj
import torch
import xarray as xr
from scipy.ndimage import gaussian_filter

from glide.data import load_greenland_preprocessed

# The loader caches into ./data relative to the CWD, so running from a new directory
# re-downloads 165 MB.  Point every run at the workspace-level cache instead.
_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'data')
_CACHE = os.path.abspath(_CACHE) if os.path.isdir(
    os.path.abspath(_CACHE)) else None
from glide.model import IceDynamics
from glide.torch import glide_step

from common import surface_velocity, depth_averaged_velocity
import plotting


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--beta', required=True, help='beta_opt.nc to plot')
    p.add_argument('--levels', type=int, default=6)
    p.add_argument('--dt', type=float, default=10.0)
    p.add_argument('--ssa', action='store_true',
                   help='the field came from an SSA run; compare the depth average')
    p.add_argument('--outdir', default='inverse_greenland_saved')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    beta_da = xr.open_dataset(args.beta)
    beta = cp.asarray(beta_da[list(beta_da.data_vars)[0]].values, dtype=cp.float32)
    print(f'beta from {args.beta}: {beta.shape}, '
          f'median {float(cp.median(beta)):.3f}, range {float(beta.min()):.3g}'
          f'-{float(beta.max()):.3g}')

    dataset = load_greenland_preprocessed(cache_dir=_CACHE)
    ny, nx, dx = dataset.ny, dataset.nx, dataset.dx
    if beta.shape != (ny, nx):
        raise SystemExit(f'beta is {beta.shape} but the dataset is {(ny, nx)}; pass the '
                         f'level-0 field, or one matching this dataset')

    model = IceDynamics(n_levels=args.levels, ny=ny, nx=nx, dx=dx,
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
    mg.rheology.stress_balance.set(0.0 if args.ssa else 1.0)
    mg.rheology.n_sigma.set(8.0); mg.rheology.eps_reg_shear.set(1e-12)

    mg.sliding.beta.set(beta)
    mg.sliding.m.set(1. / 3.); mg.sliding.water_drag.set(1e-4)
    mg.calving.calving_rate.set(2000.0)
    mg.forcing.smb.set(dataset.smb.values)

    model.forward_solver.fas_options.set(
        coarsest_steps=200, pre_steps=10, post_steps=150, finest_steps=0,
        relative_tolerance=1e-2, absolute_tolerance=10.0, report_norms=False)

    print('running one forward solve at the saved beta...')
    g = mg[0]
    H_prev = torch.tensor(g.state.H_prev.data)
    bed_t = torch.tensor(g.geometry.bed.data)
    smb_t = torch.tensor(g.forcing.smb.data)
    with contextlib.redirect_stdout(io.StringIO()):
        out = glide_step(cp.float32(0.0), cp.float32(args.dt), model, 0,
                         H_prev, bed_t, torch.tensor(beta), smb_t,
                         return_u_s=not args.ssa)
    if args.ssa:
        mx, my = depth_averaged_velocity(out[0], out[1])
    else:
        mx, my = surface_velocity(out[0], out[1], out[4])
    mod_speed = torch.sqrt(mx.detach() ** 2 + my.detach() ** 2)

    u_obs = torch.as_tensor(cp.asarray(dataset.vx.values, dtype=cp.float32))
    v_obs = torch.as_tensor(cp.asarray(dataset.vy.values, dtype=cp.float32))
    obs_speed = torch.sqrt(u_obs ** 2 + v_obs ** 2)
    valid = (obs_speed > 0.01).float()

    m = cp.asnumpy(valid) > 0
    resid = cp.asnumpy(mod_speed) - cp.asnumpy(obs_speed)
    print(f'  speed misfit over observed ice: median |model-obs| = '
          f'{np.median(np.abs(resid[m])):.2f} m/a, RMSE = '
          f'{np.sqrt((resid[m] ** 2).mean()):.2f} m/a')

    label = 'depth-averaged' if args.ssa else 'surface'
    paths = [
        plotting.plot_beta(None, beta, mask=valid, path=f'{args.outdir}/beta.png'),
        plotting.plot_velocity(obs_speed, mod_speed, mask=valid,
                               path=f'{args.outdir}/velocity.png',
                               labels=('observed surface speed',
                                       f'modelled {label} speed')),
    ]
    if not args.ssa:
        u, v = g.state.u.data, g.state.v.data
        ubar = cp.hypot(0.5 * (u[:, :-1] + u[:, 1:]), 0.5 * (v[:-1, :] + v[1:, :]))
        paths.append(plotting.plot_sliding_fraction(g.state.u_b.data, ubar, mask=valid,
                                                   path=f'{args.outdir}/sliding_fraction.png'))
    for p in paths:
        print(f'  {p}')


if __name__ == '__main__':
    main()
