"""Figures for a surface-velocity inversion.

Four plots, each answering one question:

    beta      -- did the inversion recover the traction field?
    velocity  -- does the model match the observations it was fitted to?
    sliding   -- how much of the flow is sliding rather than deformation?
    history   -- did the optimizer converge, and which term dominates?

Colour conventions: magnitude fields (speed, beta) use a single ordered
perceptually-uniform ramp, light to dark, on a log scale because both span orders
of magnitude.  Signed misfits use a diverging map with a neutral midpoint, always
symmetric about zero so that white means zero and never means "middle of the data".
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize

SPEED_CMAP = 'viridis'      # ordered, perceptually uniform, CVD-safe
BETA_CMAP = 'magma'
DIVERGING = 'RdBu_r'        # neutral midpoint; only ever used symmetric about 0

# Greenland at 900 m is 3040 x 1696, so a 3-inch panel at 130 dpi threw away most of the
# field.  240 dpi with wider panels keeps the outlet glaciers and shear margins legible.
DPI = 240

plt.rcParams.update({
    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    'legend.frameon': False,
})


def _asnumpy(a):
    """numpy view of a numpy / cupy / torch-cuda array, whichever the caller had."""
    if hasattr(a, 'detach'):                       # torch, possibly on the GPU
        return a.detach().cpu().numpy()
    if hasattr(a, 'get'):                          # cupy
        return a.get()
    return np.asarray(a)


def _field(ax, data, title, cmap, norm, mask=None, extent=None):
    d = _asnumpy(data).astype(np.float64)
    if mask is not None:
        d = np.where(_asnumpy(mask) > 0, d, np.nan)
    im = ax.imshow(d, cmap=cmap, norm=norm, origin='upper',
                   extent=extent, interpolation='nearest')
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def _log_norm(*fields, floor=1e-3):
    vals = np.concatenate([_asnumpy(f).ravel() for f in fields])
    vals = vals[np.isfinite(vals) & (vals > floor)]
    if vals.size == 0:
        return Normalize()
    return LogNorm(vmin=max(np.percentile(vals, 1), floor),
                   vmax=np.percentile(vals, 99.5))


def _sym_norm(d, pct=98):
    d = _asnumpy(d)
    d = d[np.isfinite(d)]
    lim = np.percentile(np.abs(d), pct) if d.size else 1.0
    return Normalize(vmin=-lim, vmax=lim)


def plot_beta(beta_init, beta_opt, beta_true=None, mask=None, path='beta.png'):
    """Recovered traction, next to the starting guess and (twin runs only) the truth.

    beta_init may be None -- for a field loaded from disk there is no starting guess to
    show, and inventing a flat one would imply this run began there.
    """
    panels = [(beta_opt, 'recovered')]
    if beta_init is not None:
        panels.insert(0, (beta_init, 'initial guess'))
    if beta_true is not None:
        panels.insert(0, (beta_true, 'true'))
    norm = _log_norm(*[p[0] for p in panels])

    ncol = len(panels) + (1 if beta_true is not None else 0)
    fig, axes = plt.subplots(1, ncol, figsize=(4.4 * ncol, 4.8), constrained_layout=True)
    axes = np.atleast_1d(axes)

    for ax, (d, t) in zip(axes, panels):
        im = _field(ax, d, rf'$\beta$, {t}', BETA_CMAP, norm, mask)
    fig.colorbar(im, ax=axes[:len(panels)], shrink=0.85, label=r'$\beta$')

    if beta_true is not None:
        # log-ratio, so over- and under-estimation read symmetrically
        r = np.log10(np.maximum(_asnumpy(beta_opt), 1e-12) /
                     np.maximum(_asnumpy(beta_true), 1e-12))
        im2 = _field(axes[-1], r, r'$\log_{10}(\beta_{\rm rec}/\beta_{\rm true})$',
                     DIVERGING, _sym_norm(r), mask)
        fig.colorbar(im2, ax=axes[-1], shrink=0.85, label='dex')

    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_velocity(obs_speed, mod_speed, mask=None, path='velocity.png',
                  labels=('observed surface speed', 'modelled surface speed')):
    """Fit to the data the inversion was actually given."""
    norm = _log_norm(obs_speed, mod_speed)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), constrained_layout=True)

    im = _field(axes[0], obs_speed, labels[0], SPEED_CMAP, norm, mask)
    _field(axes[1], mod_speed, labels[1], SPEED_CMAP, norm, mask)
    fig.colorbar(im, ax=axes[:2], shrink=0.85, label='m a$^{-1}$')

    d = _asnumpy(mod_speed) - _asnumpy(obs_speed)
    if mask is not None:
        d = np.where(_asnumpy(mask) > 0, d, np.nan)
    im2 = _field(axes[2], d, 'model $-$ observed', DIVERGING, _sym_norm(d))
    fig.colorbar(im2, ax=axes[2], shrink=0.85, label='m a$^{-1}$')

    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_sliding_fraction(u_b, u_bar, mask=None, path='sliding_fraction.png'):
    """U_b/|ubar| -- the fraction of the depth-averaged flow that is sliding.

    Bounded in [0,1] by the kinematics alone, and readable without reference to any
    scheme: 1 is plug flow (all sliding, the SSA limit), 0 is a frozen bed where every
    bit of motion is internal deformation.  It follows from

        Ubar = U_b + tau_b*I2   =>   U_b/Ubar = 1 - tau_b*I2/Ubar

    with I2 > 0 and tau_b >= 0, so deformation can only add to sliding, never subtract
    (notes/diva_numerics.md 2.5).  Preferred over u_s/|ubar| because that ratio has no
    fixed upper bound -- the uniform-slab value 1.25 is a reference, not a limit, and it
    reaches 1.43 on Greenland -- whereas this one is a fraction and reads directly.
    """
    u_b, u_bar = _asnumpy(u_b), _asnumpy(u_bar)
    frac = np.where(u_bar > 1e-6, u_b / np.maximum(u_bar, 1e-12), np.nan)
    frac = np.clip(frac, 0.0, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)
    im = _field(axes[0], frac, r'sliding fraction $U_b/|\bar{u}|$', SPEED_CMAP,
                Normalize(vmin=0.0, vmax=1.0), mask)
    fig.colorbar(im, ax=axes[0], shrink=0.85, label='fraction')

    v = frac
    if mask is not None:
        v = np.where(_asnumpy(mask) > 0, v, np.nan)
    v = v[np.isfinite(v)]
    axes[1].hist(v, bins=60, color='#4c72b0')
    for x, lab in ((0.0, 'frozen bed'), (1.0, 'plug flow')):
        axes[1].axvline(x, color='0.35', lw=1.5, ls='--')
        axes[1].annotate(lab, xy=(x, 0.94), xycoords=('data', 'axes fraction'),
                         xytext=(6 if x == 0 else -6, 0), textcoords='offset points',
                         fontsize=8, color='0.35',
                         ha='left' if x == 0 else 'right')
    axes[1].set_xlabel(r'$U_b/|\bar{u}|$')
    axes[1].set_ylabel('cells')
    axes[1].set_title('how much of the flow is sliding')

    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_history(history, path='convergence.png'):
    """Objective by iteration, one panel per multigrid level.

    Single y-axis, log scale; the three terms share units of the objective so they
    belong on the same axis.
    """
    levels = sorted(history.keys(), reverse=True)
    fig, axes = plt.subplots(1, len(levels), figsize=(2.9 * len(levels), 3.0),
                             constrained_layout=True, sharey=True)
    axes = np.atleast_1d(axes)

    series = [('J', '#333333', 2.0), ('J_data', '#4c72b0', 1.6),
              ('J_reg', '#dd8452', 1.6)]
    for ax, lvl in zip(axes, levels):
        h = history[lvl]
        for name, color, lw in series:
            if name in h and len(h[name]):
                ax.plot(h[name], color=color, lw=lw, label=name)
        ax.set_yscale('log')
        ax.set_xlabel('iteration')
        ax.set_title(f'level {lvl}')
    axes[0].set_ylabel('objective')
    axes[0].legend(loc='upper right')

    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path
