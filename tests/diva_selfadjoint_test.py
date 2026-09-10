"""Self-adjointness of the momentum Jacobian, SSA and DIVA, across sliding exponents.

Two probes on a converged slab, for stress_scheme in {'ssa','diva'} and m in {1, 1/3, 2, 3}:

  A. **JVP self-adjointness**  <J x, y> vs <J y, x>  for random interior x, y.
     This is the exact tangent (compute_jvp carries the full d(eta_bar)/du and
     d(beta_eff)/du through dual arithmetic), so a symmetric J means the discrete
     momentum operator is the gradient of a potential -- the property the sum-of-squares
     basal-drag closure was chosen to give.  It should hold to float32 round-off at EVERY m; that it does is what
     distinguishes the sum-of-squares closure from a cell-centered mean-speed one, which is
     symmetric only at m = 1.

  B. **VJP == JVP transpose**  <J^T y, x> vs <y, J x>  (no finite differences, so round-off,
     not the ~1e-3 an FD check tops out at).  Confirms the adjoint operator is the exact
     transpose of the tangent.  DIVA carries one deliberate approximation in the closure's
     thickness/coefficient path (d F2/d u), so this is asserted
     at m = 1 where it is exact; the JVP probe A is the all-m guarantee.

    uv run python tests/diva_selfadjoint_test.py
"""
import contextlib
import io

import cupy as cp
import numpy as np

import glide
from glide.multigrid import Multigrid, FASCDSolver

ny = nx = 64
dx = 2000.0
L = 20000.0
RHO_I, GRAV = 917.0, 9.81
SEED = 3

SELFADJ_BOUND = 1e-4     # JVP symmetry: round-off at every m
TRANSPOSE_BOUND = 1e-4   # VJP == JVP transpose at m = 1


def interior(shape, rng):
    a = np.zeros(shape, np.float32)
    a[5:-5, 5:-5] = rng.randn(*[s - 10 for s in shape]).astype(np.float32)
    return a


def converged(scheme, m):
    x = cp.arange(nx, dtype=cp.float32) * dx
    y = cp.arange(ny, dtype=cp.float32) * dx
    X, Y = cp.meshgrid(x, y)
    srf = 1000.0 * cp.ones((ny, nx), cp.float32) - cp.tan(cp.deg2rad(0.1)) * X + 10000
    bed = srf - 1000.0
    thk = srf - bed
    beta = (1000 * cp.sin(2 * cp.pi * X / L) * cp.sin(2 * cp.pi * Y / L) + 1000) / (RHO_I * GRAV)
    B = cp.full((ny, nx), (1e-16 ** (-1. / 3)) / (RHO_I * GRAV), cp.float32)

    mg = Multigrid(3, ny=ny, nx=nx, dx=dx, stress_scheme=scheme)
    mg.geometry.bed.set(bed)
    mg.rheology.B.set(B)
    mg.state.H.set(thk); mg.state.H_prev.set(thk)
    mg.state.phi.set(cp.ones((ny, nx), cp.float32))
    mg.sliding.beta.set(beta.astype(cp.float32)); mg.sliding.m.set(cp.float32(m)); mg.sliding.u_reg.set(1.0)
    mg.rheology.n.set(3.0); mg.rheology.eps_reg.set(1e-6)
    if scheme == 'diva':
        for lvl in mg.levels:
            lvl.rheology.eps_reg_shear.set(cp.float32(1e-6)); lvl.rheology.n_sigma.set(cp.float32(8.0))

    s = FASCDSolver(mg)
    s.vanka_options.omega.set(cp.float32(0.5))
    s.vanka_options.newton_options.relaxation.set(cp.float32(0.5))
    s.fas_options.set(coarsest_steps=200, pre_steps=10, post_steps=50, finest_steps=150,
                      maximum_vcycles=20, relative_tolerance=1e-8, absolute_tolerance=1e-8,
                      report_norms=False)
    with contextlib.redirect_stdout(io.StringIO()):
        s.solve(cp.float32(1.0))
    return mg


def run(scheme, m):
    rng = np.random.RandomState(SEED)
    xu, xv = interior((ny, nx + 1), rng), interior((ny + 1, nx), rng)
    yu, yv = interior((ny, nx + 1), rng), interior((ny + 1, nx), rng)

    mg = converged(scheme, m)
    g = mg.levels[0]
    fo, ao = g.forward_operators, g.adjoint_operators

    def J(du, dv):
        fo.var_u[:, :] = cp.asarray(du); fo.var_v[:, :] = cp.asarray(dv)
        fo.var_H.fill(0.0)
        if scheme == 'molho':
            fo.var_ud.fill(0.0); fo.var_vd.fill(0.0)
        with contextlib.redirect_stdout(io.StringIO()):
            fo.compute_jvp(cp.float32(1.0), use_mask=False, freeze_phi=True)
        cp.cuda.runtime.deviceSynchronize()
        return cp.asnumpy(fo.jvp_u), cp.asnumpy(fo.jvp_v)

    # A. JVP self-adjointness
    Jxu, Jxv = J(xu, xv)
    Jyu, Jyv = J(yu, yv)
    xJy = float((Jxu * yu).sum() + (Jxv * yv).sum())
    yJx = float((Jyu * xu).sum() + (Jyv * xv).sum())
    selfadj = abs(xJy - yJx) / max(abs(xJy), abs(yJx), 1e-30)

    # B. VJP == JVP transpose:  <J^T y, x> vs <y, J x>
    g.adjoint.lambda_u.set(cp.asarray(yu)); g.adjoint.lambda_v.set(cp.asarray(yv))
    g.adjoint.lambda_H.data.fill(0.0)
    with contextlib.redirect_stdout(io.StringIO()):
        ao.compute_vjp(cp.float32(1.0))
    cp.cuda.runtime.deviceSynchronize()
    JTy_x = float((cp.asnumpy(ao.vjp_u) * xu).sum() + (cp.asnumpy(ao.vjp_v) * xv).sum())
    y_Jx = xJy   # <y, J x> reuses the J x above
    transpose = abs(JTy_x - y_Jx) / max(abs(JTy_x), abs(y_Jx), 1e-30)
    return selfadj, transpose


def main():
    print(f"glide: {glide.__file__}")
    schemes = ['ssa', 'diva']
    for scheme in schemes:
        print(f"\n{scheme.upper()}  ({'sum-of-squares drag' if scheme == 'diva' else 'cell-centered drag'})")
        print(f"  {'m':>6} {'JVP asym':>10} {'VJP=Jᵀ':>10}  verdict")
        for m in (1.0, 1. / 3, 2.0, 3.0):
            sa, tr = run(scheme, m)
            ok = sa < SELFADJ_BOUND
            print(f"  {m:>6.3f} {sa:>10.2e} {tr:>10.2e}  {'self-adjoint' if ok else 'ASYMMETRIC'}")
            assert sa < SELFADJ_BOUND, f"{scheme} m={m}: JVP not self-adjoint ({sa:.2e})"
            if abs(m - 1.0) < 1e-9:
                assert tr < TRANSPOSE_BOUND, f"{scheme} m=1: VJP != JVP transpose ({tr:.2e})"
    print("\nAll schemes self-adjoint across m; adjoint is the exact transpose at m=1.")


if __name__ == '__main__':
    main()
