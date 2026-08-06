"""The four total derivatives of the DIVA cell-local closure.

compute_diva_derivs obtains, by five dual seedings of diva_coeffs_cell,

    d(eta_bar)/d(.),  d(beta_eff)/d(.),  d(u_s)/d(.)     for . in
        eps_mem^2, Ubar          -> the state transpose
        beta, u_c, m             -> the parameter gradients

which is everything the adjoint needs from the closure. The u_s row is for an objective built on
SURFACE velocity observations rather than the depth average: it supplies both the adjoint
right-hand side (the eps/Ubar entries) and the objective's explicit parameter dependence (the
beta/u_c/m entries), which a depth-averaged objective does not have at all: with these the transpose can be
applied without re-running the vertical quadrature, and without assuming the operator is
symmetric.

The check probes diva_coeffs_cell directly as a function of its two scalar inputs rather
than through a velocity field -- perturbing velocities would move eps_mem^2 and Ubar
together, so the partials could not be separated.

Step sizes matter here and are swept rather than guessed. eta_bar is O(1e2) while
d(eta_bar)/d(Ubar) is O(1e-3), so a small step changes eta_bar by only a few float32
ULPs and the finite difference is pure round-off. Each derivative is therefore compared
at several steps and judged on the best agreement -- the standard way to verify against
finite differences.

    uv run python tests/diva_derivs_test.py
"""
from pathlib import Path

import cupy as cp
import numpy as np

CUDA = Path(__file__).resolve().parents[1] / "glide" / "cuda"
from glide.operators import CUDA_FILES

PROBE = r'''
extern "C" __global__
void deriv_probe(const float* eps, const float* Ubar, const float* beta,
                 const float* u_cs, const float* ms,
                 float* eta_v, float* be_v, float* us_v,
                 float* deta_deps, float* deta_dU, float* dbe_deps, float* dbe_dU,
                 float* deta_dbeta, float* dbe_dbeta,
                 float* deta_duc, float* dbe_duc, float* deta_dm, float* dbe_dm,
                 float* dus_deps, float* dus_dU, float* dus_dbeta, float* dus_duc, float* dus_dm,
                 const float* zq, const float* wq,
                 float H_c, float B_c,
                 float u_reg, float wd, float law,
                 float glen_exp, float eps_reg, int n_sigma, float U_b_warm, int n)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    float e = eps[i], U = Ubar[i], bg = beta[i], u_c_c = u_cs[i], m = ms[i];

    int cap = 0;
    float eta_f, F1_f, F2_f, Ub_f, be_f, us_f;
    diva_coeffs_cell<float>(e, U, H_c,B_c,bg,u_c_c, m,u_reg,wd,law,
                            glen_exp,eps_reg,n_sigma,zq,wq,U_b_warm, eta_f,F1_f,F2_f,Ub_f,be_f,us_f, cap);
    eta_v[i]=eta_f; be_v[i]=be_f; us_v[i]=us_f;

    DualFloat a,f1d,b,c,d,usd;
    diva_coeffs_cell<DualFloat>({e,1.0f},{U,0.0f}, H_c,B_c,{bg,0.0f},{u_c_c,0.0f}, {m,0.0f},u_reg,wd,law,
                                glen_exp,eps_reg,n_sigma,zq,wq,U_b_warm, a,f1d,b,c,d,usd, cap);
    deta_deps[i]=a.d; dbe_deps[i]=d.d; dus_deps[i]=usd.d;

    diva_coeffs_cell<DualFloat>({e,0.0f},{U,1.0f}, H_c,B_c,{bg,0.0f},{u_c_c,0.0f}, {m,0.0f},u_reg,wd,law,
                                glen_exp,eps_reg,n_sigma,zq,wq,U_b_warm, a,f1d,b,c,d,usd, cap);
    deta_dU[i]=a.d; dbe_dU[i]=d.d; dus_dU[i]=usd.d;

    diva_coeffs_cell<DualFloat>({e,0.0f},{U,0.0f}, H_c,B_c,{bg,1.0f},{u_c_c,0.0f}, {m,0.0f},u_reg,wd,law,
                                glen_exp,eps_reg,n_sigma,zq,wq,U_b_warm, a,f1d,b,c,d,usd, cap);
    deta_dbeta[i]=a.d; dbe_dbeta[i]=d.d; dus_dbeta[i]=usd.d;

    diva_coeffs_cell<DualFloat>({e,0.0f},{U,0.0f}, H_c,B_c,{bg,0.0f},{u_c_c,1.0f}, {m,0.0f},u_reg,wd,law,
                                glen_exp,eps_reg,n_sigma,zq,wq,U_b_warm, a,f1d,b,c,d,usd, cap);
    deta_duc[i]=a.d; dbe_duc[i]=d.d; dus_duc[i]=usd.d;

    diva_coeffs_cell<DualFloat>({e,0.0f},{U,0.0f}, H_c,B_c,{bg,0.0f},{u_c_c,0.0f}, {m,1.0f},u_reg,wd,law,
                                glen_exp,eps_reg,n_sigma,zq,wq,U_b_warm, a,f1d,b,c,d,usd, cap);
    deta_dm[i]=a.d; dbe_dm[i]=d.d; dus_dm[i]=usd.d;
}
'''

RHO_I, GRAV = 917.0, 9.81
# Gauss-Legendre rule, the same one operators.py builds
_x, _w = np.polynomial.legendre.leggauss(8)
ZQ = cp.asarray(0.5*(_x + 1.0), dtype=cp.float32)
WQ = cp.asarray(0.5*_w, dtype=cp.float32)
H_C = 1000.0
B_C = (1e-16 ** -(1. / 3)) / (RHO_I * GRAV)
U_C_C, U_REG, WATER_DRAG = 100.0, 1.0, 0.0
GLEN_N, EPS_REG, N_SIGMA = 3.0, 1e-6, 8
GLEN_EXP = (1.0 - GLEN_N) / (2.0 * GLEN_N)

EPS0, U0, WARM = 1e-4, 20.0, 15.0
TOL = 1e-2


def build_module():
    src = "\n".join((CUDA / f).read_text() for f in CUDA_FILES)
    return cp.RawModule(code=src + PROBE, options=("--use_fast_math",))


def probe(mod, eps_arr, U_arr, beta_arr, u_c_arr, m_arr, law):
    n = len(eps_arr)
    out = [cp.zeros(n, cp.float32) for _ in range(18)]
    mod.get_function("deriv_probe")((n // 64 + 1,), (64,),
        (cp.asarray(eps_arr, cp.float32), cp.asarray(U_arr, cp.float32),
         cp.asarray(beta_arr, cp.float32), cp.asarray(u_c_arr, cp.float32),
         cp.asarray(m_arr, cp.float32), *out,
         ZQ, WQ,
         np.float32(H_C), np.float32(B_C),
         np.float32(U_REG), np.float32(WATER_DRAG), np.float32(law),
         np.float32(GLEN_EXP), np.float32(EPS_REG), np.int32(N_SIGMA),
         np.float32(WARM), n))
    return [cp.asnumpy(o) for o in out]


def best_agreement(mod, m, law, beta_g, which):
    """Compare one dual derivative against FD over a range of steps; return the best
    relative agreement and the step that achieved it."""
    best = (np.inf, None, None, None)
    for frac in (1e-3, 1e-2, 5e-2, 2e-1):
        eps_arr = np.array([EPS0] * 3, np.float32)
        U_arr = np.array([U0] * 3, np.float32)
        beta_arr = np.array([beta_g] * 3, np.float32)
        u_c_arr = np.array([U_C_C] * 3, np.float32)
        m_arr = np.array([m] * 3, np.float32)
        if which.endswith("deps"):
            h = EPS0 * frac; eps_arr[1] -= h; eps_arr[2] += h
        elif which.endswith("dbeta"):
            h = beta_g * frac; beta_arr[1] -= h; beta_arr[2] += h
        elif which.endswith("duc"):
            h = U_C_C * frac; u_c_arr[1] -= h; u_c_arr[2] += h
        elif which.endswith("dm"):
            h = m * frac; m_arr[1] -= h; m_arr[2] += h
        else:
            h = U0 * frac; U_arr[1] -= h; U_arr[2] += h

        (eta, be, us, de_de, de_dU, db_de, db_dU,
         de_db, db_db, de_duc, db_duc, de_dm, db_dm,
         du_de, du_dU, du_db, du_duc, du_dm) = probe(
                mod, eps_arr, U_arr, beta_arr, u_c_arr, m_arr, law)
        value = us if which.startswith("dus") else (eta if which.startswith("deta") else be)
        dual = {"deta_deps": de_de, "deta_dU": de_dU, "deta_dbeta": de_db,
                "deta_duc": de_duc, "deta_dm": de_dm,
                "dbe_deps": db_de, "dbe_dU": db_dU, "dbe_dbeta": db_db,
                "dbe_duc": db_duc, "dbe_dm": db_dm,
                "dus_deps": du_de, "dus_dU": du_dU, "dus_dbeta": du_db,
                "dus_duc": du_duc, "dus_dm": du_dm}[which][0]
        fd = (value[2] - value[1]) / (2 * h)
        rel = abs(dual - fd) / max(abs(fd), 1e-30)
        if rel < best[0]:
            best = (rel, dual, fd, frac)
    return best


def main():
    mod = build_module()
    for tag, m, law, beta_g in (("Weertman m=0.5", 0.5, 0.0, 0.05),
                                ("regularized Coulomb", 1.0, 1.0, 6.0)):
        print(f"{tag}:")
        # u_c only enters the Coulomb branch and m only the Weertman branch, so each
        # law exercises just one of the two.  The other is checked to be exactly zero.
        active = (("deta_duc", "dbe_duc", "dus_duc") if law > 0.5
                  else ("deta_dm", "dbe_dm", "dus_dm"))
        inert  = (("deta_dm", "dbe_dm", "dus_dm") if law > 0.5
                  else ("deta_duc", "dbe_duc", "dus_duc"))
        for which in ("deta_deps", "deta_dU", "dbe_deps", "dbe_dU",
                      "deta_dbeta", "dbe_dbeta",
                      "dus_deps", "dus_dU", "dus_dbeta") + active:
            rel, dual, fd, frac = best_agreement(mod, m, law, beta_g, which)
            print(f"  {which:<10} dual = {dual:+.6e}  FD = {fd:+.6e}  "
                  f"rel = {rel:.2e}  (best at step {frac:g})")
            assert rel < TOL, f"{tag} {which}: dual derivative disagrees with FD ({rel:.2e})"

        # The other law's parameter must come back identically zero -- not merely small.
        # A nonzero value here would mean the untaken branch is still contributing, which
        # would put a spurious search direction into an inversion over that parameter.
        for which in inert:
            _, dual, _, _ = best_agreement(mod, m, law, beta_g, which)
            print(f"  {which:<10} dual = {dual:+.6e}  (inert under this law)")
            assert dual == 0.0, f"{tag} {which}: expected exactly zero, got {dual:.6e}"

    print("\nOK: all closure derivatives agree with finite differences")


if __name__ == '__main__':
    main()
