"""The four total derivatives of the DIVA cell-local closure.

compute_diva_derivs obtains, by three dual seedings of diva_coeffs_cell,

    d(eta_bar)/d(eps_mem^2),  d(eta_bar)/d(Ubar),  d(eta_bar)/d(beta),
    d(beta_eff)/d(eps_mem^2), d(beta_eff)/d(Ubar), d(beta_eff)/d(beta)

which is everything the adjoint needs from the closure -- the first four for the state
transpose, the last two for the parameter gradient: with these the transpose can be
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
                 float* eta_v, float* be_v,
                 float* deta_deps, float* deta_dU, float* dbe_deps, float* dbe_dU,
                 float* deta_dbeta, float* dbe_dbeta,
                 float H_c, float B_c, float u_c_c,
                 float m, float u_reg, float wd, float law,
                 float glen_exp, float eps_reg, int n_sigma, float U_b_warm, int n)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    float e = eps[i], U = Ubar[i], bg = beta[i];

    float eta_f, F2_f, Ub_f, be_f;
    diva_coeffs_cell<float>(e, U, H_c,B_c,bg,u_c_c, m,u_reg,wd,law,
                            glen_exp,eps_reg,n_sigma,U_b_warm, eta_f,F2_f,Ub_f,be_f);
    eta_v[i]=eta_f; be_v[i]=be_f;

    DualFloat a,b,c,d;
    diva_coeffs_cell<DualFloat>({e,1.0f},{U,0.0f}, H_c,B_c,{bg,0.0f},u_c_c, m,u_reg,wd,law,
                                glen_exp,eps_reg,n_sigma,U_b_warm, a,b,c,d);
    deta_deps[i]=a.d; dbe_deps[i]=d.d;

    diva_coeffs_cell<DualFloat>({e,0.0f},{U,1.0f}, H_c,B_c,{bg,0.0f},u_c_c, m,u_reg,wd,law,
                                glen_exp,eps_reg,n_sigma,U_b_warm, a,b,c,d);
    deta_dU[i]=a.d; dbe_dU[i]=d.d;

    diva_coeffs_cell<DualFloat>({e,0.0f},{U,0.0f}, H_c,B_c,{bg,1.0f},u_c_c, m,u_reg,wd,law,
                                glen_exp,eps_reg,n_sigma,U_b_warm, a,b,c,d);
    deta_dbeta[i]=a.d; dbe_dbeta[i]=d.d;
}
'''

RHO_I, GRAV = 917.0, 9.81
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


def probe(mod, eps_arr, U_arr, beta_arr, m, law):
    n = len(eps_arr)
    out = [cp.zeros(n, cp.float32) for _ in range(8)]
    mod.get_function("deriv_probe")((n // 64 + 1,), (64,),
        (cp.asarray(eps_arr, cp.float32), cp.asarray(U_arr, cp.float32),
         cp.asarray(beta_arr, cp.float32), *out,
         np.float32(H_C), np.float32(B_C), np.float32(U_C_C),
         np.float32(m), np.float32(U_REG), np.float32(WATER_DRAG), np.float32(law),
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
        if which.endswith("deps"):
            h = EPS0 * frac; eps_arr[1] -= h; eps_arr[2] += h
        elif which.endswith("dbeta"):
            h = beta_g * frac; beta_arr[1] -= h; beta_arr[2] += h
        else:
            h = U0 * frac; U_arr[1] -= h; U_arr[2] += h

        eta, be, de_de, de_dU, db_de, db_dU, de_db, db_db = probe(
                mod, eps_arr, U_arr, beta_arr, m, law)
        value = {"deta_deps": eta, "deta_dU": eta, "deta_dbeta": eta,
                 "dbe_deps": be, "dbe_dU": be, "dbe_dbeta": be}[which]
        dual = {"deta_deps": de_de, "deta_dU": de_dU, "deta_dbeta": de_db,
                "dbe_deps": db_de, "dbe_dU": db_dU, "dbe_dbeta": db_db}[which][0]
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
        for which in ("deta_deps", "deta_dU", "dbe_deps", "dbe_dU",
                      "deta_dbeta", "dbe_dbeta"):
            rel, dual, fd, frac = best_agreement(mod, m, law, beta_g, which)
            print(f"  {which:<10} dual = {dual:+.6e}  FD = {fd:+.6e}  "
                  f"rel = {rel:.2e}  (best at step {frac:g})")
            assert rel < TOL, f"{tag} {which}: dual derivative disagrees with FD ({rel:.2e})"

    print("\nOK: all six closure derivatives agree with finite differences")


if __name__ == '__main__':
    main()
