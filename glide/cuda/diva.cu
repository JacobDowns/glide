/*==================================================
  ============= DIVA COEFFICIENTS ==================
  ==================================================

  Depth-integrated viscosity approximation (Goldberg, 2011, J. Glaciol. 57(201)).
  DIVA reuses the SSA elliptic operator; only two coefficients change definition,
  and this file diagnoses both from the current state:

     eta_bar : depth-averaged effective viscosity.  The effective strain rate gains
               the vertical shear terms eps_xz, eps_yz, so eta varies with depth.
     F2      : the vertical shear integral, F2 = int eta^-1 ((s-z)/H)^2 dz = omega/H
               in Goldberg's notation (his eq 35).

  The sliding law then closes the system for the basal speed.  DIVA supplies a
  kinematic relation independent of the law,

     U_bar = U_b + |tau_b|*F2,

  and the law supplies |tau_b| = f(U_b), so U_b solves

     F(U_b) = U_b + f(U_b)*F2( f(U_b) ) - U_bar = 0.

  F2 depends on U_b through tau_b -- more drag, more shear, thinner ice, larger F2 --
  so this is a genuine root find for every sliding law, linear included.  It is
  cell-local: Goldberg's eqs 38-39, "solved at a location along the base independently
  of other locations".  No spatial coupling, so no multigrid transfer is involved, and
  nothing in this file is compiled or reached unless GLIDE_DIVA = 1 (stress_scheme='diva').

  The full derivation, the convergence analysis and the measurements behind the
  iteration counts are in notes/diva_numerics.md sections 2.5-2.7 and 4.
  ==================================================*/

// Thin wrappers over membrane_eps_sq<T> (viscosity.cu), which is the single definition of
// the membrane strain-rate invariant shared with populate_viscosity.  DIVA needs it
// UNREGULARIZED so the vertical shear terms can be added before eta is formed -- see
// diva_coeffs_cell -- whereas populate_viscosity adds eps_reg immediately.  That is the only
// difference between the two uses, and it is why the shared function returns the raw sum.
__device__ __forceinline__
float get_membrane_eps_sq(
    const float* __restrict__ u,
    const float* __restrict__ v,
    int i, int j,
    float dx,
    int ny, int nx){

    return membrane_eps_sq<float>(u, v, nullptr, nullptr, i, j, dx, ny, nx);
}

__device__ __forceinline__
DualFloat get_membrane_eps_sq(
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ du,
    const float* __restrict__ dv,
    int i, int j,
    float dx,
    int ny, int nx){

    return membrane_eps_sq<DualFloat>(u, v, du, dv, i, j, dx, ny, nx);
}


__device__ __forceinline__
void diva_add_W(float* __restrict__ W, int i, int j, int ny, int nx, float val){
    // Bounds-guarded global accumulate.  Used by the DIVA VJP to push
    // lambda_row * d(r_row)/d(coefficient) onto the owning CELL.  Splitting the
    // transpose at the cell like this keeps both halves within a +/-1 reach: the
    // composite row->facet dependence is +/-2, but row->cell and cell->facet are each
    // +/-1, so neither step needs a wider halo.
    if (i >= 0 && i < ny && j >= 0 && j < nx) atomicAdd(&W[i*nx + j], val);
}

__device__ __forceinline__ float diva_primal(float x)     { return x; }
__device__ __forceinline__ float diva_primal(DualFloat x) { return x.v; }

// Seed a scalar of either type from a float.  A dual gets a zero perturbation, which is
// correct for the closure's warm start: the converged sensitivity is fixed by the fixed
// point, not by where the iteration started.
__device__ __forceinline__ void diva_from_float(float& x, float v)     { x = v; }
__device__ __forceinline__ void diva_from_float(DualFloat& x, float v) { x = {v, 0.0f}; }

// Number of closure-Newton steps to take AFTER the primal convergence criterion fires.  The
// outer closure step uses a PRIMAL (frozen) denominator Fp, so the "one extra step" fixed-point
// trick makes the first derivative exact (N'(root) ~ 1 - F_Ub/Fp ~ 0).  float/DualFloat carry at
// most a first derivative, so one extra step is exact and bit-identical to before.  The inner
// level-viscosity Newton uses a dual denominator and is self-correcting, so it needs no analogue.
__device__ __forceinline__ int diva_n_post(float)      { return 1; }
__device__ __forceinline__ int diva_n_post(DualFloat)  { return 1; }

template <typename T>
__device__ __forceinline__
T get_diva_c_of_U(
    T U, T beta_grounded, T m, float u_reg, float water_drag,
    T u_c, float sliding_law){

    // The drag coefficient c(U) of get_diva_drag_coeff (stress.cu), templated on the
    // scalar type so the same expression serves the diagnostic kernel (T = float) and
    // the JVP/derivative kernels, where T = DualFloat carries d/d(seeded direction).
    //
    // Every input a gradient is ever wanted for is typed T, not float: the velocity
    // (through U), and the three sliding parameters beta, m and u_c.  Seeding any one of
    // them and reading the .d of the outputs is then the whole parameter gradient, with
    // no second derivation to keep in sync with this expression.
    T U_sq_reg = U*U + u_reg;

    T c;
    if (sliding_law < 0.5f) {
        c = beta_grounded * __powf(U_sq_reg, 0.5f*(m - 1.0f));
    } else {
        c = beta_grounded / (sqrtf(U_sq_reg) + u_c);
    }

    return c + water_drag;
}

// Solve one level's viscosity fixed point.  Glen's law plus the DIVA shear ansatz
// eps_xz = tau_b*zeta/(2*eta) leave eta on both sides, so it is a scalar root problem:
//
//   eta = 0.5*B*[ A + k/eta^2 ]^p  =  G(eta),   k = (tau_b*zeta/2)^2,  p = glen_exp
//
// A is the regularized membrane strain rate; which regularization goes in it is the caller's
// choice, and diva_coeffs_cell calls this twice with two different ones (see 3.2).
//
// Solved by Newton on eta - G(eta).  G' lies in (0, 2|p|) subset (0,1), so 1 - G' is bounded
// away from zero and the step needs no safeguarding, for any p.
//
// For n = 3 this is a depressed cubic, 8A*eta^3 + 8k*eta - B^3 = 0, with a Cardano solution --
// deliberately NOT used: its two cube roots nearly cancel when the shear dominates and it
// loses three digits in float32, precisely in the regime of interest.  See 4.3.
//
// k must be typed T, not taken from primals: it carries tau_b, hence the seeded perturbation,
// so a primal-only k would silently drop d(eta)/d(tau_b).
//
// Returns false if the iteration ran out of steps, which the caller must report.
template <typename T>
__device__ __forceinline__ bool diva_level_viscosity(
    float B_c, T A, T k_shear, float glen_exp,
    int iters_max, float eta_tol, float near_tol, float stag_gain,
    T& eta_out)
{
    // Start from the smaller of the two asymptotic limits (shear-free and shear-dominated).
    // Both overestimate, since dropping either positive term of the cubic inflates eta, so the
    // min is the better bracket and roughly halves the iteration count against using the
    // shear-free value alone.
    T eta_k = 0.5f*B_c*__powf(A, glen_exp);
    if (diva_primal(k_shear) > 0.0f) {
        T eta_shear = __powf(0.5f*B_c*__powf(k_shear, glen_exp),
                             1.0f/(1.0f + 2.0f*glen_exp));
        eta_k = fminf(eta_k, eta_shear);
    }

    bool ok = false;
    bool primal_ok = false;
    float prev_step = 3.4e38f;
    for (int e = 0; e < iters_max; ++e) {
        T s_sh = k_shear/(eta_k*eta_k);           // s = k/eta^2
        T E_sh = A + s_sh;                        // E = A + s
        T G_of = 0.5f*B_c*__powf(E_sh, glen_exp); // G(eta)
        // G'(eta) = -2p * (s/E) * G(eta)/eta   (exact, not the at-the-root form)
        T Gp = (-2.0f*glen_exp)*(s_sh/E_sh)*(G_of/eta_k);
        T step = (eta_k - G_of)/(1.0f - Gp);
        eta_k = eta_k - step;
        // Relative, so the criterion does not depend on the scale of eta -- which spans four
        // orders of magnitude across depth and stiffness.
        float rel = fabsf(diva_primal(step))/fmaxf(fabsf(diva_primal(eta_k)), 1e-30f);
        if (primal_ok) { ok = true; break; }      // this was the derivative's pass
        if (rel <= eta_tol || (rel < near_tol && rel >= stag_gain*prev_step)) primal_ok = true;
        prev_step = rel;
    }
    eta_out = eta_k;
    return ok;
}

template <typename T>
__device__ void diva_coeffs_cell(
    T eps_mem_sq, T U_bar,                       // the two velocity-dependent inputs
    T H_c, float B_c, T beta_grounded, T u_c_c,
    T m, float u_reg, float water_drag, float sliding_law,
    float glen_exp, float eps_reg, float eps_reg_shear,
    int n_sigma,
    const float* __restrict__ zeta_q,     // quadrature nodes on [0,1], Gauss-Legendre
    const float* __restrict__ w_q,        // matching weights, summing to 1
    float U_b_warm,
    T& eta_bar_out, T& F1_out, T& F2_out, T& U_b_out, T& beta_eff_out, T& u_s_out,
    int& cap_flags_out)          // bit 0: closure Newton hit its cap; bit 1: an eta solve did
{
    // Given U_bar, produce the two coefficients the momentum operator needs.  One unknown --
    // how much of U_bar is sliding rather than internal deformation -- fixed by
    //
    //     F(U_b) = U_b + f(U_b)*F2( f(U_b) ) - U_bar = 0        (Goldberg eq 34)
    //              ^^^^   ^^^^^^^^^^^^^^^^^
    //            sliding      deformation
    //
    // with f the sliding law (tau_b = f(U_b) = c(U_b)*U_b) and F2 = H*int zeta^2/eta the
    // vertical shear integral (his omega/H).  Both outputs then follow:
    //
    //     eta_bar  = int eta(zeta) dzeta
    //     beta_eff = c(U_b)/(1 + c(U_b)*F2)                     (eq 41)
    //
    // F1 = H*int (zeta/eta) dzeta is the first shear moment, accumulated in the same loop.
    // It is not needed by the momentum balance -- F2 is -- but it gives the SURFACE velocity,
    // u_s = u_b + tau_b*F1 (Arthern eq 10), which is what velocity observations measure.  For
    // SSA that coincides with the depth average; under DIVA it does not.
    //
    // TWO REGULARIZATIONS, which is deliberate.  The level viscosity is solved twice per
    // quadrature node, from the same shear term but two different regularized membrane strain
    // rates: eps_reg for eta_bar, which the membrane operator sees, and the much smaller
    // eps_reg_shear for F1 and F2.  Rationale at the level solve below; setting the two equal
    // reduces this exactly to a single viscosity.  Motivation and measurements in
    // notes/diva_numerics.md 3.2; SSA is untouched, since it uses neither F1/F2 nor this
    // closure.
    //
    // The quadrature rule is passed IN rather than built here.  It is Gauss-Legendre: the
    // integrands are zeta/eta and zeta^2/eta, and eta ~ zeta^(1-n) in the shear-dominated
    // limit, so they behave like zeta^n and zeta^(n+1) -- polynomials for integer n, which
    // Gauss-Legendre integrates EXACTLY with ceil((n+2)/2) nodes.  Midpoint needed 32+ nodes
    // for what 3 or 4 Gauss nodes deliver (notes/diva_numerics.md 3.1).  Nodes and weights
    // come from numpy on the host, so any n_sigma works without a table here.
    //
    // Newton on F, with the FULL slope
    //
    //     F'(U_b) = 1 + f'*F2 + f*f' * dF2/dtau_b
    //
    // Every term is non-negative (monotone law => f, f' >= 0; F2 > 0; dF2/dtau_b > 0 because
    // more drag means more shear means thinner ice), so F' >= 1: the root is unique and no
    // safeguarding or relaxation is needed.  All three terms matter -- dropping the third
    // makes the iteration 2-cycle at high basal drag (notes/diva_numerics.md 4.1).
    //
    //   closure Newton   solve F(U_b) = 0
    //     n_sigma loop   quadrature for eta_bar, F2 and dF2/dtau_b -- a sum, not a solve
    //       eta Newton   per level, eta against the shear ansatz (see that loop)
    //
    // The quadrature sits inside the Newton loop because F2 depends on U_b.
    //
    // Shared by compute_diva_coeffs (T = float) and the JVP/derivative kernels
    // (T = DualFloat).  Newton DENOMINATORS are taken from primals even when T is dual: by the
    // implicit function theorem the converged root and its sensitivity do not depend on the
    // step size used to reach them, so an inexact denominator costs iterations, never
    // accuracy.  A term dropped from a residual has no such licence.
    // Both loops are ADAPTIVE: they run to a tolerance and stop, with the counts below as
    // backstops rather than the operating cost.  Typical usage is far lower -- most sigma
    // levels exit in 2 to 5, and a warm-started closure in 1 to 2.
    const int eta_iters_max    = 12;
    const int newton_iters_max = 20;
    const float eta_tol     = 1e-7f;    // on the relative Newton step in eta
    const float closure_tol = 1e-7f;    // on |F| relative to the velocity scale
    const float near_tol    = 1e-4f;    // arm stagnation detection below this
    const float stag_gain   = 0.5f;     // a step that shrank by less than this has stalled
    //
    // Three properties of the termination that are not obvious, all three necessary:
    //
    //  1. Accepted on a tolerance OR on STAGNATION.  __powf under --use_fast_math leaves a
    //     few ulp in G(eta), so the correction bottoms out in a limit cycle instead of
    //     reaching zero; a pure tolerance would have to be tuned above a parameter-dependent
    //     floor.  Stagnation is a RATIO test -- a step that failed to shrink by stag_gain has
    //     stalled -- rather than merely a non-decreasing one: at the floor the jitter can
    //     drift downward for many iterations, which a monotonicity test sits through (it cost
    //     13 of the 12 permitted iterations once eps_reg_shear made the shear term dominant).
    //     Both loops converge quadratically, so below near_tol the next step should be roughly
    //     squared, and a ratio anywhere near 1 is the floor and not slow progress.
    //
    //  2. Exactly ONE MORE iteration after the criterion fires.  The criterion is on the
    //     primal and the seeded derivative rides one step behind it.  Newton's map N has
    //     N'(root) = 0, so a further step leaves the value at the root while replacing the
    //     derivative with d(root)/d(seed) exactly.
    //
    //  3. Failure to converge is REPORTED, via cap_flags_out.  A loop that silently caps is
    //     worse than a fixed count, because it looks converged.

    cap_flags_out = 0;

    T U_b; diva_from_float(U_b, U_b_warm);
    T coeff = get_diva_c_of_U<T>(U_b, beta_grounded, m, u_reg, water_drag, u_c_c, sliding_law);
    T tau_b = coeff * U_b;

    T eta_avg = T();
    T F1_c = T();
    T F2_c = T();

    // Breaks only immediately AFTER a quadrature, never after a step, so eta_bar, F2 and
    // beta_eff are always mutually consistent with the U_b returned.

    int newton_used = 0;
    bool closure_ok = false;
    const int n_post = diva_n_post(T());   // extra Newton steps after the convergence criterion fires
    int post = -1;                         // -1 until the criterion fires, then counts steps down
    float closure_prev = 3.4e38f;
    for (int it = 0; ; ++it) {
        coeff = get_diva_c_of_U<T>(U_b, beta_grounded, m, u_reg, water_drag, u_c_c, sliding_law);
        tau_b = coeff * U_b;

        eta_avg = T();
        F1_c = T();
        F2_c = T();
        float dF2_dtau = 0.0f;              // primal only; see the Newton step below
        float tau_p = diva_primal(tau_b);

        for (int k = 0; k < n_sigma; ++k) {
            float zeta = zeta_q[k];
            float wq   = w_q[k];

            // ---- per-level viscosities ----
            //
            // TWO of them, from the same shear term but two different regularizations of the
            // membrane strain rate (notes/diva_numerics.md 3.2):
            //
            //   eta_mem, with eps_reg        -> eta_bar, which the 2-D membrane operator sees
            //   eta_sh,  with eps_reg_shear  -> F1 and F2, the shear moments
            //
            // eta_bar must carry eps_reg because at zero shear it has to reproduce SSA's
            // viscosity EXACTLY, or DIVA stops being a strict generalization of SSA -- and it
            // is bounded by 0.5*B*eps_reg^p for free, so the membrane operator's conditioning
            // cannot get worse than SSA's.
            //
            // The moments must carry the much smaller eps_reg_shear because eps_reg was sized
            // for membrane strain rates and swamps the far smaller vertical shear one.  Adding
            // to the strain rate can only LOWER eta, so an oversized regularization makes the
            // column spuriously SOFT and F2 too large -- 9% at 44 kPa and 60x in nearly
            // stagnant ice, measured in tests/diva_slab_test.py against the analytic slab.
            //
            // The second solve costs 0.5% of the forward solve, measured -- eta_mem's fixed
            // point is dominated by eps_reg and converges in 2-3 steps against eta_sh's 8.
            // When the two regularizations are set equal the two solves coincide and this
            // reduces to the single-viscosity form exactly, which is the whole of what would
            // be needed to adopt a single DIVA-wide eps_reg if SSA's ever changes (5.10).
            T k_shear = tau_b*(0.5f*zeta);
            k_shear = k_shear*k_shear;
            T A_sh  = eps_mem_sq + eps_reg_shear;
            T A_mem = eps_mem_sq + eps_reg;

            T eta_k, eta_mem;
            bool ok_sh  = diva_level_viscosity<T>(B_c, A_sh, k_shear, glen_exp,
                              eta_iters_max, eta_tol, near_tol, stag_gain, eta_k);
            bool ok_mem = diva_level_viscosity<T>(B_c, A_mem, k_shear, glen_exp,
                              eta_iters_max, eta_tol, near_tol, stag_gain, eta_mem);
            if (!ok_sh || !ok_mem) cap_flags_out |= 2;

            eta_avg = eta_avg + wq*eta_mem;
            F1_c    = F1_c + wq*zeta/eta_k;
            F2_c    = F2_c + wq*zeta*zeta/eta_k;

            // d(F2)/d(tau_b) for the closure Newton's slope, accumulated here because
            // everything it needs is already in registers.  From the implicit function theorem
            // on the converged level root eta = G(eta; tau_b):
            //
            //   d(eta)/d(tau_b) = G_tau/(1 - G_eta),
            //       G_tau = 2p*s*eta/(E*tau_b),   G_eta = -2p*s/E
            //
            // s/tau_b is written as tau_b*zeta^2/(4*eta^2) so nothing divides by tau_b, making
            // the expression correctly zero at tau_b = 0 rather than 0/0.  1 - G_eta lies in
            // (1/3, 1] for n = 3, so the denominator never pinches.
            float eta_p = diva_primal(eta_k);
            float s_p   = diva_primal(k_shear)/(eta_p*eta_p);
            float E_p   = diva_primal(A_sh) + s_p;
            float dnm   = 1.0f + 2.0f*glen_exp*(s_p/E_p);       // = 1 - G_eta
            float deta_dtau = glen_exp*tau_p*zeta*zeta/(2.0f*eta_p*E_p*dnm);
            dF2_dtau += wq*zeta*zeta*(-deta_dtau/(eta_p*eta_p));
        }
        // H is the ONLY place thickness enters the closure: the quadrature runs over
        // zeta in [0,1], so the shear moments carry one factor of H and everything else
        // -- the level viscosities, the sliding law, the closure root -- depends on H
        // only through them.  Typing H_c as T therefore captures the whole path
        //     H -> I1, I2 -> U_b, tau_b -> eta_bar, beta_eff, u_s.
        F1_c     = F1_c * H_c;
        F2_c     = F2_c * H_c;
        // Primal only: this is a Newton DENOMINATOR (see the note at the top), so it may
        // be inexact without costing accuracy in the converged root or its sensitivity.
        dF2_dtau = dF2_dtau * diva_primal(H_c);

        // The closure residual at the CURRENT U_b, scaled by a velocity that cannot vanish
        // (U_bar is zero in stagnant and ice-free cells).  u_reg has units of velocity^2, so
        // its square root is the natural floor.
        T F = U_b + coeff*U_b*F2_c - U_bar;
        float F_scale = diva_primal(U_bar) + sqrtf(u_reg);
        float F_rel = fabsf(diva_primal(F))/F_scale;
        newton_used = it;
        if (post == 0) { closure_ok = true; break; }   // final consistent recompute done
        if (post < 0 && (F_rel <= closure_tol || (F_rel < near_tol && F_rel >= stag_gain*closure_prev))) {
            post = n_post;                             // criterion fires; take n_post more steps
        }
        closure_prev = F_rel;
        if (it >= newton_iters_max) break;

        // The Newton step.  F' from primals (see the note on denominators at the top); the
        // residual F itself is typed T, so the seeded derivative propagates exactly.
        DualFloat cs = get_diva_drag_coeff({diva_primal(U_b),1.0f},diva_primal(beta_grounded),diva_primal(m),u_reg,water_drag,diva_primal(u_c_c),sliding_law);
        DualFloat fs = cs * DualFloat{diva_primal(U_b),1.0f};
        float F2_p = diva_primal(F2_c);
        float Fp   = 1.0f + fs.d*F2_p + fs.v*fs.d*dF2_dtau;

        U_b = fmaxf(U_b - F/Fp, 0.0f);
        if (post > 0) post--;                          // count down the post-convergence steps
    }
    if (!closure_ok) cap_flags_out |= 1;

    eta_bar_out  = eta_avg;
    F1_out       = F1_c;
    F2_out       = F2_c;
    U_b_out      = U_b;
    beta_eff_out = coeff/(1.0f + coeff*F2_c);
    // Surface SPEED, u_s = u_b + tau_b*F1.  Emitted from here rather than assembled by the
    // caller so that a dual seeding of any input yields d(u_s)/d(that input) directly -- which
    // is what an objective built on surface-velocity observations needs, both for the adjoint
    // right-hand side (seeds 1-2) and for the parameter gradient's explicit term (seeds 3-5).
    u_s_out      = U_b + coeff*U_b*F1_c;
}



template <int HT, int WT>
__device__ void populate_diva_coeffs_dual(
    DualFloat (&eta_local)[HT][WT],
    DualFloat (&beta_eff_local)[HT][WT],
    int bi, int bj,
    int i, int j,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ du,
    const float* __restrict__ dv,
    const float* __restrict__ thk,
    const float* __restrict__ d_thk,
    const float* __restrict__ phi,
    const float* __restrict__ B,
    const float* __restrict__ beta,
    const float* __restrict__ u_c,
    const float* __restrict__ u_b,
    float m, float u_reg, float water_drag, float sliding_law,
    float n, float eps_reg, float eps_reg_shear, float dx,
    int n_sigma,
    const float* __restrict__ zeta_q,
    const float* __restrict__ w_q,
    int ny, int nx){

    // Fills the eta_bar and beta_eff tiles as DUALS: the exact d/d(velocity direction) of the
    // whole cell-local closure -- quadrature, per-level eta Newton and closure Newton all
    // differentiated by running them in dual arithmetic, nothing hand-derived.  The
    // frozen-coefficient adjoint (still selectable) drops this path;
    // tests/diva_dotproduct_test.py measures what that costs.
    float glen_exp = (1.0f - n)/(2.0f * n);

    DualFloat eps_mem_sq = get_membrane_eps_sq(u, v, du, dv, i, j, dx, ny, nx);

    DualFloat u_l = get_vfacet(u, du, i, j, ny, nx);
    DualFloat u_r = get_vfacet(u, du, i, j + 1, ny, nx);
    DualFloat v_t = get_hfacet(v, dv, i, j, ny, nx);
    DualFloat v_b = get_hfacet(v, dv, i + 1, j, ny, nx);

    DualFloat u_ctr = 0.5f*(u_l + u_r);
    DualFloat v_ctr = 0.5f*(v_t + v_b);
    DualFloat U_bar = sqrtf(0.5f*(u_l*u_l + u_r*u_r) + 0.5f*(v_t*v_t + v_b*v_b));

    // Thickness is seeded too, so a JVP direction with d_H != 0 carries the closure's
    // response to thickness -- not only the residual's explicit H terms.
    DualFloat H_c = get_cell(thk, d_thk, i, j, ny, nx);
    float B_c = get_cell(B, i, j, ny, nx);
    float grounded = get_cell(phi, i, j, ny, nx);
    float beta_grounded = get_cell(beta, i, j, ny, nx) * grounded;
    float u_c_c = get_cell(u_c, i, j, ny, nx);
    float U_b_warm = fminf(fmaxf(get_cell(u_b, i, j, ny, nx), 0.0f), U_bar.v);

    DualFloat eta_d, F1_d, F2_d, U_b_d, beta_eff_d, u_s_d;
    int cap_sink = 0;      // the diagnostic path (compute_diva_coeffs) owns cap reporting
    diva_coeffs_cell<DualFloat>(eps_mem_sq, U_bar,
            H_c, B_c, DualFloat{beta_grounded, 0.0f}, DualFloat{u_c_c, 0.0f},
            DualFloat{m, 0.0f}, u_reg, water_drag, sliding_law,
            glen_exp, eps_reg, eps_reg_shear, n_sigma, zeta_q, w_q,
            U_b_warm,
            eta_d, F1_d, F2_d, U_b_d, beta_eff_d, u_s_d, cap_sink);

    eta_local[bi][bj] = eta_d;
    beta_eff_local[bi][bj] = beta_eff_d;
}

extern "C" __global__
void compute_diva_coeffs(
    float* __restrict__ eta_bar,
    float* __restrict__ F2,
    float* __restrict__ u_b,
    float* __restrict__ beta_eff,
    float* __restrict__ F1,            // first shear moment; gives the SURFACE velocity
    float* __restrict__ u_s,           // surface speed, u_b + tau_b*F1
    float* __restrict__ cap_flags,     // per-cell: which local solve failed to reach tolerance
    const float* __restrict__ zeta_q,
    const float* __restrict__ w_q,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ H,
    const float* __restrict__ phi,
    const float* __restrict__ B,
    const float* __restrict__ beta,
    const float* __restrict__ u_c,
    float m, float u_reg, float water_drag, float sliding_law,
    float n, float eps_reg, float eps_reg_shear, float dx,
    int n_sigma,
    int ny, int nx,
    int stride, int halo
    )
{
    int j = blockIdx.x * stride + (threadIdx.x - halo);
    int i = blockIdx.y * stride + (threadIdx.y - halo);

    if (i < 0 || i >= ny || j < 0 || j >= nx) return;

    // Only interior (non-halo) threads write.  Each cell then has exactly one writer,
    // which matters because u_b is read in place as the closure's warm start: without
    // this, a halo thread from a neighbouring block could race the owning thread and
    // make the result depend on scheduling.
    bool is_active = (threadIdx.x >= halo && threadIdx.x < blockDim.x - halo) &&
                     (threadIdx.y >= halo && threadIdx.y < blockDim.y - halo);

    float glen_exp = (1.0f - n)/(2.0f * n);

    // Depth-averaged speed at the cell centre.
    float u_l = get_vfacet(u, i, j, ny, nx);
    float u_r = get_vfacet(u, i, j + 1, ny, nx);
    float v_t = get_hfacet(v, i, j, ny, nx);
    float v_b = get_hfacet(v, i + 1, j, ny, nx);

    float u_ctr = 0.5f*(u_l + u_r);
    float v_ctr = 0.5f*(v_t + v_b);
    float U_bar = sqrtf(0.5f*(u_l*u_l + u_r*u_r) + 0.5f*(v_t*v_t + v_b*v_b));

    float eps_mem_sq = get_membrane_eps_sq(u, v, i, j, dx, ny, nx);

    float H_c = get_cell(H, i, j, ny, nx);
    float B_c = get_cell(B, i, j, ny, nx);
    float grounded = get_cell(phi, i, j, ny, nx);
    // Grounding is folded in here, so beta_eff already carries it and the momentum
    // kernels must not apply the grounded factor a second time.
    float beta_grounded = get_cell(beta, i, j, ny, nx) * grounded;
    float u_c_c = get_cell(u_c, i, j, ny, nx);

    int idx = i * nx + j;

    // Warm start from the stored basal speed, bounded by the depth-averaged speed:
    // deformation can only add to sliding, so 0 <= U_b <= U_bar.
    float U_b_warm = fminf(fmaxf(u_b[idx], 0.0f), U_bar);

    float eta_avg, F1_c, F2_c, U_b, beta_eff_c, u_s_c;
    int caps = 0;
    diva_coeffs_cell<float>(eps_mem_sq, U_bar,
            H_c, B_c, beta_grounded, u_c_c,
            m, u_reg, water_drag, sliding_law,
            glen_exp, eps_reg, eps_reg_shear, n_sigma, zeta_q, w_q,
            U_b_warm,
            eta_avg, F1_c, F2_c, U_b, beta_eff_c, u_s_c, caps);

    if (is_active) {
        eta_bar[idx] = eta_avg;
        F2[idx] = F2_c;
        u_b[idx] = U_b;
        // Goldberg eq 41: the secant drag the 2D momentum solve sees, tau_b = beta_eff*U_bar.
        // Strictly non-negative, and finite at rest (no division by the speed).
        beta_eff[idx] = beta_eff_c;
        F1[idx] = F1_c;
        u_s[idx] = u_s_c;
        // Reduced and reported by the caller alongside |r_Ub|; should always be zero.
        cap_flags[idx] = (float)caps;
    }
}

/*=========================================================
  ====== THE CLOSURE RESIDUAL: DIVA's second criterion ====
  =========================================================*/
/*
  The closure residual at the CURRENT velocity, using the STORED coefficients:

      r_Ub = U_b + f(U_b)*F2 - |Ubar|

  eta_bar and beta_eff are frozen during a V-cycle, so this measures how far they have drifted
  out of consistency with the velocity while the smoother worked -- and doubles as a standing
  check that the closure solve converges at all.  Must be evaluated BEFORE the next
  compute_diva_coeffs call, or it is zero by construction and says nothing.

  Reported separately and scaled by |Ubar|, never folded into the combined momentum norm:
  r_Ub is a velocity residual and r_u a momentum one (notes/open_questions.md Q6).
*/
extern "C" __global__
void compute_diva_closure_residual(
    float* __restrict__ r_ub,
    float* __restrict__ U_bar_out,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ phi,
    const float* __restrict__ beta,
    const float* __restrict__ u_c,
    const float* __restrict__ u_b,
    const float* __restrict__ F2,
    float m, float u_reg, float water_drag, float sliding_law,
    int ny, int nx,
    int stride, int halo
    )
{
    int j = blockIdx.x * stride + (threadIdx.x - halo);
    int i = blockIdx.y * stride + (threadIdx.y - halo);

    if (i < 0 || i >= ny || j < 0 || j >= nx) return;

    bool is_active = (threadIdx.x >= halo && threadIdx.x < blockDim.x - halo) &&
                     (threadIdx.y >= halo && threadIdx.y < blockDim.y - halo);
    if (!is_active) return;

    float u_l = get_vfacet(u, i, j, ny, nx);
    float u_r = get_vfacet(u, i, j + 1, ny, nx);
    float v_t = get_hfacet(v, i, j, ny, nx);
    float v_b = get_hfacet(v, i + 1, j, ny, nx);
    float u_ctr = 0.5f*(u_l + u_r);
    float v_ctr = 0.5f*(v_t + v_b);
    float U_bar = sqrtf(0.5f*(u_l*u_l + u_r*u_r) + 0.5f*(v_t*v_t + v_b*v_b));

    int idx = i * nx + j;
    float grounded = get_cell(phi, i, j, ny, nx);
    float beta_g = get_cell(beta, i, j, ny, nx) * grounded;
    float u_c_c = get_cell(u_c, i, j, ny, nx);
    float U_b = u_b[idx];
    float F2_c = F2[idx];

    // c(U_b) from the same helper the closure itself uses, so this cannot drift out of step
    // with the sliding law.
    float c = get_diva_drag_coeff({U_b,0.0f},beta_g,m,u_reg,water_drag,u_c_c,sliding_law).v;

    r_ub[idx]      = U_b + c*U_b*F2_c - U_bar;
    U_bar_out[idx] = U_bar;
}

extern "C" __global__
void compute_diva_derivs(
    float* __restrict__ deta_deps,
    float* __restrict__ deta_dU,
    float* __restrict__ dbe_deps,
    float* __restrict__ dbe_dU,
    float* __restrict__ deta_dbeta,
    float* __restrict__ dbe_dbeta,
    float* __restrict__ deta_duc,
    float* __restrict__ dbe_duc,
    float* __restrict__ deta_dm,
    float* __restrict__ dbe_dm,
    float* __restrict__ dus_deps,      // the surface-speed sensitivities, for an objective
    float* __restrict__ dus_dU,        // built on surface-velocity observations
    float* __restrict__ dus_dbeta,
    float* __restrict__ dus_duc,
    float* __restrict__ dus_dm,
    float* __restrict__ deta_dH,       // the closure's THICKNESS derivatives, which
    float* __restrict__ dbe_dH,        // complete the coupled (u,v,H) linearization
    float* __restrict__ dus_dH,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ H,
    const float* __restrict__ phi,
    const float* __restrict__ B,
    const float* __restrict__ beta,
    const float* __restrict__ u_c,
    const float* __restrict__ u_b,
    float m, float u_reg, float water_drag, float sliding_law,
    float n, float eps_reg, float eps_reg_shear, float dx,
    int n_sigma,
    const float* __restrict__ zeta_q,
    const float* __restrict__ w_q,
    int ny, int nx,
    int stride, int halo
    )
{
    // Total derivatives of the cell-local closure, one dual seeding per input.  These are
    // what the adjoint needs and the forward does not: with them the transpose can be
    // applied without re-running the quadrature, and without assuming symmetry.
    //
    //   seeds 1-2 (eps_mem^2, Ubar) -> the state transpose
    //   seeds 3-5 (beta, u_c, m)    -> the parameter gradients
    //
    // The two groups differ only in which input is seeded; there is one closure, and the
    // quadrature/fixed-point/Newton are differentiated by running them in dual arithmetic
    // rather than by any hand-derived expression.
    int j = blockIdx.x * stride + (threadIdx.x - halo);
    int i = blockIdx.y * stride + (threadIdx.y - halo);

    if (i < 0 || i >= ny || j < 0 || j >= nx) return;

    bool is_active = (threadIdx.x >= halo && threadIdx.x < blockDim.x - halo) &&
                     (threadIdx.y >= halo && threadIdx.y < blockDim.y - halo);

    float glen_exp = (1.0f - n)/(2.0f * n);

    float u_l = get_vfacet(u, i, j, ny, nx);
    float u_r = get_vfacet(u, i, j + 1, ny, nx);
    float v_t = get_hfacet(v, i, j, ny, nx);
    float v_b = get_hfacet(v, i + 1, j, ny, nx);
    float u_ctr = 0.5f*(u_l + u_r);
    float v_ctr = 0.5f*(v_t + v_b);
    float U_bar_v = sqrtf(0.5f*(u_l*u_l + u_r*u_r) + 0.5f*(v_t*v_t + v_b*v_b));

    float eps_mem_v = get_membrane_eps_sq(u, v, i, j, dx, ny, nx);

    float H_c = get_cell(H, i, j, ny, nx);
    float B_c = get_cell(B, i, j, ny, nx);
    float grounded = get_cell(phi, i, j, ny, nx);
    float beta_grounded = get_cell(beta, i, j, ny, nx) * grounded;
    float u_c_c = get_cell(u_c, i, j, ny, nx);
    int idx = i * nx + j;
    float U_b_warm = fminf(fmaxf(u_b[idx], 0.0f), U_bar_v);

    DualFloat eta_d, F1_d, F2_d, U_b_d, be_d, us_d;
    int cap_sink = 0;      // derivative path; compute_diva_coeffs owns cap reporting

    // Seed 1: d/d(eps_mem^2)
    diva_coeffs_cell<DualFloat>({eps_mem_v,1.0f}, {U_bar_v,0.0f},
            {H_c,0.0f}, B_c, {beta_grounded,0.0f}, {u_c_c,0.0f},
            {m,0.0f}, u_reg, water_drag, sliding_law,
            glen_exp, eps_reg, eps_reg_shear, n_sigma, zeta_q, w_q, U_b_warm,
            eta_d, F1_d, F2_d, U_b_d, be_d, us_d, cap_sink);
    float d_eta_deps = eta_d.d;
    float d_be_deps  = be_d.d;
    float d_us_deps  = us_d.d;

    // Seed 2: d/d(Ubar)
    diva_coeffs_cell<DualFloat>({eps_mem_v,0.0f}, {U_bar_v,1.0f},
            {H_c,0.0f}, B_c, {beta_grounded,0.0f}, {u_c_c,0.0f},
            {m,0.0f}, u_reg, water_drag, sliding_law,
            glen_exp, eps_reg, eps_reg_shear, n_sigma, zeta_q, w_q, U_b_warm,
            eta_d, F1_d, F2_d, U_b_d, be_d, us_d, cap_sink);
    float d_eta_dU = eta_d.d;
    float d_be_dU  = be_d.d;
    float d_us_dU  = us_d.d;

    // Seed 3: d/d(beta).  The perturbation is `grounded` rather than 1 because the
    // closure is given beta*grounded, so this yields the derivative with respect to the
    // raw beta the inversion actually controls.  Note beta moves eta_bar as well as
    // beta_eff (beta -> c -> tau_b -> shear term), which is the path the parameter
    // gradient was missing.
    diva_coeffs_cell<DualFloat>({eps_mem_v,0.0f}, {U_bar_v,0.0f},
            {H_c,0.0f}, B_c, {beta_grounded,grounded}, {u_c_c,0.0f},
            {m,0.0f}, u_reg, water_drag, sliding_law,
            glen_exp, eps_reg, eps_reg_shear, n_sigma, zeta_q, w_q, U_b_warm,
            eta_d, F1_d, F2_d, U_b_d, be_d, us_d, cap_sink);
    float d_eta_dbeta = eta_d.d;
    float d_be_dbeta  = be_d.d;
    float d_us_dbeta  = us_d.d;

    // Seed 4: d/d(u_c), the regularized-Coulomb threshold speed.  Identically zero under
    // Weertman, where the u_c branch is not taken -- the same way SSA's d_u_c is.
    diva_coeffs_cell<DualFloat>({eps_mem_v,0.0f}, {U_bar_v,0.0f},
            {H_c,0.0f}, B_c, {beta_grounded,0.0f}, {u_c_c,1.0f},
            {m,0.0f}, u_reg, water_drag, sliding_law,
            glen_exp, eps_reg, eps_reg_shear, n_sigma, zeta_q, w_q, U_b_warm,
            eta_d, F1_d, F2_d, U_b_d, be_d, us_d, cap_sink);
    float d_eta_duc = eta_d.d;
    float d_be_duc  = be_d.d;
    float d_us_duc  = us_d.d;

    // Seed 5: d/d(m), the Weertman exponent.  m is a single global scalar, so these are
    // the per-cell contributions that compute_gradient_param_sum_diva reduces.  Getting
    // this needed __powf(dual, dual) -- the exponent, not just the base, is now dual.
    // Identically zero under regularized Coulomb.
    diva_coeffs_cell<DualFloat>({eps_mem_v,0.0f}, {U_bar_v,0.0f},
            {H_c,0.0f}, B_c, {beta_grounded,0.0f}, {u_c_c,0.0f},
            {m,1.0f}, u_reg, water_drag, sliding_law,
            glen_exp, eps_reg, eps_reg_shear, n_sigma, zeta_q, w_q, U_b_warm,
            eta_d, F1_d, F2_d, U_b_d, be_d, us_d, cap_sink);
    float d_eta_dm = eta_d.d;
    float d_be_dm  = be_d.d;
    float d_us_dm  = us_d.d;

    // Seed 6: d/d(H).  Thickness is a STATE variable, not a parameter, so this one
    // completes the coupled (u,v,H) Jacobian rather than adding a control.  H enters the
    // closure only through the shear moments I1 = H*int(zeta/eta) and I2 = H*int(zeta^2/eta),
    // but that is enough to move everything downstream of them: thicker ice means a larger
    // I2, hence a smaller basal speed for the same depth-averaged speed, hence a different
    // traction, vertical shear, viscosity profile and therefore eta_bar and beta_eff too.
    // Without this the DIVA adjoint is exact only at fixed thickness.
    diva_coeffs_cell<DualFloat>({eps_mem_v,0.0f}, {U_bar_v,0.0f},
            {H_c,1.0f}, B_c, {beta_grounded,0.0f}, {u_c_c,0.0f},
            {m,0.0f}, u_reg, water_drag, sliding_law,
            glen_exp, eps_reg, eps_reg_shear, n_sigma, zeta_q, w_q, U_b_warm,
            eta_d, F1_d, F2_d, U_b_d, be_d, us_d, cap_sink);
    float d_eta_dH = eta_d.d;
    float d_be_dH  = be_d.d;
    float d_us_dH  = us_d.d;

    if (is_active) {
        deta_deps[idx] = d_eta_deps;
        deta_dU[idx]   = d_eta_dU;
        dbe_deps[idx]  = d_be_deps;
        dbe_dU[idx]    = d_be_dU;
        deta_dbeta[idx] = d_eta_dbeta;
        dbe_dbeta[idx]  = d_be_dbeta;
        deta_duc[idx]   = d_eta_duc;
        dbe_duc[idx]    = d_be_duc;
        deta_dm[idx]    = d_eta_dm;
        dbe_dm[idx]     = d_be_dm;
        dus_deps[idx]   = d_us_deps;
        dus_dU[idx]     = d_us_dU;
        dus_dbeta[idx]  = d_us_dbeta;
        dus_duc[idx]    = d_us_duc;
        dus_dm[idx]     = d_us_dm;
        deta_dH[idx]    = d_eta_dH;
        dbe_dH[idx]     = d_be_dH;
        dus_dH[idx]     = d_us_dH;
    }
}

/*=========================================================
  ==== DIVA VJP: coefficient adjoints -> velocity ==========
  =========================================================*/
/*
  Second half of the DIVA transpose.  vjp_body pushes lambda_row * d(r_row)/d(coeff)
  onto the owning cell, giving W_eta and W_be.  This kernel converts those per-cell
  coefficient adjoints into velocity sensitivities:

      (J^T lambda)_j += sum_c [ A_c * d(eps_mem^2)_c/d(u_j) + B_c * d(Ubar)_c/d(u_j) ]

      A_c = W_eta_c * d(eta_bar)/d(eps_mem^2) + W_be_c * d(beta_eff)/d(eps_mem^2)
      B_c = W_eta_c * d(eta_bar)/d(Ubar)      + W_be_c * d(beta_eff)/d(Ubar)

  One thread per cell, scattering to that cell's own stencil, so the reach is +/-1 --
  the composite row->facet dependence is +/-2 but neither half exceeds +/-1.  No
  symmetry is assumed anywhere: this is the honest transpose of the coefficient paths,
  which is what the frozen adjoint omitted.
*/
/*
  The cell->facet half of a coefficient transpose.  One thread owns a cell, is handed that
  cell's cotangents on the two closure inputs,

      A = d(objective)/d(eps_mem^2)_c ,   B = d(objective)/d(Ubar)_c

  and scatters them onto the facets those inputs depend on, through the analytic partials of
  eps_mem^2 and Ubar.  Reach is +/-1 in each direction.

  Shared by two callers with different A and B: the adjoint residual, where they come from
  W_eta/W_be times the coefficient derivatives, and a surface-velocity objective, where they
  come from the misfit times d(u_s)/d(.).  The scatter itself does not care which.
*/
/*
  The partials of a cell's two closure INPUTS with respect to its OWN four velocity
  facets, in the block order (u_l, u_r, v_t, v_b).  Same expressions as
  diva_scatter_cell_to_facets below -- that function scatters A*d(q1) + B*d(q2) to every
  facet the inputs touch, whereas the Vanka block needs only the four it owns, and needs
  them as coefficients rather than as a scatter.

  Kept adjacent to the scatter deliberately: the two must agree term for term, and
  membrane_eps_sq is the definition both are differentiating.
*/
__device__ __forceinline__
void diva_cell_own_partials(
    int i, int j,
    const float* __restrict__ u,
    const float* __restrict__ v,
    float dx, int ny, int nx,
    float* __restrict__ dq1,        // d(eps_mem^2)/d(u_l,u_r,v_t,v_b)
    float* __restrict__ dq2)        // d(Ubar)/d(u_l,u_r,v_t,v_b)
{
    float dx_inv = 1.0f/dx;
    float h = 0.5f*dx_inv;

    float u_l = get_vfacet(u, i, j, ny, nx);
    float u_r = get_vfacet(u, i, j + 1, ny, nx);
    float v_t = get_hfacet(v, i, j, ny, nx);
    float v_b = get_hfacet(v, i + 1, j, ny, nx);

    float dudx = (u_r - u_l)*dx_inv;
    float dvdy = (v_t - v_b)*dx_inv;

    float tl_mask = i > 0 && j > 0;
    float tr_mask = i > 0 && j < (nx - 1);
    float bl_mask = i < (ny - 1) && j > 0;
    float br_mask = i < (ny - 1) && j < (nx - 1);

    float u_tl = get_vfacet(u, i - 1, j, ny, nx);
    float v_lt = get_hfacet(v, i, j - 1, ny, nx);
    float u_tr = get_vfacet(u, i - 1, j + 1, ny, nx);
    float v_rt = get_hfacet(v, i, j + 1, ny, nx);
    float u_bl = get_vfacet(u, i + 1, j, ny, nx);
    float v_lb = get_hfacet(v, i + 1, j - 1, ny, nx);
    float u_br = get_vfacet(u, i + 1, j + 1, ny, nx);
    float v_rb = get_hfacet(v, i + 1, j + 1, ny, nx);

    float e_tl = 0.5f*((u_tl - u_l)*dx_inv + (v_t - v_lt)*dx_inv)*tl_mask;
    float e_tr = 0.5f*((u_tr - u_r)*dx_inv + (v_rt - v_t)*dx_inv)*tr_mask;
    float e_bl = 0.5f*((u_l - u_bl)*dx_inv + (v_b - v_lb)*dx_inv)*bl_mask;
    float e_br = 0.5f*((u_r - u_br)*dx_inv + (v_rb - v_b)*dx_inv)*br_mask;

    float P = 2.0f*dudx + dvdy;
    float Q = 2.0f*dvdy + dudx;
    float R_tl = 0.5f*e_tl*tl_mask;
    float R_tr = 0.5f*e_tr*tr_mask;
    float R_bl = 0.5f*e_bl*bl_mask;
    float R_br = 0.5f*e_br*br_mask;

    dq1[0] = -P*dx_inv - R_tl*h + R_bl*h;
    dq1[1] =  P*dx_inv - R_tr*h + R_br*h;
    dq1[2] =  Q*dx_inv + R_tl*h - R_tr*h;
    dq1[3] = -Q*dx_inv + R_bl*h - R_br*h;

    float u_ctr = 0.5f*(u_l + u_r);
    float v_ctr = 0.5f*(v_t + v_b);
    float U_bar = sqrtf(0.5f*(u_l*u_l + u_r*u_r) + 0.5f*(v_t*v_t + v_b*v_b));
    float inv_U = U_bar > 1e-6f ? 1.0f/U_bar : 0.0f;
    dq2[0] = 0.5f*u_ctr*inv_U;
    dq2[1] = dq2[0];
    dq2[2] = 0.5f*v_ctr*inv_U;
    dq2[3] = dq2[2];
}


__device__ __forceinline__
void diva_scatter_cell_to_facets(
    float A, float B,
    int i, int j,
    const float* __restrict__ u,
    const float* __restrict__ v,
    float* __restrict__ r_u,
    float* __restrict__ r_v,
    float dx,
    int ny, int nx)
{
    float dx_inv = 1.0f/dx;
    float h = 0.5f*dx_inv;

    // Rebuild the same strain rates get_membrane_eps_sq forms, so the partials below
    // match it term for term.
    float u_l = get_vfacet(u, i, j, ny, nx);
    float u_r = get_vfacet(u, i, j + 1, ny, nx);
    float v_t = get_hfacet(v, i, j, ny, nx);
    float v_b = get_hfacet(v, i + 1, j, ny, nx);

    float dudx = (u_r - u_l)*dx_inv;
    float dvdy = (v_t - v_b)*dx_inv;

    float tl_mask = i > 0 && j > 0;
    float tr_mask = i > 0 && j < (nx - 1);
    float bl_mask = i < (ny - 1) && j > 0;
    float br_mask = i < (ny - 1) && j < (nx - 1);

    float u_tl = get_vfacet(u, i - 1, j, ny, nx);
    float v_lt = get_hfacet(v, i, j - 1, ny, nx);
    float u_tr = get_vfacet(u, i - 1, j + 1, ny, nx);
    float v_rt = get_hfacet(v, i, j + 1, ny, nx);
    float u_bl = get_vfacet(u, i + 1, j, ny, nx);
    float v_lb = get_hfacet(v, i + 1, j - 1, ny, nx);
    float u_br = get_vfacet(u, i + 1, j + 1, ny, nx);
    float v_rb = get_hfacet(v, i + 1, j + 1, ny, nx);

    float e_tl = 0.5f*((u_tl - u_l)*dx_inv + (v_t - v_lt)*dx_inv)*tl_mask;
    float e_tr = 0.5f*((u_tr - u_r)*dx_inv + (v_rt - v_t)*dx_inv)*tr_mask;
    float e_bl = 0.5f*((u_l - u_bl)*dx_inv + (v_b - v_lb)*dx_inv)*bl_mask;
    float e_br = 0.5f*((u_r - u_br)*dx_inv + (v_rb - v_b)*dx_inv)*br_mask;

    // d(eps_mem^2)/d(.) where eps_mem^2 = dudx^2 + dvdy^2 + dudx*dvdy + eps_xy2_bar
    float P = 2.0f*dudx + dvdy;          // d/d(dudx)
    float Q = 2.0f*dvdy + dudx;          // d/d(dvdy)
    float R_tl = 0.5f*e_tl*tl_mask;      // d/d(eps_xy_tl)
    float R_tr = 0.5f*e_tr*tr_mask;
    float R_bl = 0.5f*e_bl*bl_mask;
    float R_br = 0.5f*e_br*br_mask;

    // d(Ubar)/d(.) : Ubar = |0.5(u_l+u_r), 0.5(v_t+v_b)|
    float u_ctr = 0.5f*(u_l + u_r);
    float v_ctr = 0.5f*(v_t + v_b);
    float U_bar = sqrtf(0.5f*(u_l*u_l + u_r*u_r) + 0.5f*(v_t*v_t + v_b*v_b));
    float inv_U = U_bar > 1e-6f ? 1.0f/U_bar : 0.0f;
    float dU_du = 0.5f*u_ctr*inv_U;      // for both u_l and u_r
    float dU_dv = 0.5f*v_ctr*inv_U;      // for both v_t and v_b

    // Scatter.  u facets are (ny, nx+1); v facets are (ny+1, nx).
    // Skip the Dirichlet facets.  The main VJP kernel replaces those rows with an
    // identity (lambda = 0), so anything added here could never be reduced by the
    // smoother and would sit in the residual forever as a convergence floor.
    #define DIVA_ADD_U(I,J,VAL) if ((I) >= 0 && (I) < ny && (J) > 0 && (J) < nx) \
        atomicAdd(&r_u[(I)*(nx + 1) + (J)], (VAL));
    #define DIVA_ADD_V(I,J,VAL) if ((I) > 0 && (I) < ny && (J) >= 0 && (J) < nx) \
        atomicAdd(&r_v[(I)*nx + (J)], (VAL));

    DIVA_ADD_U(i,   j,     A*(-P*dx_inv - R_tl*h + R_bl*h) + B*dU_du)
    DIVA_ADD_U(i,   j + 1, A*( P*dx_inv - R_tr*h + R_br*h) + B*dU_du)
    DIVA_ADD_V(i,   j,     A*( Q*dx_inv + R_tl*h - R_tr*h) + B*dU_dv)
    DIVA_ADD_V(i + 1, j,   A*(-Q*dx_inv + R_bl*h - R_br*h) + B*dU_dv)

    DIVA_ADD_U(i - 1, j,     A*( R_tl*h))
    DIVA_ADD_U(i - 1, j + 1, A*( R_tr*h))
    DIVA_ADD_U(i + 1, j,     A*(-R_bl*h))
    DIVA_ADD_U(i + 1, j + 1, A*(-R_br*h))

    DIVA_ADD_V(i,     j - 1, A*(-R_tl*h))
    DIVA_ADD_V(i,     j + 1, A*( R_tr*h))
    DIVA_ADD_V(i + 1, j - 1, A*(-R_bl*h))
    DIVA_ADD_V(i + 1, j + 1, A*( R_br*h))

    #undef DIVA_ADD_U
    #undef DIVA_ADD_V
}

extern "C" __global__
void compute_diva_vjp_coeffs(
    float* __restrict__ r_u,
    float* __restrict__ r_v,
    float* __restrict__ r_H,           // thickness cotangent, ACCUMULATED into
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ W_eta,
    const float* __restrict__ W_be,
    const float* __restrict__ deta_deps,
    const float* __restrict__ deta_dU,
    const float* __restrict__ dbe_deps,
    const float* __restrict__ dbe_dU,
    const float* __restrict__ deta_dH,
    const float* __restrict__ dbe_dH,
    float dx,
    int ny, int nx,
    int stride, int halo
    )
{
    int j = blockIdx.x * stride + (threadIdx.x - halo);
    int i = blockIdx.y * stride + (threadIdx.y - halo);

    if (i < 0 || i >= ny || j < 0 || j >= nx) return;

    bool is_active = (threadIdx.x >= halo && threadIdx.x < blockDim.x - halo) &&
                     (threadIdx.y >= halo && threadIdx.y < blockDim.y - halo);
    if (!is_active) return;

    int idx = i * nx + j;
    float we = W_eta[idx];
    float wb = W_be[idx];
    float A = we*deta_deps[idx] + wb*dbe_deps[idx];
    float B = we*deta_dU[idx]   + wb*dbe_dU[idx];

    // The thickness half of the same contraction.  Unlike the velocity inputs, which are
    // assembled from surrounding facets and so need a scatter, H_c IS the cell's own
    // value -- so the closure-through-thickness path lands on one cell with no stencil
    // and no atomics.  Every cell is visited exactly once, hence the plain +=.
    //
    //     K_c = W_eta_c * d(eta_bar)/dH + W_be_c * d(beta_eff)/dH
    //
    // This is added to the DIRECT thickness cotangent that vjp_body already accumulated
    // (from H*eta_bar, the driving stress, the fluxes and so on), which is why r_H is
    // accumulated into rather than written.
    if (r_H != nullptr) {
        float K = we*deta_dH[idx] + wb*dbe_dH[idx];
        if (K != 0.0f) r_H[idx] += K;
    }

    if (A == 0.0f && B == 0.0f) return;

    diva_scatter_cell_to_facets(A, B, i, j, u, v, r_u, r_v, dx, ny, nx);
}

/*=========================================================
  ==== d(surface velocity)/d(state), transposed ===========
  =========================================================*/
/*
  For an objective built on SURFACE velocity rather than the depth average.  Given the per-cell
  cotangent cot_c = d(objective)/d(u_s)_c, produce d(objective)/d(u,v) on the facets:

      d(obj)/d(u_j) = sum_c cot_c * [ dus_deps_c * d(eps_mem^2)_c/d(u_j)
                                    + dus_dU_c   * d(Ubar)_c/d(u_j) ]

  which is the same scatter as the adjoint residual's coefficient transpose with A and B
  redefined.  The caller negates this to form the adjoint right-hand side, f = -d(obj)/d(x).

  Under SSA there is no vertical shear, u_s == |ubar|, and this whole path is unnecessary --
  which is what makes the SSA limit a usable control on it.
*/
extern "C" __global__
void compute_diva_us_vjp(
    float* __restrict__ out_u,
    float* __restrict__ out_v,
    float* __restrict__ out_H,           // u_s depends on H through the shear moments
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ cot,       // d(objective)/d(u_s), per cell
    const float* __restrict__ dus_deps,
    const float* __restrict__ dus_dU,
    const float* __restrict__ dus_dH,
    float dx,
    int ny, int nx,
    int stride, int halo
    )
{
    int j = blockIdx.x * stride + (threadIdx.x - halo);
    int i = blockIdx.y * stride + (threadIdx.y - halo);

    if (i < 0 || i >= ny || j < 0 || j >= nx) return;

    bool is_active = (threadIdx.x >= halo && threadIdx.x < blockDim.x - halo) &&
                     (threadIdx.y >= halo && threadIdx.y < blockDim.y - halo);
    if (!is_active) return;

    int idx = i * nx + j;
    float c = cot[idx];
    if (c == 0.0f) return;

    // Surface speed depends on thickness the same way the coefficients do -- through
    // I1 and I2 and the basal-speed root -- and, like them, it lands on this cell alone.
    if (out_H != nullptr) out_H[idx] += c*dus_dH[idx];

    diva_scatter_cell_to_facets(c*dus_deps[idx], c*dus_dU[idx],
                                i, j, u, v, out_u, out_v, dx, ny, nx);
}


/*=========================================================
  === DIVA parameter gradients: cell-local W products ======
  =========================================================*/
/*
  W_eta and W_be, filled by vjp_body, already ARE lambda^T d(r)/d(coefficient) summed
  over every row that touches the cell.  So a parameter gradient is just the chain rule
  applied per cell, with no facet loop at all:

      dJ/d(p)_c = W_eta_c * d(eta_bar_c)/d(p_c) + W_be_c * d(beta_eff_c)/d(p_c)

  Nothing in this expression knows *which* parameter p is -- the identity of p lives
  entirely in which pair of derivative fields the caller passes.  So one kernel serves
  beta and u_c, and the reducing variant below serves the global m.  Compare the SSA
  path, which needs a separate ~90-line facet-walking kernel per parameter
  (compute_gradient_beta / _u_c / _m in grad.cu) because there beta, u_c and m enter the
  momentum stencils directly rather than through a state-dependent coefficient.

  This replaced an earlier facet-walking compute_gradient_beta_diva that carried only the
  beta_eff path.  Every parameter also moves eta_bar, through p -> c -> tau_b -> the
  shear term in the effective strain rate; for beta that omission was the whole error.
*/
template <bool REDUCE>
__device__ __forceinline__
void diva_param_gradient_body(
    float* __restrict__ grad,
    const float* __restrict__ W_eta,
    const float* __restrict__ W_be,
    const float* __restrict__ deta_dp,
    const float* __restrict__ dbe_dp,
    int ny, int nx,
    int stride, int halo
    )
{
    int j = blockIdx.x * stride + (threadIdx.x - halo);
    int i = blockIdx.y * stride + (threadIdx.y - halo);

    if (i < 0 || i >= ny || j < 0 || j >= nx) return;

    bool is_active = (threadIdx.x >= halo && threadIdx.x < blockDim.x - halo) &&
                     (threadIdx.y >= halo && threadIdx.y < blockDim.y - halo);
    if (!is_active) return;

    int idx = i * nx + j;
    float g = W_eta[idx]*deta_dp[idx] + W_be[idx]*dbe_dp[idx];

    if (REDUCE) {
        // A single global scalar (the Weertman m), so every cell's contribution is summed
        // into one slot.  Same convention as SSA's compute_gradient_m; note the atomics
        // make the summation order nondeterministic, so this gradient is reproducible
        // only to float32 round-off, unlike the per-cell variant.
        atomicAdd(&grad[0], g);
    } else {
        grad[idx] = g;
    }
}

// Per-cell parameter: used for both beta and u_c, which differ only in the derivative
// fields passed in.
extern "C" __global__
void compute_gradient_param_diva(
    float* __restrict__ grad,
    const float* __restrict__ W_eta,
    const float* __restrict__ W_be,
    const float* __restrict__ deta_dp,
    const float* __restrict__ dbe_dp,
    int ny, int nx,
    int stride, int halo
    )
{
    diva_param_gradient_body<false>(grad, W_eta, W_be, deta_dp, dbe_dp, ny, nx, stride, halo);
}

// Global scalar parameter: used for m.  grad must be zeroed by the caller.
extern "C" __global__
void compute_gradient_param_sum_diva(
    float* __restrict__ grad,
    const float* __restrict__ W_eta,
    const float* __restrict__ W_be,
    const float* __restrict__ deta_dp,
    const float* __restrict__ dbe_dp,
    int ny, int nx,
    int stride, int halo
    )
{
    diva_param_gradient_body<true>(grad, W_eta, W_be, deta_dp, dbe_dp, ny, nx, stride, halo);
}
