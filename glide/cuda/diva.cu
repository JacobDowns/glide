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

     R(U_b) = U_b + f(U_b)*F2 - U_bar = 0.

  For a linear law this inverts in closed form; in general it is a per-cell root
  find (Goldberg's eqs 38-39, "solved at a location along the base independently of
  other locations").  It is well behaved: physical laws are monotone (f' >= 0) and
  F2 > 0, so R' = 1 + f'*F2 > 0 and the root is unique.

  Everything here is cell-local -- no spatial coupling -- so no multigrid transfer
  is involved.  Nothing in this file is reached unless stress_balance = 1.
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

template <typename T>
__device__ void diva_coeffs_cell(
    T eps_mem_sq, T U_bar,                       // the two velocity-dependent inputs
    float H_c, float B_c, T beta_grounded, T u_c_c,
    T m, float u_reg, float water_drag, float sliding_law,
    float glen_exp, float eps_reg, int n_sigma,
    float U_b_warm,
    T& eta_bar_out, T& F2_out, T& U_b_out, T& beta_eff_out)
{
    // Per-cell DIVA closure, shared by compute_diva_coeffs (T = float) and the JVP
    // (T = DualFloat, seeded with a velocity perturbation direction).
    //
    // Note the Newton denominator dR stays a plain float even when T is dual.  That is
    // deliberate and costs nothing: for U <- U - R(U)/c with any constant c, the
    // converged derivative satisfies dU = -R_dir/R', independent of c.  So the step
    // size does not affect the sensitivity, and the second differentiation (f' w.r.t.
    // U_b, which is not the direction being seeded) can be taken from the primals.
    const int coupling_iters = 3;
    const int eta_iters = 3;
    const int newton_iters = 4;

    float w = 1.0f/(float)n_sigma;

    T U_b; diva_from_float(U_b, U_b_warm);
    T coeff = get_diva_c_of_U<T>(U_b, beta_grounded, m, u_reg, water_drag, u_c_c, sliding_law);
    T tau_b = coeff * U_b;

    T eta_avg = T();
    T F2_c = T();

    for (int c = 0; c < coupling_iters; ++c) {
        eta_avg = T();
        F2_c = T();

        for (int k = 0; k < n_sigma; ++k) {
            float zeta = ((float)k + 0.5f)*w;

            T eta_k = 0.5f*B_c*__powf(eps_mem_sq + eps_reg, glen_exp);
            for (int e = 0; e < eta_iters; ++e) {
                T eps_shear = tau_b*zeta/(2.0f*eta_k);
                eta_k = 0.5f*B_c*__powf(eps_mem_sq + eps_shear*eps_shear + eps_reg, glen_exp);
            }

            eta_avg = eta_avg + w*eta_k;
            F2_c    = F2_c + w*zeta*zeta/eta_k;
        }
        F2_c = F2_c * H_c;

        for (int it = 0; it < newton_iters; ++it) {
            coeff = get_diva_c_of_U<T>(U_b, beta_grounded, m, u_reg, water_drag, u_c_c, sliding_law);
            T f = coeff * U_b;
            T R = U_b + f*F2_c - U_bar;
            // f'(U_b) from the primals: a separate differentiation from the seeded one.
            DualFloat cs = get_diva_drag_coeff({diva_primal(U_b),1.0f},diva_primal(beta_grounded),diva_primal(m),u_reg,water_drag,diva_primal(u_c_c),sliding_law);
            DualFloat fs = cs * DualFloat{diva_primal(U_b),1.0f};
            float dR = 1.0f + fs.d*diva_primal(F2_c);
            U_b = fmaxf(U_b - R/dR, 0.0f);
        }

        coeff = get_diva_c_of_U<T>(U_b, beta_grounded, m, u_reg, water_drag, u_c_c, sliding_law);
        tau_b = coeff * U_b;
    }

    eta_bar_out  = eta_avg;
    F2_out       = F2_c;
    U_b_out      = U_b;
    beta_eff_out = coeff/(1.0f + coeff*F2_c);
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
    const float* __restrict__ phi,
    const float* __restrict__ B,
    const float* __restrict__ beta,
    const float* __restrict__ u_c,
    const float* __restrict__ u_b,
    float m, float u_reg, float water_drag, float sliding_law,
    float n, float eps_reg, float dx, int n_sigma,
    int ny, int nx){

    // Fills the eta_bar and beta_eff tiles as DUALS, i.e. with the exact
    // d/d(velocity direction) of the whole cell-local closure -- the vertical
    // quadrature, the eta_k fixed point and the U_b Newton all differentiated by
    // running them in dual arithmetic.  This is what the frozen adjoint omits.
    float glen_exp = (1.0f - n)/(2.0f * n);

    DualFloat eps_mem_sq = get_membrane_eps_sq(u, v, du, dv, i, j, dx, ny, nx);

    DualFloat u_l = get_vfacet(u, du, i, j, ny, nx);
    DualFloat u_r = get_vfacet(u, du, i, j + 1, ny, nx);
    DualFloat v_t = get_hfacet(v, dv, i, j, ny, nx);
    DualFloat v_b = get_hfacet(v, dv, i + 1, j, ny, nx);

    DualFloat u_ctr = 0.5f*(u_l + u_r);
    DualFloat v_ctr = 0.5f*(v_t + v_b);
    DualFloat U_bar = sqrtf(u_ctr*u_ctr + v_ctr*v_ctr);

    float H_c = get_cell(thk, i, j, ny, nx);
    float B_c = get_cell(B, i, j, ny, nx);
    float grounded = get_cell(phi, i, j, ny, nx);
    float beta_grounded = get_cell(beta, i, j, ny, nx) * grounded;
    float u_c_c = get_cell(u_c, i, j, ny, nx);
    float U_b_warm = fminf(fmaxf(get_cell(u_b, i, j, ny, nx), 0.0f), U_bar.v);

    DualFloat eta_d, F2_d, U_b_d, beta_eff_d;
    diva_coeffs_cell<DualFloat>(eps_mem_sq, U_bar,
            H_c, B_c, DualFloat{beta_grounded, 0.0f}, DualFloat{u_c_c, 0.0f},
            DualFloat{m, 0.0f}, u_reg, water_drag, sliding_law,
            glen_exp, eps_reg, n_sigma,
            U_b_warm,
            eta_d, F2_d, U_b_d, beta_eff_d);

    eta_local[bi][bj] = eta_d;
    beta_eff_local[bi][bj] = beta_eff_d;
}

extern "C" __global__
void compute_diva_coeffs(
    float* __restrict__ eta_bar,
    float* __restrict__ F2,
    float* __restrict__ u_b,
    float* __restrict__ beta_eff,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ H,
    const float* __restrict__ phi,
    const float* __restrict__ B,
    const float* __restrict__ beta,
    const float* __restrict__ u_c,
    float m, float u_reg, float water_drag, float sliding_law,
    float n, float eps_reg, float dx,
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
    float U_bar = sqrtf(u_ctr*u_ctr + v_ctr*v_ctr);

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

    float eta_avg, F2_c, U_b, beta_eff_c;
    diva_coeffs_cell<float>(eps_mem_sq, U_bar,
            H_c, B_c, beta_grounded, u_c_c,
            m, u_reg, water_drag, sliding_law,
            glen_exp, eps_reg, n_sigma,
            U_b_warm,
            eta_avg, F2_c, U_b, beta_eff_c);

    if (is_active) {
        eta_bar[idx] = eta_avg;
        F2[idx] = F2_c;
        u_b[idx] = U_b;
        // Goldberg eq 41: the secant drag the 2D momentum solve sees, tau_b = beta_eff*U_bar.
        // Strictly non-negative, and finite at rest (no division by the speed).
        beta_eff[idx] = beta_eff_c;
    }
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
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ H,
    const float* __restrict__ phi,
    const float* __restrict__ B,
    const float* __restrict__ beta,
    const float* __restrict__ u_c,
    const float* __restrict__ u_b,
    float m, float u_reg, float water_drag, float sliding_law,
    float n, float eps_reg, float dx,
    int n_sigma,
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
    float U_bar_v = sqrtf(u_ctr*u_ctr + v_ctr*v_ctr);

    float eps_mem_v = get_membrane_eps_sq(u, v, i, j, dx, ny, nx);

    float H_c = get_cell(H, i, j, ny, nx);
    float B_c = get_cell(B, i, j, ny, nx);
    float grounded = get_cell(phi, i, j, ny, nx);
    float beta_grounded = get_cell(beta, i, j, ny, nx) * grounded;
    float u_c_c = get_cell(u_c, i, j, ny, nx);
    int idx = i * nx + j;
    float U_b_warm = fminf(fmaxf(u_b[idx], 0.0f), U_bar_v);

    DualFloat eta_d, F2_d, U_b_d, be_d;

    // Seed 1: d/d(eps_mem^2)
    diva_coeffs_cell<DualFloat>({eps_mem_v,1.0f}, {U_bar_v,0.0f},
            H_c, B_c, {beta_grounded,0.0f}, {u_c_c,0.0f},
            {m,0.0f}, u_reg, water_drag, sliding_law,
            glen_exp, eps_reg, n_sigma, U_b_warm,
            eta_d, F2_d, U_b_d, be_d);
    float d_eta_deps = eta_d.d;
    float d_be_deps  = be_d.d;

    // Seed 2: d/d(Ubar)
    diva_coeffs_cell<DualFloat>({eps_mem_v,0.0f}, {U_bar_v,1.0f},
            H_c, B_c, {beta_grounded,0.0f}, {u_c_c,0.0f},
            {m,0.0f}, u_reg, water_drag, sliding_law,
            glen_exp, eps_reg, n_sigma, U_b_warm,
            eta_d, F2_d, U_b_d, be_d);
    float d_eta_dU = eta_d.d;
    float d_be_dU  = be_d.d;

    // Seed 3: d/d(beta).  The perturbation is `grounded` rather than 1 because the
    // closure is given beta*grounded, so this yields the derivative with respect to the
    // raw beta the inversion actually controls.  Note beta moves eta_bar as well as
    // beta_eff (beta -> c -> tau_b -> shear term), which is the path the parameter
    // gradient was missing.
    diva_coeffs_cell<DualFloat>({eps_mem_v,0.0f}, {U_bar_v,0.0f},
            H_c, B_c, {beta_grounded,grounded}, {u_c_c,0.0f},
            {m,0.0f}, u_reg, water_drag, sliding_law,
            glen_exp, eps_reg, n_sigma, U_b_warm,
            eta_d, F2_d, U_b_d, be_d);
    float d_eta_dbeta = eta_d.d;
    float d_be_dbeta  = be_d.d;

    // Seed 4: d/d(u_c), the regularized-Coulomb threshold speed.  Identically zero under
    // Weertman, where the u_c branch is not taken -- the same way SSA's d_u_c is.
    diva_coeffs_cell<DualFloat>({eps_mem_v,0.0f}, {U_bar_v,0.0f},
            H_c, B_c, {beta_grounded,0.0f}, {u_c_c,1.0f},
            {m,0.0f}, u_reg, water_drag, sliding_law,
            glen_exp, eps_reg, n_sigma, U_b_warm,
            eta_d, F2_d, U_b_d, be_d);
    float d_eta_duc = eta_d.d;
    float d_be_duc  = be_d.d;

    // Seed 5: d/d(m), the Weertman exponent.  m is a single global scalar, so these are
    // the per-cell contributions that compute_gradient_param_sum_diva reduces.  Getting
    // this needed __powf(dual, dual) -- the exponent, not just the base, is now dual.
    // Identically zero under regularized Coulomb.
    diva_coeffs_cell<DualFloat>({eps_mem_v,0.0f}, {U_bar_v,0.0f},
            H_c, B_c, {beta_grounded,0.0f}, {u_c_c,0.0f},
            {m,1.0f}, u_reg, water_drag, sliding_law,
            glen_exp, eps_reg, n_sigma, U_b_warm,
            eta_d, F2_d, U_b_d, be_d);
    float d_eta_dm = eta_d.d;
    float d_be_dm  = be_d.d;

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
extern "C" __global__
void compute_diva_vjp_coeffs(
    float* __restrict__ r_u,
    float* __restrict__ r_v,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ W_eta,
    const float* __restrict__ W_be,
    const float* __restrict__ deta_deps,
    const float* __restrict__ deta_dU,
    const float* __restrict__ dbe_deps,
    const float* __restrict__ dbe_dU,
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
    if (A == 0.0f && B == 0.0f) return;

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
    float U_bar = sqrtf(u_ctr*u_ctr + v_ctr*v_ctr);
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
