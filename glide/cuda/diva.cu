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

__device__ __forceinline__
float get_membrane_eps_sq(
    const float* __restrict__ u,
    const float* __restrict__ v,
    int i, int j,
    float dx,
    int ny, int nx){

    // Membrane (horizontal) part of the effective strain-rate invariant, returned
    // *unregularized* so the caller can add the vertical shear terms before forming
    // eta.  The strain rates mirror populate_viscosity (viscosity.cu); that function
    // is left untouched so the SSA path is unaffected.
    float dx_inv = 1.0f/dx;

    float u_l = get_vfacet(u, i, j, ny, nx);
    float u_r = get_vfacet(u, i, j + 1, ny, nx);
    float v_t = get_hfacet(v, i, j, ny, nx);
    float v_b = get_hfacet(v, i + 1, j, ny, nx);

    float dudx = (u_r - u_l)*dx_inv;
    float dvdy = (v_t - v_b)*dx_inv;

    float tl_mask = i > 0 && j > 0;
    float u_tl = get_vfacet(u, i - 1, j, ny, nx);
    float v_lt = get_hfacet(v, i, j - 1, ny, nx);
    float eps_xy_tl = 0.5f*((u_tl - u_l)*dx_inv + (v_t - v_lt)*dx_inv)*tl_mask;

    float tr_mask = i > 0 && j < (nx - 1);
    float u_tr = get_vfacet(u, i - 1, j + 1, ny, nx);
    float v_rt = get_hfacet(v, i, j + 1, ny, nx);
    float eps_xy_tr = 0.5f*((u_tr - u_r)*dx_inv + (v_rt - v_t)*dx_inv)*tr_mask;

    float bl_mask = i < (ny - 1) && j > 0;
    float u_bl = get_vfacet(u, i + 1, j, ny, nx);
    float v_lb = get_hfacet(v, i + 1, j - 1, ny, nx);
    float eps_xy_bl = 0.5f*((u_l - u_bl)*dx_inv + (v_b - v_lb)*dx_inv)*bl_mask;

    float br_mask = i < (ny - 1) && j < (nx - 1);
    float u_br = get_vfacet(u, i + 1, j + 1, ny, nx);
    float v_rb = get_hfacet(v, i + 1, j + 1, ny, nx);
    float eps_xy_br = 0.5f*((u_r - u_br)*dx_inv + (v_rb - v_b)*dx_inv)*br_mask;

    float eps_xy2_bar = 0.25f*(eps_xy_tl*eps_xy_tl + eps_xy_tr*eps_xy_tr + eps_xy_bl*eps_xy_bl + eps_xy_br*eps_xy_br);

    return dudx*dudx + dvdy*dvdy + dudx*dvdy + eps_xy2_bar;
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
    T U, float beta_grounded, float m, float u_reg, float water_drag,
    float u_c, float sliding_law){

    // The drag coefficient c(U) of get_diva_drag_coeff (stress.cu), templated on the
    // scalar type so the same expression serves the diagnostic kernel (T = float) and
    // the JVP, where T = DualFloat carries d/d(velocity perturbation).
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
    float H_c, float B_c, float beta_grounded, float u_c_c,
    float m, float u_reg, float water_drag, float sliding_law,
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
            DualFloat cs = get_diva_drag_coeff({diva_primal(U_b),1.0f},beta_grounded,m,u_reg,water_drag,u_c_c,sliding_law);
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


__device__ __forceinline__
DualFloat get_membrane_eps_sq(
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ du,
    const float* __restrict__ dv,
    int i, int j,
    float dx,
    int ny, int nx){

    // Dual counterpart of the float version above: identical expression, with the
    // perturbation direction carried through so the caller gets
    // d(eps_mem_sq)/d(direction) alongside the value.
    float dx_inv = 1.0f/dx;

    DualFloat u_l = get_vfacet(u, du, i, j, ny, nx);
    DualFloat u_r = get_vfacet(u, du, i, j + 1, ny, nx);
    DualFloat v_t = get_hfacet(v, dv, i, j, ny, nx);
    DualFloat v_b = get_hfacet(v, dv, i + 1, j, ny, nx);

    DualFloat dudx = (u_r - u_l)*dx_inv;
    DualFloat dvdy = (v_t - v_b)*dx_inv;

    float tl_mask = i > 0 && j > 0;
    DualFloat u_tl = get_vfacet(u, du, i - 1, j, ny, nx);
    DualFloat v_lt = get_hfacet(v, dv, i, j - 1, ny, nx);
    DualFloat eps_xy_tl = 0.5f*((u_tl - u_l)*dx_inv + (v_t - v_lt)*dx_inv)*tl_mask;

    float tr_mask = i > 0 && j < (nx - 1);
    DualFloat u_tr = get_vfacet(u, du, i - 1, j + 1, ny, nx);
    DualFloat v_rt = get_hfacet(v, dv, i, j + 1, ny, nx);
    DualFloat eps_xy_tr = 0.5f*((u_tr - u_r)*dx_inv + (v_rt - v_t)*dx_inv)*tr_mask;

    float bl_mask = i < (ny - 1) && j > 0;
    DualFloat u_bl = get_vfacet(u, du, i + 1, j, ny, nx);
    DualFloat v_lb = get_hfacet(v, dv, i + 1, j - 1, ny, nx);
    DualFloat eps_xy_bl = 0.5f*((u_l - u_bl)*dx_inv + (v_b - v_lb)*dx_inv)*bl_mask;

    float br_mask = i < (ny - 1) && j < (nx - 1);
    DualFloat u_br = get_vfacet(u, du, i + 1, j + 1, ny, nx);
    DualFloat v_rb = get_hfacet(v, dv, i + 1, j + 1, ny, nx);
    DualFloat eps_xy_br = 0.5f*((u_r - u_br)*dx_inv + (v_rb - v_b)*dx_inv)*br_mask;

    DualFloat eps_xy2_bar = 0.25f*(eps_xy_tl*eps_xy_tl + eps_xy_tr*eps_xy_tr + eps_xy_bl*eps_xy_bl + eps_xy_br*eps_xy_br);

    return dudx*dudx + dvdy*dvdy + dudx*dvdy + eps_xy2_bar;
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
            H_c, B_c, beta_grounded, u_c_c,
            m, u_reg, water_drag, sliding_law,
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
