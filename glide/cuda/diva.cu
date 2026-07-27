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
    float U_b = fminf(fmaxf(u_b[idx], 0.0f), U_bar);
    DualFloat coeff = get_diva_drag_coeff({U_b, 1.0f}, beta_grounded, m, u_reg, water_drag, u_c_c, sliding_law);
    float tau_b = coeff.v * U_b;

    // Fixed iteration counts (no per-cell convergence test) keep the cost uniform
    // across threads and avoid divergence; the outer nonlinear iteration that calls
    // this kernel absorbs any residual error.
    const int coupling_iters = 3;   // tau_b <-> (eta(z), F2) <-> U_b
    const int eta_iters = 3;        // eta_k is implicit in itself through the shear term
    const int newton_iters = 4;     // R' >= 1, so this converges very fast

    // Midpoint rule in zeta = (s - z)/H over [0,1]: zeta = 0 at the surface, 1 at the bed.
    float w = 1.0f/(float)n_sigma;

    float eta_avg = 0.0f;
    float F2_c = 0.0f;

    for (int c = 0; c < coupling_iters; ++c) {
        eta_avg = 0.0f;
        F2_c = 0.0f;

        for (int k = 0; k < n_sigma; ++k) {
            float zeta = ((float)k + 0.5f)*w;

            // The shear ansatz tau_xz = tau_b*(s-z)/H gives eps_xz = tau_b*zeta/(2*eta)
            // -- the H cancels -- so eta is implicit in itself.  Iterate from the
            // membrane-only value (which is also the SSA value, recovered when tau_b = 0).
            float eta_k = 0.5f*B_c*__powf(eps_mem_sq + eps_reg, glen_exp);
            for (int e = 0; e < eta_iters; ++e) {
                float eps_shear = tau_b*zeta/(2.0f*eta_k);
                eta_k = 0.5f*B_c*__powf(eps_mem_sq + eps_shear*eps_shear + eps_reg, glen_exp);
            }

            eta_avg += w*eta_k;                  // eta_bar = int_0^1 eta dzeta
            F2_c    += w*zeta*zeta/eta_k;        // F2 = H * int_0^1 zeta^2/eta dzeta
        }
        F2_c *= H_c;

        // Closure: Newton on R(U_b) = U_b + f(U_b)*F2 - U_bar, where f(U) = c(U)*U, so
        // R' = 1 + f'(U_b)*F2 with f and f' both read off the dual product c(U)*U.
        for (int it = 0; it < newton_iters; ++it) {
            coeff = get_diva_drag_coeff({U_b, 1.0f}, beta_grounded, m, u_reg, water_drag, u_c_c, sliding_law);
            DualFloat f = coeff * DualFloat{U_b, 1.0f};
            float R = U_b + f.v*F2_c - U_bar;
            float dR = 1.0f + f.d*F2_c;
            U_b = fmaxf(U_b - R/dR, 0.0f);
        }

        coeff = get_diva_drag_coeff({U_b, 1.0f}, beta_grounded, m, u_reg, water_drag, u_c_c, sliding_law);
        tau_b = coeff.v * U_b;
    }

    if (is_active) {
        eta_bar[idx] = eta_avg;
        F2[idx] = F2_c;
        u_b[idx] = U_b;
        // Goldberg eq 41: the secant drag the 2D momentum solve sees, tau_b = beta_eff*U_bar.
        // Strictly non-negative, and finite at rest (no division by the speed).
        beta_eff[idx] = coeff.v/(1.0f + coeff.v*F2_c);
    }
}
