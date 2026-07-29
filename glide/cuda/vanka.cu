
// ============================================================
// THE VANKA SMOOTHER
// ============================================================
//
// The multigrid smoother.  Its job is not to solve the system but to kill the
// high-frequency part of the error, leaving the smooth part for the coarser levels.
//
// What makes it a VANKA smoother rather than a pointwise one is the choice of block: all
// the unknowns physically coupled AT ONE CELL are updated together, by forming and solving
// the small dense system that couples them exactly.  Here that block is FIVE unknowns --
//
//     index   unknown            lives on
//       0     u_l                the cell's left  facet
//       1     u_r                the cell's right facet
//       2     v_t                the cell's top   facet
//       3     v_b                the cell's bottom facet
//       4     H_c                the cell centre
//
// -- and J is that 5x5 stored row-major, so J[a*5 + b] = d(r_a)/d(x_b).  Worth keeping in
// mind while reading build_5x5_vanka, which indexes it with bare integers: J[24] is the
// mass row's dependence on H, J[20] the mass row's dependence on u_l, and so on.
//
// The velocity/thickness coupling is why the block has to be solved rather than relaxed
// componentwise: thickness sets the driving stress and the velocities set the flux, so a
// pointwise sweep chases its own tail and the smoothing rate collapses.
//
// Three properties of the block that the rest of the file leans on:
//
//   * eta is FROZEN.  The tile arrives as plain floats and is refreshed between sweeps, so
//     the local solve is exact in the 5 unknowns but lagged in the viscosity -- a Newton /
//     Picard hybrid.  This is deliberate and applies to SSA and DIVA alike: the smoother is
//     only a preconditioner, and the true d(eta)/du lives in the JVP and VJP.
//   * Facet unknowns are SHARED between two cells, so each cell's proposed update is
//     halved and accumulated with atomicAdd -- the average of the two neighbouring blocks'
//     proposals, which is what makes this additive rather than multiplicative.  H is not
//     shared, so it is stored outright.
//   * The 5x5 is solved by Doolittle LU with NO PIVOTING (below).  That is safe only
//     because the block is diagonally dominant by construction -- basal drag on the
//     momentum diagonals, 1/dt on the mass diagonal -- and the ssa_damping / mc_damping
//     knobs exist to reinforce exactly that when a hard configuration threatens it.
//
// ============================================================
// LU Solve for 5x5 Systems (Vanka smoother)
// ============================================================
__device__ void lu_5x5_solve(
    const float* A,  // 25 entries: full 5x5 row-major
    const float* b,  // 5 entries
    float* x)        // 5 entries (output)
{
    float LU[5][5];

    #pragma unroll
    for (int i = 0; i < 5; i++) {
        #pragma unroll
        for (int j = 0; j < 5; j++) {
            LU[i][j] = A[i * 5 + j];
        }
    }

    // LU factorization (Doolittle, no pivoting)
    #pragma unroll
    for (int k = 0; k < 5; k++) {
        float inv_diag = 1.0f / LU[k][k];
        #pragma unroll
        for (int i = k + 1; i < 5; i++) {
            LU[i][k] *= inv_diag;
            #pragma unroll
            for (int j = k + 1; j < 5; j++) {
                LU[i][j] -= LU[i][k] * LU[k][j];
            }
        }
    }

    // Forward solve: L*y = b
    float y[5];
    y[0] = b[0];
    y[1] = b[1] - LU[1][0]*y[0];
    y[2] = b[2] - LU[2][0]*y[0] - LU[2][1]*y[1];
    y[3] = b[3] - LU[3][0]*y[0] - LU[3][1]*y[1] - LU[3][2]*y[2];
    y[4] = b[4] - LU[4][0]*y[0] - LU[4][1]*y[1] - LU[4][2]*y[2] - LU[4][3]*y[3];

    // Backward solve: U*x = y
    x[4] = y[4] / LU[4][4];
    x[3] = (y[3] - LU[3][4]*x[4]) / LU[3][3];
    x[2] = (y[2] - LU[2][3]*x[3] - LU[2][4]*x[4]) / LU[2][2];
    x[1] = (y[1] - LU[1][2]*x[2] - LU[1][3]*x[3] - LU[1][4]*x[4]) / LU[1][1];
    x[0] = (y[0] - LU[0][1]*x[1] - LU[0][2]*x[2] - LU[0][3]*x[3] - LU[0][4]*x[4]) / LU[0][0];
}

// ============================================================
// LU Solve for 6x6 Systems (DIVA Vanka smoother)
// ============================================================
// Identical scheme to lu_5x5_solve, one row wider: DIVA augments the local system
// with the basal speed U_b and its closure row.
__device__ void lu_6x6_solve(
    const float* A,  // 36 entries: full 6x6 row-major
    const float* b,  // 6 entries
    float* x)        // 6 entries (output)
{
    float LU[6][6];

    #pragma unroll
    for (int i = 0; i < 6; i++) {
        #pragma unroll
        for (int j = 0; j < 6; j++) {
            LU[i][j] = A[i * 6 + j];
        }
    }

    // LU factorization (Doolittle, no pivoting)
    #pragma unroll
    for (int k = 0; k < 6; k++) {
        float inv_diag = 1.0f / LU[k][k];
        #pragma unroll
        for (int i = k + 1; i < 6; i++) {
            LU[i][k] *= inv_diag;
            #pragma unroll
            for (int j = k + 1; j < 6; j++) {
                LU[i][j] -= LU[i][k] * LU[k][j];
            }
        }
    }

    // Forward solve: L*y = b
    float y[6];
    y[0] = b[0];
    y[1] = b[1] - LU[1][0]*y[0];
    y[2] = b[2] - LU[2][0]*y[0] - LU[2][1]*y[1];
    y[3] = b[3] - LU[3][0]*y[0] - LU[3][1]*y[1] - LU[3][2]*y[2];
    y[4] = b[4] - LU[4][0]*y[0] - LU[4][1]*y[1] - LU[4][2]*y[2] - LU[4][3]*y[3];
    y[5] = b[5] - LU[5][0]*y[0] - LU[5][1]*y[1] - LU[5][2]*y[2] - LU[5][3]*y[3] - LU[5][4]*y[4];

    // Backward solve: U*x = y
    x[5] = y[5] / LU[5][5];
    x[4] = (y[4] - LU[4][5]*x[5]) / LU[4][4];
    x[3] = (y[3] - LU[3][4]*x[4] - LU[3][5]*x[5]) / LU[3][3];
    x[2] = (y[2] - LU[2][3]*x[3] - LU[2][4]*x[4] - LU[2][5]*x[5]) / LU[2][2];
    x[1] = (y[1] - LU[1][2]*x[2] - LU[1][3]*x[3] - LU[1][4]*x[4] - LU[1][5]*x[5]) / LU[1][1];
    x[0] = (y[0] - LU[0][1]*x[1] - LU[0][2]*x[2] - LU[0][3]*x[3] - LU[0][4]*x[4] - LU[0][5]*x[5]) / LU[0][0];
}

__device__ __forceinline__
void mat5x5_mat(const float* __restrict__ A,
                const float* __restrict__ B,
                float* __restrict__ C)
{
    #pragma unroll
    for (int i = 0; i < 5; ++i)
    {
        #pragma unroll
        for (int j = 0; j < 5; ++j)
        {
            float sum = 0.0;
            #pragma unroll
            for (int k = 0; k < 5; ++k)
            {
                sum += A[5*i + k] * B[5*k + j];
            }
            C[5*i + j] = sum;
        }
    }
}

__device__ __forceinline__
void mat5x5_vec(const float* __restrict__ A,
                const float* __restrict__ x,
                float* __restrict__ y)
{
    #pragma unroll
    for (int i = 0; i < 5; ++i)
    {
        double sum = 0.0;
        #pragma unroll
        for (int j = 0; j < 5; ++j)
        {
            sum += A[5*i + j] * x[j];
        }
        y[i] = sum;
    }
}
	
// Shared local-block assembly for the SSA and DIVA smoothers.  As in residual_body,
// DIVA is a compile-time flag and the two schemes differ only in the basal stress:
// SSA evaluates the sliding law, DIVA uses the effective drag beta_eff.  When DIVA is
// set, dr_dU_b receives d(r)/d(beta_eff of this cell) for the four momentum rows --
// the coupling the augmented block needs; it is left untouched for SSA.
// Assemble the local 5x5 Jacobian J and residual r for the cell (i,j).
//
// Every entry comes from the same get_*_jac functions the global residual uses, so this is
// not a second discretisation -- it is a projection of the same one onto the five unknowns
// this block owns.  Contributions from neighbouring cells' unknowns are simply dropped:
// that is what makes it a block SMOOTHER rather than a solve, and what the outer multigrid
// V-cycle exists to make up for.
//
// For DIVA it additionally returns dr_dbeta_eff, the five rows' sensitivity to this cell's
// effective drag.  That is the vector the rank-1 condensation in vanka_smooth_body needs;
// see the long comment there.
template <bool DIVA, int height, int width>
__device__ void build_5x5_vanka(
    float* __restrict__ J,
    float* __restrict__ r,
    float* __restrict__ dr_dbeta_eff,
    float u_l, float u_r,
    float v_t, float v_b,
    float H_c,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ H,
    const float (&eta_local)[height][width],
    const float* __restrict__ phi,
    const float* __restrict__ bed,
    const float* __restrict__ B,
    const float* __restrict__ beta,
    const float* __restrict__ u_c,
    const float* __restrict__ beta_eff,
    const float* __restrict__ gamma,
    float n, float eps_reg, float flotation_reg_driving,
    float m, float u_reg, float water_drag, float flotation_reg_sliding, float sliding_law, 
    float calving_rate, float flotation_reg_calving,
    float dx, float dt,
    int ny, int nx,
    int i, int j,
    int bi, int bj)
{
    float dx_inv = 1.0f/dx;

    for (int k=0;k<25;k++) J[k] = 0.0f;
    for (int k=0;k<5;k++) r[k] = 0.0f;
    if (DIVA) { for (int k=0;k<5;k++) dr_dbeta_eff[k] = 0.0f; }

    float phi_c = get_cell(phi,i,j,ny,nx);
    float phi_l = get_cell(phi,i,j-1,ny,nx);
    float phi_r = get_cell(phi,i,j+1,ny,nx);
    float phi_t = get_cell(phi,i-1,j,ny,nx);
    float phi_b = get_cell(phi,i+1,j,ny,nx);

    // Mass Conservation Assembly
    {
    // Standard Mass Conservation: dH/dt + div(q) - smb = 0
    J[24] = 1.0f / dt;
    r[4] += H_c/dt;

    // X-Fluxes
    float H_l = get_cell(H,i,j-1,ny,nx);
    HorizontalFluxJacobian j_l = get_horizontal_flux_jac({u_l, H_l, H_c}, i, j, ny, nx);
    J[20] -= j_l.d_u   * dx_inv;
    J[24] -= j_l.d_H_r * dx_inv;
    r[4]  -= j_l.res   * dx_inv;


    FacetCalvingJacobian j_calve_l = get_facet_calving_jac({H_c,H_l,phi_c,phi_l,calving_rate,flotation_reg_calving},i,j,ny,nx);
    J[24] += j_calve_l.d_H_this * dx_inv;
    r[4] += j_calve_l.res*dx_inv;

    float H_r = get_cell(H,i,j+1,ny,nx);
    HorizontalFluxJacobian j_r = get_horizontal_flux_jac({u_r, H_c, H_r}, i, j+1, ny, nx);
    J[21] += j_r.d_u   * dx_inv;
    J[24] += j_r.d_H_l * dx_inv;
    r[4]  += j_r.res   * dx_inv;
    
    FacetCalvingJacobian j_calve_r = get_facet_calving_jac({H_c,H_r,phi_c,phi_r,calving_rate,flotation_reg_calving},i,j,ny,nx);
    J[24] += j_calve_r.d_H_this * dx_inv;
    r[4] += j_calve_r.res * dx_inv;

    // Y-Fluxes (Vertical in grid coordinates)
    float H_t = get_cell(H,i-1,j,ny,nx);
    VerticalFluxJacobian j_t = get_vertical_flux_jac({v_t, H_t, H_c}, i, j, ny, nx);
    J[22] += j_t.d_v   * dx_inv;
    J[24] += j_t.d_H_b * dx_inv;
    r[4]  += j_t.res   * dx_inv;

    FacetCalvingJacobian j_calve_t = get_facet_calving_jac({H_c,H_t,phi_c,phi_t,calving_rate,flotation_reg_calving},i,j,ny,nx);
    J[24] += j_calve_t.d_H_this * dx_inv;
    r[4] += j_calve_t.res * dx_inv;

    float H_b = get_cell(H,i+1,j,ny,nx);
    VerticalFluxJacobian j_b = get_vertical_flux_jac({v_b, H_c, H_b}, i+1, j, ny, nx);
    J[23] -= j_b.d_v   * dx_inv;
    J[24] -= j_b.d_H_t * dx_inv;
    r[4]  -= j_b.res   * dx_inv;

    FacetCalvingJacobian j_calve_b = get_facet_calving_jac({H_c,H_b,phi_c,phi_b,calving_rate,flotation_reg_calving},i,j,ny,nx);
    J[24] += j_calve_b.d_H_this * dx_inv;
    r[4] += j_calve_b.res * dx_inv;
    }
    
    {
    float eta_c = eta_local[bi][bj];
    EtaHCellJacobian eta_H_c = get_eta_H_cell_jac({eta_c,H_c});
    
    // Compute the contribution of sigma_xx at the center to both the left and right u-residuals (since it is used by both)
    SigmaNormalJacobian sigma_xx_c = get_sigma_xx_jac({u_l,u_r,v_t,v_b,eta_H_c.res},dx_inv,i,j,ny,nx);
    
    r[0] += sigma_xx_c.res * dx_inv;
    J[0] += sigma_xx_c.d_u_l * dx_inv;
    J[1] += sigma_xx_c.d_u_r * dx_inv;
    J[2] += sigma_xx_c.d_v_t * dx_inv;
    J[3] += sigma_xx_c.d_v_b * dx_inv;
    J[4] += sigma_xx_c.d_eta_H * eta_H_c.d_H * dx_inv;
    
    r[1] -= sigma_xx_c.res * dx_inv;
    J[5] -= sigma_xx_c.d_u_l * dx_inv;
    J[6] -= sigma_xx_c.d_u_r * dx_inv;
    J[7] -= sigma_xx_c.d_v_t * dx_inv;
    J[8] -= sigma_xx_c.d_v_b * dx_inv;
    J[9] -= sigma_xx_c.d_eta_H * eta_H_c.d_H * dx_inv;

    SigmaNormalJacobian sigma_yy_c = get_sigma_yy_jac({u_l,u_r,v_t,v_b,eta_H_c.res},dx_inv,i,j,ny,nx);
    r[2]  -= sigma_yy_c.res * dx_inv;
    J[10] -= sigma_yy_c.d_u_l * dx_inv;
    J[11] -= sigma_yy_c.d_u_r * dx_inv;
    J[12] -= sigma_yy_c.d_v_t * dx_inv;
    J[13] -= sigma_yy_c.d_v_b * dx_inv;
    J[14] -= sigma_yy_c.d_eta_H * eta_H_c.d_H * dx_inv;

    r[3]  += sigma_yy_c.res * dx_inv;
    J[15] += sigma_yy_c.d_u_l * dx_inv;
    J[16] += sigma_yy_c.d_u_r * dx_inv;
    J[17] += sigma_yy_c.d_v_t * dx_inv;
    J[18] += sigma_yy_c.d_v_b * dx_inv;
    J[19] += sigma_yy_c.d_eta_H * eta_H_c.d_H * dx_inv;
    }

    // Compute the contribution of sigma_xx from the left cell to the left u-residual
    {
    float eta_l  = eta_local[bi][bj - 1];
    float H_l    = get_cell(H,i,j-1,ny,nx);
    EtaHCellJacobian eta_H_l = get_eta_H_cell_jac({eta_l,H_l});

    float u_ll   = get_vfacet(u,i,j-1,ny,nx);
    float v_lt   = get_hfacet(v,i,j-1,ny,nx);
    float v_lb   = get_hfacet(v,i+1,j-1,ny,nx);
    SigmaNormalJacobian sigma_xx_l = get_sigma_xx_jac({u_ll,u_l,v_lt,v_lb,eta_H_l.res},dx_inv,i,j - 1,ny,nx);
    r[0] -= sigma_xx_l.res * dx_inv;
    J[0] -= sigma_xx_l.d_u_r * dx_inv;
    }

    // Compute the contribution of sigma_xx from the right cell to the right u-residual
    {
    float eta_r  = eta_local[bi][bj + 1];
    float H_r    = get_cell(H,i,j+1,ny,nx);
    EtaHCellJacobian eta_H_r = get_eta_H_cell_jac({eta_r,H_r});

    float u_rr   = get_vfacet(u,i,j+2,ny,nx);
    float v_rt   = get_hfacet(v,i,j+1,ny,nx);
    float v_rb   = get_hfacet(v,i+1,j+1,ny,nx);
    SigmaNormalJacobian sigma_xx_r = get_sigma_xx_jac({u_r,u_rr,v_rt,v_rb,eta_H_r.res},dx_inv,i,j + 1,ny,nx);
    r[1] += sigma_xx_r.res * dx_inv;
    J[6] += sigma_xx_r.d_u_l * dx_inv;
    }

    // Compute the contribution of sigma_yy from the top cell to the top v-residual
    {
    float eta_t  = eta_local[bi - 1][bj];
    float H_t    = get_cell(H,i-1,j,ny,nx);
    EtaHCellJacobian eta_H_t = get_eta_H_cell_jac({eta_t,H_t});

    float u_tl   = get_vfacet(u,i-1,j,ny,nx);
    float u_tr   = get_vfacet(u,i-1,j+1,ny,nx);
    float v_tt   = get_hfacet(v,i-1,j,ny,nx);
    SigmaNormalJacobian sigma_yy_t = get_sigma_yy_jac({u_tl,u_tr,v_tt,v_t,eta_H_t.res},dx_inv,i - 1,j,ny,nx);
    r[2] += sigma_yy_t.res * dx_inv;
    J[12] += sigma_yy_t.d_v_b * dx_inv;
    }

    // Compute the contribution of sigma_yy from the bottom cell to the bottom v-residual
    {
    float eta_b  = eta_local[bi + 1][bj];
    float H_b    = get_cell(H,i + 1,j,ny,nx);
    EtaHCellJacobian eta_H_b = get_eta_H_cell_jac({eta_b,H_b});

    float u_bl   = get_vfacet(u,i+1,j,ny,nx);
    float u_br   = get_vfacet(u,i+1,j+1,ny,nx);
    float v_bb   = get_hfacet(v,i+2,j,ny,nx);
    SigmaNormalJacobian sigma_yy_b = get_sigma_yy_jac({u_bl,u_br,v_b,v_bb,eta_H_b.res},dx_inv,i + 1,j,ny,nx);
    r[3] -= sigma_yy_b.res * dx_inv;
    J[18] -= sigma_yy_b.d_v_t * dx_inv;
    }
    
    
    // Compute the contribution of sigma_xy from the top-left corner to the left u-residual and top v-residual
    {
    float eta_tl = eta_local[bi - 1][bj - 1];
    float eta_t  = eta_local[bi - 1][bj];
    float eta_l  = eta_local[bi][bj - 1];
    float eta_c  = eta_local[bi][bj];
    
    float H_tl   = get_cell(H,i-1,j-1,ny,nx);
    float H_t    = get_cell(H,i-1,j,ny,nx);
    float H_l    = get_cell(H,i,j-1,ny,nx);
    
    EtaHVertexJacobian eta_H_tl = get_eta_H_vertex_jac({eta_tl,eta_t,eta_l,eta_c,H_tl,H_t,H_l,H_c});
    
    float u_tl = get_vfacet(u,i-1,j,ny,nx);
    float v_lt = get_hfacet(v,i,j-1,ny,nx);
    
    SigmaShearJacobian sigma_xy_tl = get_sigma_xy_jac({u_tl,u_l,v_lt,v_t,eta_H_tl.res},dx_inv,i,j,ny,nx);
    r[0] += sigma_xy_tl.res * dx_inv;
    J[0] += sigma_xy_tl.d_u_b * dx_inv;
    J[4] += sigma_xy_tl.d_eta_H * eta_H_tl.d_H_br * dx_inv;

    r[2] -= sigma_xy_tl.res * dx_inv;
    J[12] -= sigma_xy_tl.d_v_r * dx_inv;
    J[14] -= sigma_xy_tl.d_eta_H * eta_H_tl.d_H_br * dx_inv;
    }

    // Compute the contribution of sigma_xy from the top-right corner to the right u-residual and top v-residual
    {
    float eta_t  = eta_local[bi - 1][bj];
    float eta_tr = eta_local[bi - 1][bj + 1];
    float eta_c  = eta_local[bi][bj];
    float eta_r  = eta_local[bi][bj + 1];
    
    float H_t    = get_cell(H,i-1,j,ny,nx);
    float H_tr   = get_cell(H,i-1,j+1,ny,nx);
    float H_r    = get_cell(H,i,j+1,ny,nx);
    
    EtaHVertexJacobian eta_H_tr = get_eta_H_vertex_jac({eta_t,eta_tr,eta_c,eta_r,H_t,H_tr,H_c,H_r});
    
    float u_tr = get_vfacet(u,i-1,j+1,ny,nx);
    float v_rt = get_hfacet(v,i,j+1,ny,nx);
    
    SigmaShearJacobian sigma_xy_tr = get_sigma_xy_jac({u_tr,u_r,v_t,v_rt,eta_H_tr.res},dx_inv,i,j+1,ny,nx);
    r[1] += sigma_xy_tr.res * dx_inv;
    J[6] += sigma_xy_tr.d_u_b * dx_inv;
    J[9] += sigma_xy_tr.d_eta_H * eta_H_tr.d_H_bl * dx_inv;

    r[2] += sigma_xy_tr.res * dx_inv;
    J[12] += sigma_xy_tr.d_v_l * dx_inv;
    J[14] += sigma_xy_tr.d_eta_H * eta_H_tr.d_H_bl * dx_inv;
    }

    // Compute the contribution of sigma_xy from the bottom-left corner to the left u-residual and bottom v-residual
    {
    float eta_l  = eta_local[bi][bj - 1];
    float eta_c  = eta_local[bi][bj];
    float eta_bl = eta_local[bi + 1][bj - 1];
    float eta_b  = eta_local[bi + 1][bj];
    
    float H_l    = get_cell(H,i,j-1,ny,nx);
    float H_bl   = get_cell(H,i+1,j-1,ny,nx);
    float H_b    = get_cell(H,i+1,j,ny,nx);

    EtaHVertexJacobian eta_H_bl = get_eta_H_vertex_jac({eta_l,eta_c,eta_bl,eta_b,H_l,H_c,H_bl,H_b});
    
    float u_bl   = get_vfacet(u,i+1,j,ny,nx);
    float v_lb   = get_hfacet(v,i+1,j-1,ny,nx);
    SigmaShearJacobian sigma_xy_bl = get_sigma_xy_jac({u_l,u_bl,v_lb,v_b,eta_H_bl.res},dx_inv,i + 1,j,ny,nx);
    r[0] -= sigma_xy_bl.res * dx_inv;
    J[0] -= sigma_xy_bl.d_u_t * dx_inv;
    J[4] -= sigma_xy_bl.d_eta_H * eta_H_bl.d_H_tr * dx_inv;

    r[3] -= sigma_xy_bl.res * dx_inv;
    J[18] -= sigma_xy_bl.d_v_r * dx_inv;
    J[19] -= sigma_xy_bl.d_eta_H * eta_H_bl.d_H_tr * dx_inv;
    }

    // Compute the contribution of sigma_xy from the bottom-right corner to the right u-residual and bottom v-residual
    {
    float eta_c  = eta_local[bi][bj];
    float eta_r  = eta_local[bi][bj + 1];
    float eta_b  = eta_local[bi + 1][bj];
    float eta_br = eta_local[bi + 1][bj + 1];
    
    float H_r    = get_cell(H,i,j+1,ny,nx);
    float H_b    = get_cell(H,i+1,j,ny,nx);
    float H_br   = get_cell(H,i+1,j+1,ny,nx);

    EtaHVertexJacobian eta_H_br = get_eta_H_vertex_jac({eta_c,eta_r,eta_b,eta_br,H_c,H_r,H_b,H_br});
    
    float u_br   = get_vfacet(u,i+1,j+1,ny,nx);
    float v_rb   = get_hfacet(v,i+1,j+1,ny,nx);
    SigmaShearJacobian sigma_xy_br = get_sigma_xy_jac({u_r,u_br,v_b,v_rb,eta_H_br.res},dx_inv,i + 1,j + 1,ny,nx);
    r[1] -= sigma_xy_br.res * dx_inv;
    J[6] -= sigma_xy_br.d_u_t * dx_inv;
    J[9] -= sigma_xy_br.d_eta_H * eta_H_br.d_H_tl * dx_inv;

    r[3] += sigma_xy_br.res * dx_inv;
    J[18] += sigma_xy_br.d_v_l * dx_inv;
    J[19] += sigma_xy_br.d_eta_H * eta_H_br.d_H_tl * dx_inv;
    }
    
    
    // Basal shear stress for left momentum
    {
    float v_tl   = get_hfacet(v,i,j-1,ny,nx);
    float v_bl   = get_hfacet(v,i+1,j-1,ny,nx);

    float H_l    = get_cell(H,i,j-1,ny,nx);
    float beta_l = get_cell(beta,i,j-1,ny,nx);
    float u_c_l = get_cell(u_c,i,j-1,ny,nx);
    float beta_c = get_cell(beta,i,j,ny,nx);
    float u_c_c = get_cell(u_c,i,j,ny,nx);
    if (DIVA) {
	// This facet's cells are (i,j-1) and (i,j); the block owns (i,j), so it is the
	// "r" side whose beta_eff carries the in-block U_b dependence.
	float beta_eff_l = get_cell(beta_eff,i,j-1,ny,nx);
	float beta_eff_c = get_cell(beta_eff,i,j,ny,nx);
	TauBxDivaJacobian tau_bx_l = get_tau_bx_diva_jac(u_l,beta_eff_l,beta_eff_c);
	r[0] += tau_bx_l.res;
	J[0] += tau_bx_l.d_u;
	dr_dbeta_eff[0] += tau_bx_l.d_beta_eff_r;
    } else {
	TauBxJacobian tau_bx_l = get_tau_bx_jac({u_l,v_tl,v_t,v_bl,v_b,H_l,H_c,phi_l,phi_c,beta_l,beta_c,m,u_reg,water_drag,flotation_reg_sliding,u_c_l,u_c_c,sliding_law});
	r[0] += tau_bx_l.res;
	J[0] += tau_bx_l.d_u;
	J[2] += tau_bx_l.d_v_tr;
	J[3] += tau_bx_l.d_v_br;
	J[4] += tau_bx_l.d_H_r;
    }
    }

    // Basal shear stress for right momentum
    {
    float v_tr   = get_hfacet(v,i,j+1,ny,nx);
    float v_br   = get_hfacet(v,i+1,j+1,ny,nx);
    
    float H_r    = get_cell(H,i,j+1,ny,nx);
    float beta_c = get_cell(beta,i,j,ny,nx);
    float u_c_c = get_cell(u_c,i,j,ny,nx);
    float beta_r = get_cell(beta,i,j+1,ny,nx);
    float u_c_r = get_cell(u_c,i,j+1,ny,nx);
    if (DIVA) {
	// Cells (i,j) and (i,j+1): the block owns the "l" side here.
	float beta_eff_c = get_cell(beta_eff,i,j,ny,nx);
	float beta_eff_r = get_cell(beta_eff,i,j+1,ny,nx);
	TauBxDivaJacobian tau_bx_r = get_tau_bx_diva_jac(u_r,beta_eff_c,beta_eff_r);
	r[1] += tau_bx_r.res;
	J[6] += tau_bx_r.d_u;
	dr_dbeta_eff[1] += tau_bx_r.d_beta_eff_l;
    } else {
	TauBxJacobian tau_bx_r = get_tau_bx_jac({u_r,v_t,v_tr,v_b,v_br,H_c,H_r,phi_c,phi_r,beta_c,beta_r,m,u_reg,water_drag,flotation_reg_sliding,u_c_c,u_c_r,sliding_law});
	r[1] += tau_bx_r.res;
	J[6] += tau_bx_r.d_u;
	J[7] += tau_bx_r.d_v_tl;
	J[8] += tau_bx_r.d_v_bl;
	J[9] += tau_bx_r.d_H_l;
    }
    }

    // Basal shear stress for top momentum
    {
    float u_tl = get_vfacet(u,i-1,j,ny,nx);
    float u_tr = get_vfacet(u,i-1,j+1,ny,nx);

    float H_t    = get_cell(H,i-1,j,ny,nx);
    float beta_t = get_cell(beta,i-1,j,ny,nx);
    float u_c_t = get_cell(u_c,i-1,j,ny,nx);
    float beta_c = get_cell(beta,i,j,ny,nx);
    float u_c_c = get_cell(u_c,i,j,ny,nx);
    if (DIVA) {
	// Cells (i-1,j) and (i,j): the block owns the "b" side here.
	float beta_eff_t = get_cell(beta_eff,i-1,j,ny,nx);
	float beta_eff_c = get_cell(beta_eff,i,j,ny,nx);
	TauByDivaJacobian tau_by_t = get_tau_by_diva_jac(v_t,beta_eff_t,beta_eff_c);
	r[2]  += tau_by_t.res;
	J[12] += tau_by_t.d_v;
	dr_dbeta_eff[2] += tau_by_t.d_beta_eff_b;
    } else {
	TauByJacobian tau_by_t = get_tau_by_jac({v_t,u_tl,u_tr,u_l,u_r,H_t,H_c,phi_t,phi_c,beta_t,beta_c,m,u_reg,water_drag,flotation_reg_sliding,u_c_t,u_c_c,sliding_law});
	r[2]  += tau_by_t.res;
	J[12] += tau_by_t.d_v;
	J[10] += tau_by_t.d_u_bl;
	J[11] += tau_by_t.d_u_br;
	J[14] += tau_by_t.d_H_b;
    }
    }

    // Basal shear stress for bottom momentum
    {
    float u_bl = get_vfacet(u,i+1,j,ny,nx);
    float u_br = get_vfacet(u,i+1,j+1,ny,nx);

    float H_b    = get_cell(H,i+1,j,ny,nx);
    float beta_c = get_cell(beta,i,j,ny,nx);
    float u_c_c = get_cell(u_c,i,j,ny,nx);
    float beta_b = get_cell(beta,i+1,j,ny,nx);
    float u_c_b = get_cell(u_c,i+1,j,ny,nx);
    if (DIVA) {
	// Cells (i,j) and (i+1,j): the block owns the "t" side here.
	float beta_eff_c = get_cell(beta_eff,i,j,ny,nx);
	float beta_eff_b = get_cell(beta_eff,i+1,j,ny,nx);
	TauByDivaJacobian tau_by_b = get_tau_by_diva_jac(v_b,beta_eff_c,beta_eff_b);
	r[3]  += tau_by_b.res;
	J[18] += tau_by_b.d_v;
	dr_dbeta_eff[3] += tau_by_b.d_beta_eff_t;
    } else {
	TauByJacobian tau_by_b = get_tau_by_jac({v_b,u_l,u_r,u_bl,u_br,H_c,H_b,phi_c,phi_b,beta_c,beta_b,m,u_reg,water_drag,flotation_reg_sliding,u_c_c,u_c_b,sliding_law});
	r[3]  += tau_by_b.res;
	J[18] += tau_by_b.d_v;
	J[15] += tau_by_b.d_u_tl;
	J[16] += tau_by_b.d_u_tr;
	J[19] += tau_by_b.d_H_t;
    }
    }
    
    // Driving stress for left momentum (u)
    {
    float H_l    = get_cell(H,i,j-1,ny,nx);
    float bed_l  = get_cell(bed,i,j-1,ny,nx);
    float bed_c  = get_cell(bed,i,j,ny,nx);
    TauDxJacobian tau_dx_l = get_tau_dx_jac({H_l,H_c,bed_l,bed_c,phi_l,phi_c,flotation_reg_driving},dx_inv,i,j,ny,nx);
    r[0] -= tau_dx_l.res;
    J[4] -= tau_dx_l.d_H_r;
    }

    // Driving stress for right momentum (u)
    {
    float H_r    = get_cell(H,i,j+1,ny,nx);
    float bed_c  = get_cell(bed,i,j,ny,nx);
    float bed_r  = get_cell(bed,i,j+1,ny,nx);
    TauDxJacobian tau_dx_r = get_tau_dx_jac({H_c,H_r,bed_c,bed_r,phi_c,phi_r,flotation_reg_driving},dx_inv,i,j+1,ny,nx);
    r[1] -= tau_dx_r.res;
    J[9] -= tau_dx_r.d_H_l;
    }

    // Driving stress for top momentum (v)
    {
    float H_t    = get_cell(H,i-1,j,ny,nx);
    float bed_t  = get_cell(bed,i-1,j,ny,nx);
    float bed_c  = get_cell(bed,i,j,ny,nx);
    TauDyJacobian tau_dy_t = get_tau_dy_jac({H_t,H_c,bed_t,bed_c,phi_t,phi_c,flotation_reg_driving},dx_inv,i,j,ny,nx);
    r[2]  -= tau_dy_t.res;
    J[14] -= tau_dy_t.d_H_b;
    }

    // Driving stress for bottom momentum (v)
    {
    float H_b    = get_cell(H,i+1,j,ny,nx);
    float bed_c  = get_cell(bed,i,j,ny,nx);
    float bed_b  = get_cell(bed,i+1,j,ny,nx);
    TauDyJacobian tau_dy_b = get_tau_dy_jac({H_c,H_b,bed_c,bed_b,phi_c,phi_b,flotation_reg_driving},dx_inv,i+1,j,ny,nx);
    r[3]  -= tau_dy_b.res;
    J[19] -= tau_dy_b.d_H_t;
    }
}

// Shared body for the SSA and DIVA Vanka smoothers; see build_5x5_vanka above.
// One smoothing sweep: for every cell, take a few damped Newton steps on its local 5x5
// block and write out the resulting increment.  Shared by SSA and DIVA.
//
// The Newton loop is local -- it re-assembles and re-solves the SAME 5x5 with the block's
// own updated values, which converges the block against frozen neighbours.  Iterating here
// rather than taking one step per sweep is cheap (the assembly is already in registers) and
// buys robustness where the sliding law is strongly nonlinear.
template <bool DIVA>
__device__ void vanka_smooth_body(
    float* __restrict__ delta_u,
    float* __restrict__ delta_v,
    float* __restrict__ delta_H,
    float* __restrict__ mask,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ H,
    const float* __restrict__ phi,
    const float* __restrict__ f_u,
    const float* __restrict__ f_v,
    const float* __restrict__ f_H,
    const float* __restrict__ bed,
    const float* __restrict__ B,
    const float* __restrict__ beta,
    const float* __restrict__ u_c,
    const float* __restrict__ gamma,
    const float* __restrict__ eta_bar,     // DIVA only
    const float* __restrict__ beta_eff,    // DIVA only
    const float* __restrict__ u_b,         // DIVA only
    const float* __restrict__ F2,          // DIVA only
    float n, float eps_reg, float flotation_reg_driving,
    float m, float u_reg, float water_drag, float flotation_reg_sliding, float sliding_law,
    float calving_rate, float flotation_reg_calving,
    float dx, float dt,
    int ny, int nx, int stride, int halo,
    int newton_steps, float relaxation,
    float ssa_damping, float mc_damping
    ) 
{
    const int bny = 16;
    const int bnx = 16;

    int bi = threadIdx.y;
    int bj = threadIdx.x;

    int j = blockIdx.x * stride + (threadIdx.x - halo);
    int i = blockIdx.y * stride + (threadIdx.y - halo);

    __shared__ float eta_local[bny][bnx];

    if (i < 0 || i >= ny || j<0 || j >= nx) return;

    if (DIVA) {
	eta_local[bi][bj] = get_cell(eta_bar, i, j, ny, nx);
    } else {
	populate_viscosity(eta_local, bi, bj, i, j, u, v, B, n, eps_reg, dx, ny, nx);
    }
    __syncthreads();

    bool is_active = (threadIdx.x >= halo && threadIdx.x < blockDim.x - halo) &&
                     (threadIdx.y >= halo && threadIdx.y < blockDim.y - halo);

    if ( is_active ) {
	float dx_inv = 1.0f/dx;
	float dr_dbeta_eff[5] = {0};

	float masked = get_cell(mask, i, j, ny, nx);
	float u_l = get_vfacet(u, i, j, ny, nx);
	float u_r = get_vfacet(u, i, j + 1, ny, nx);
	float v_t = get_hfacet(v, i, j, ny, nx);
	float v_b = get_hfacet(v, i + 1, j, ny, nx);
	float H_c = get_cell(H, i, j, ny, nx);
	float thklim = get_cell(gamma,i,j,ny,nx);

	float c_u_l = 0.0f;
	float c_u_r = 0.0f;
	float c_v_t = 0.0f;
	float c_v_b = 0.0f;
	float c_H_c = 0.0f;

	float rnorm = 1.0f;
	float tol = 0.000001f;
	int k = 0;

	float J[25] = {0};
        float r[5] = {0};

	while (k<newton_steps && rnorm>tol){

	    build_5x5_vanka<DIVA>(J, r, dr_dbeta_eff,
		    u_l, u_r, v_t, v_b, H_c,
		    u, v, H, eta_local, phi,
                    bed, B, beta, u_c, beta_eff, gamma,
		    n, eps_reg, flotation_reg_driving,
                    m, u_reg, water_drag, flotation_reg_sliding, sliding_law,
		    calving_rate, flotation_reg_calving,
                    dx, dt, ny, nx, i, j, bi, bj);

	    r[0] -= get_vfacet(f_u,i,j,ny,nx);
	    r[1] -= get_vfacet(f_u,i,j+1,ny,nx);
	    r[2] -= get_hfacet(f_v,i,j,ny,nx);
	    r[3] -= get_hfacet(f_v,i+1,j,ny,nx);
	    r[4] -= get_hfacet(f_H,i,j,ny,nx);

	    if (DIVA) {
		// Augmented basal-speed unknown, eliminated exactly.  Carrying U_b as a
		// sixth unknown gives the local system
		//     [ A    b ] [ du   ]   [ r    ]
		//     [ c^T  d ] [ dU_b ] = [ r_Ub ] ,
		// whose (2,2) block is the scalar d = dR_Ub/dU_b = 1 + f'(U_b)*F2 >= 1.
		// Being 1x1 and never singular, U_b can be condensed out analytically:
		//     (A - b c^T/d) du = r - b*r_Ub/d,
		// which is algebraically identical to solving the 6x6 but leaves the 5x5
		// layout (and lu_5x5_solve) untouched.  This is what upgrades the secant
		// drag that build_5x5_vanka assembled into the tangent the Newton step
		// needs -- the two differ for any nonlinear sliding law.
		//
		// U_b itself is not updated here: compute_diva_coeffs re-solves the closure
		// exactly before the next sweep, which is at least as good as taking one
		// Newton step on it.
		float phi_cc  = get_cell(phi,i,j,ny,nx);
		float U_b_c   = get_cell(u_b,i,j,ny,nx);
		float F2_c    = get_cell(F2,i,j,ny,nx);
		float beta_g  = get_cell(beta,i,j,ny,nx)*phi_cc;
		float u_c_cc  = get_cell(u_c,i,j,ny,nx);

		DualFloat coeff = get_diva_drag_coeff({U_b_c,1.0f},beta_g,m,u_reg,water_drag,u_c_cc,sliding_law);
		DualFloat f_b   = coeff * DualFloat{U_b_c,1.0f};       // f = c*U, f' = f_b.d
		float d_diag    = 1.0f + f_b.d*F2_c;

		float dbe = get_diva_dbeta_eff_du_b(U_b_c,F2_c,beta_g,m,u_reg,water_drag,u_c_cc,sliding_law);

		float u_ctr = 0.5f*(u_l + u_r);
		float v_ctr = 0.5f*(v_t + v_b);
		float U_bar = sqrtf(u_ctr*u_ctr + v_ctr*v_ctr);
		float inv_U = U_bar > 1e-6f ? 1.0f/U_bar : 0.0f;

		// b = dr/dU_b, through beta_eff of this cell.
		float bvec[5];
		for (int a=0;a<4;a++) bvec[a] = dr_dbeta_eff[a]*dbe;
		bvec[4] = 0.0f;

		// c = dR_Ub/d(unknowns), through U_bar = |ubar|.  F2 is held fixed with
		// respect to H inside the block, consistent with the lagged viscosity.
		float cvec[5];
		cvec[0] = -0.5f*u_ctr*inv_U;
		cvec[1] = cvec[0];
		cvec[2] = -0.5f*v_ctr*inv_U;
		cvec[3] = cvec[2];
		cvec[4] = 0.0f;

		float r_Ub  = U_b_c + f_b.v*F2_c - U_bar;
		float inv_d = 1.0f/d_diag;

		for (int a=0;a<5;a++) {
		    r[a] -= bvec[a]*r_Ub*inv_d;
		    for (int b2=0;b2<5;b2++) J[a*5 + b2] -= bvec[a]*cvec[b2]*inv_d;
		}
	    }

	    // Diagonal damping.  Note the SIGNS differ because the diagonals do: the
	    // momentum diagonals are negative (drag resists), the mass diagonal is +1/dt.
	    // Both of these therefore increase |diagonal|, pulling the block further from
	    // singular and shortening the step -- the local analogue of a trust region.
            J[0]  -= ssa_damping;
            J[6]  -= ssa_damping;
            J[12] -= ssa_damping;
            J[18] -= ssa_damping;
            J[24] += mc_damping;
	     
	    if (j == 0) {
	    	for(int k=0; k<5; ++k) J[0 + k] = 0.0f;
	        for(int k=0; k<5; ++k) J[k*5 + 0] = 0.0f;
		J[0] = 1.0f;
		r[0] = u_l;
	    }

	    if (j == (nx - 1)) {
	    	for(int k=0; k<5; ++k) J[5 + k] = 0.0f;
	        for(int k=0; k<5; ++k) J[k*5 + 1] = 0.0f;
		J[6] = 1.0f;
		r[1] = u_r;
	    }

	    if (i == 0) {
	    	for(int k=0; k<5; ++k) J[10 + k] = 0.0f;
	        for(int k=0; k<5; ++k) J[k*5 + 2] = 0.0f;
		J[12] = 1.0f;
		r[2] = v_t;
	    }

	    if (i == (ny-1)) {
	    	for(int k=0; k<5; ++k) J[15 + k] = 0.0f;
	        for(int k=0; k<5; ++k) J[k*5 + 3] = 0.0f;
		J[18] = 1.0f;
		r[3] = v_b;
	    }
	    

	    if ((H_c - dt*r[4]) <= (thklim)) {
		// Active set constraint: Force H = thklim
		masked = 1.0f;
		for(int k=0; k<5; ++k) J[20 + k] = 0.0f;
	        for(int k=0; k<5; ++k) J[k*5 + 4] = 0.0f;
		J[24] = 1.0f;
		r[4] = H_c - thklim;
	    } else {
	        masked = 0.0f;
	    
	    }

	    float delta_x[5] = {0};
	    lu_5x5_solve(J,r,delta_x);

	    rnorm = r[0]*r[0] + r[1]*r[1] + r[2]*r[2] + r[3]*r[3] + r[4]*r[4];

	    // NOTE: this overwrites the relaxation passed in from Python, making
	    // vanka_options.newton_options.relaxation a no-op.  Currently unobservable --
	    // the Python default is also 0.5 and every caller sets 0.5 -- but the knob does
	    // not work.  Recorded in notes/open_questions.md; left as-is because changing it
	    // would alter SSA behaviour for any caller that passes something else.
	    relaxation = 0.5f;

	    // The updates are accumulated with KAHAN COMPENSATED SUMMATION: c_* carries the
	    // rounding error lost from the previous step and is folded into the next one.  In
	    // float32, several damped Newton steps of steadily shrinking size would otherwise
	    // lose the tail of the correction to round-off, which shows up as a stalled
	    // smoother rather than as a wrong answer.
	    float y_u_l = -relaxation*delta_x[0] - c_u_l;
	    float t_u_l = u_l + y_u_l;
	    c_u_l = (t_u_l - u_l) - y_u_l;
	    u_l = t_u_l;
	    
	    float y_u_r = -relaxation*delta_x[1] - c_u_r;
	    float t_u_r = u_r + y_u_r;
	    c_u_r = (t_u_r - u_r) - y_u_r;
	    u_r = t_u_r;

	    float y_v_t = -relaxation*delta_x[2] - c_v_t;
	    float t_v_t = v_t + y_v_t;
	    c_v_t = (t_v_t - v_t) - y_v_t;
	    v_t = t_v_t;
	    
	    float y_v_b = -relaxation*delta_x[3] - c_v_b;
	    float t_v_b = v_b + y_v_b;
	    c_v_b = (t_v_b - v_b) - y_v_b;
	    v_b = t_v_b;

	    float y_H_c = -relaxation*delta_x[4] - c_H_c;
	    float t_H_c = H_c + y_H_c;
	    c_H_c = (t_H_c - H_c) - y_H_c;
	    H_c = t_H_c;

	    // Thickness floor, enforced inside the loop so subsequent steps see the clamped
	    // value.  Cells pinned here are the ones the mask then converts to H = thklim
	    // rows outright.
	    H_c = fmaxf(H_c,thklim);
	    k++;

        }
	
	float u_l_prev = get_vfacet(u, i, j, ny, nx);
	float u_r_prev = get_vfacet(u, i, j + 1, ny, nx);
	float v_t_prev = get_hfacet(v, i, j, ny, nx);
	float v_b_prev = get_hfacet(v, i + 1, j, ny, nx);
	float H_c_prev = get_cell(H, i, j, ny, nx);

	// Write the NET increment, not the state: the caller applies it as
	// x -= omega*delta.  The 0.5 halves each cell's claim on a shared facet, so the two
	// neighbouring blocks' proposals average rather than double.
	atomicAdd(&delta_u[i * (nx + 1) + j],       0.5f*(u_l - u_l_prev));
	atomicAdd(&delta_u[i * (nx + 1) + j + 1],   0.5f*(u_r - u_r_prev));
	atomicAdd(&delta_v[i * nx + j],             0.5f*(v_t - v_t_prev));
	atomicAdd(&delta_v[(i + 1) * nx + j ],      0.5f*(v_b - v_b_prev));
	delta_H[i * nx + j]           = (H_c - H_c_prev);
	mask[i * nx + j]              = masked;
    }
}

extern "C" __global__
void vanka_smooth(
    float* __restrict__ delta_u,
    float* __restrict__ delta_v,
    float* __restrict__ delta_H,
    float* __restrict__ mask,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ H,
    const float* __restrict__ phi,
    const float* __restrict__ f_u,
    const float* __restrict__ f_v,
    const float* __restrict__ f_H,
    const float* __restrict__ bed,
    const float* __restrict__ B,
    const float* __restrict__ beta,
    const float* __restrict__ u_c,
    const float* __restrict__ gamma,
    float n, float eps_reg, float flotation_reg_driving,
    float m, float u_reg, float water_drag, float flotation_reg_sliding, float sliding_law,
    float calving_rate, float flotation_reg_calving,
    float dx, float dt,
    int ny, int nx, int stride, int halo,
    int newton_steps, float relaxation,
    float ssa_damping, float mc_damping
    )
{
    vanka_smooth_body<false>(delta_u,delta_v,delta_H,mask,u,v,H,phi,f_u,f_v,f_H,
	    bed,B,beta,u_c,gamma,nullptr,nullptr,nullptr,nullptr,
	    n,eps_reg,flotation_reg_driving,
	    m,u_reg,water_drag,flotation_reg_sliding,sliding_law,
	    calving_rate,flotation_reg_calving,dx,dt,ny,nx,stride,halo,
	    newton_steps,relaxation,ssa_damping,mc_damping);
}

extern "C" __global__
void vanka_smooth_diva(
    float* __restrict__ delta_u,
    float* __restrict__ delta_v,
    float* __restrict__ delta_H,
    float* __restrict__ mask,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ H,
    const float* __restrict__ phi,
    const float* __restrict__ f_u,
    const float* __restrict__ f_v,
    const float* __restrict__ f_H,
    const float* __restrict__ bed,
    const float* __restrict__ B,
    const float* __restrict__ beta,
    const float* __restrict__ u_c,
    const float* __restrict__ gamma,
    const float* __restrict__ eta_bar,
    const float* __restrict__ beta_eff,
    const float* __restrict__ u_b,
    const float* __restrict__ F2,
    float n, float eps_reg, float flotation_reg_driving,
    float m, float u_reg, float water_drag, float flotation_reg_sliding, float sliding_law,
    float calving_rate, float flotation_reg_calving,
    float dx, float dt,
    int ny, int nx, int stride, int halo,
    int newton_steps, float relaxation,
    float ssa_damping, float mc_damping
    )
{
    vanka_smooth_body<true>(delta_u,delta_v,delta_H,mask,u,v,H,phi,f_u,f_v,f_H,
	    bed,B,beta,u_c,gamma,eta_bar,beta_eff,u_b,F2,
	    n,eps_reg,flotation_reg_driving,
	    m,u_reg,water_drag,flotation_reg_sliding,sliding_law,
	    calving_rate,flotation_reg_calving,dx,dt,ny,nx,stride,halo,
	    newton_steps,relaxation,ssa_damping,mc_damping);
}

// Shared body for the SSA and DIVA adjoint smoothers.  For DIVA the block assembled here
// is the *uncondensed* one, i.e. the frozen-coefficient block: it omits the closure path,
// which is exactly what the rank-1 condensation in the forward smoother encodes, so there
// is nothing to condense.
//
// That is a deliberate choice, not a missing piece.  The exact closure and d(eta_bar)/du
// paths DO exist -- they are in the VJP (see the note at the top of vjp_body in
// residuals.cu), which is what defines the operator being solved.  The smoother only has
// to PRECONDITION that operator, and the frozen block does so well: the adjoint V-cycles
// converge to 1.0e-6, the same as SSA, whose VJP likewise carries d(eta)/du while its
// smoother does not.  Making the smoother exact would mean transposing the condensed block,
// (A - b c^T/d)^T, i.e. b and c swapping roles before the transpose below -- available if
// convergence ever demands it, but it does not.
template <bool DIVA>
__device__ void vanka_smooth_adjoint_body(
    float* __restrict__ lambda_u_out,
    float* __restrict__ lambda_v_out,
    float* __restrict__ lambda_H_out,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ H,
    const float* __restrict__ phi,
    const float* __restrict__ mask,
    const float* __restrict__ r_adj_u,  
    const float* __restrict__ r_adj_v,
    const float* __restrict__ r_adj_H,
    const float* __restrict__ bed,
    const float* __restrict__ B,
    const float* __restrict__ beta,
    const float* __restrict__ u_c,
    const float* __restrict__ gamma,
    const float* __restrict__ eta_bar,     // DIVA only
    const float* __restrict__ beta_eff,    // DIVA only
    float n, float eps_reg, float flotation_reg_driving,
    float m, float u_reg, float water_drag, float flotation_reg_sliding, float sliding_law,
    float calving_rate, float flotation_reg_calving,
    float dx, float dt,
    int ny, int nx, int stride, int halo,
    float ssa_damping, float mc_damping
    ) 
{
    const int bny = 16;
    const int bnx = 16;

    int bi = threadIdx.y;
    int bj = threadIdx.x;

    int j = blockIdx.x * stride + (threadIdx.x - halo);
    int i = blockIdx.y * stride + (threadIdx.y - halo);

    __shared__ float eta_local[bny][bnx];

    if (i < 0 || i >= ny || j<0 || j >= nx) return;

    if (DIVA) {
	eta_local[bi][bj] = get_cell(eta_bar, i, j, ny, nx);
    } else {
	populate_viscosity(eta_local, bi, bj, i, j, u, v, B, n, eps_reg, dx, ny, nx);
    }

    __syncthreads();

    bool is_active = (threadIdx.x >= halo && threadIdx.x < blockDim.x - halo) &&
                     (threadIdx.y >= halo && threadIdx.y < blockDim.y - halo);

    if ( is_active ) {
	float dx_inv = 1.0f/dx;

	float masked = get_cell(mask, i, j, ny, nx);

	float u_l = get_vfacet(u, i, j, ny, nx);
	float u_r = get_vfacet(u, i, j + 1, ny, nx);
	float v_t = get_hfacet(v, i, j, ny, nx);
	float v_b = get_hfacet(v, i + 1, j, ny, nx);
	float H_c = get_cell(H, i, j, ny, nx);

	float J[25] = {0};
	float rhs[5] = {0};
	// Note that the adjoint assembles a forward problem rhs, but it's 
	// discarded.  
	float dr_dbeta_eff[5] = {0};        // populated for DIVA but unused: no condensation here
	build_5x5_vanka<DIVA>(J, rhs, dr_dbeta_eff,
		u_l, u_r, v_t, v_b, H_c,
		u, v, H, eta_local, phi,
		bed, B, beta, u_c, beta_eff, gamma,
		n, eps_reg, flotation_reg_driving,
		m, u_reg, water_drag, flotation_reg_sliding, sliding_law,
		calving_rate, flotation_reg_calving,
		dx, dt, ny, nx, i, j, bi, bj);

	J[0]  -= ssa_damping;
        J[6]  -= ssa_damping;
        J[12] -= ssa_damping;
        J[18] -= ssa_damping;
        J[24] += mc_damping;

        rhs[0] = get_vfacet(r_adj_u, i, j, ny, nx);
        rhs[1] = get_vfacet(r_adj_u, i, j+1, ny, nx);
        rhs[2] = get_hfacet(r_adj_v, i, j, ny, nx);
        rhs[3] = get_hfacet(r_adj_v, i+1, j, ny, nx);
        rhs[4] = get_cell(r_adj_H, i, j, ny, nx);

	if (j == 0) {
	    for(int k=0; k<5; ++k) J[0 + k] = 0.0f;
	    for(int k=0; k<5; ++k) J[k*5 + 0] = 0.0f;
	    J[0] = 1.0f;
	    rhs[0] = 0.0f;
	}

	if (j == (nx - 1)) {
	    for(int k=0; k<5; ++k) J[5 + k] = 0.0f;
	    for(int k=0; k<5; ++k) J[k*5 + 1] = 0.0f;
	    J[6] = 1.0f;
	    rhs[1] = 0.0f;
	}

	if (i == 0) {
	    for(int k=0; k<5; ++k) J[10 + k] = 0.0f;
	    for(int k=0; k<5; ++k) J[k*5 + 2] = 0.0f;
	    J[12] = 1.0f;
	    rhs[2] = 0.0f;
	}

	if (i == (ny-1)) {
	    for(int k=0; k<5; ++k) J[15 + k] = 0.0f;
	    for(int k=0; k<5; ++k) J[k*5 + 3] = 0.0f;
	    J[18] = 1.0f;
	    rhs[3] = 0.0f;
	}
	
	if (masked > 0.5) {
	    // Active set constraint: Force H = thklim
	    for(int k=0; k<5; ++k) J[20 + k] = 0.0f;
	    for(int k=0; k<5; ++k) J[k*5 + 4] = 0.0f;
	    J[24] = 1.0f;
	    rhs[4] = 0.0f;
	} 

        // The adjoint block is the literal transpose of the forward block.  Forming it
        // explicitly (rather than assuming symmetry) is what lets the same lu_5x5_solve
        // serve both, and it keeps the smoother honest for the terms that are genuinely
        // asymmetric -- the upwind flux and the mass row above all.
        float J_T[25];
        #pragma unroll
        for(int r=0; r<5; ++r) {
            #pragma unroll
            for(int c=0; c<5; ++c) {
                J_T[r*5 + c] = J[c*5 + r];
            }
        }
 	
	float delta_lambda[5] = {0};
	lu_5x5_solve(J_T,rhs,delta_lambda);

	atomicAdd(&lambda_u_out[i * (nx + 1) + j],      0.5f*delta_lambda[0]);
	atomicAdd(&lambda_u_out[i * (nx + 1) + j + 1],  0.5f*delta_lambda[1]);
	atomicAdd(&lambda_v_out[i * nx + j],            0.5f*delta_lambda[2]);
	atomicAdd(&lambda_v_out[(i + 1) * nx + j ],     0.5f*delta_lambda[3]);
	atomicAdd(&lambda_H_out[i * nx + j],                 delta_lambda[4]);
    }
}


extern "C" __global__
void vanka_smooth_adjoint(
    float* __restrict__ lambda_u_out,
    float* __restrict__ lambda_v_out,
    float* __restrict__ lambda_H_out,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ H,
    const float* __restrict__ phi,
    const float* __restrict__ mask,
    const float* __restrict__ r_adj_u,  
    const float* __restrict__ r_adj_v,
    const float* __restrict__ r_adj_H,
    const float* __restrict__ bed,
    const float* __restrict__ B,
    const float* __restrict__ beta,
    const float* __restrict__ u_c,
    const float* __restrict__ gamma,
    float n, float eps_reg, float flotation_reg_driving,
    float m, float u_reg, float water_drag, float flotation_reg_sliding, float sliding_law,
    float calving_rate, float flotation_reg_calving,
    float dx, float dt,
    int ny, int nx, int stride, int halo,
    float ssa_damping, float mc_damping
    ) {
    vanka_smooth_adjoint_body<false>(lambda_u_out,lambda_v_out,lambda_H_out,u,v,H,phi,mask,
	    r_adj_u,r_adj_v,r_adj_H,bed,B,beta,u_c,gamma,nullptr,nullptr,
	    n,eps_reg,flotation_reg_driving,
	    m,u_reg,water_drag,flotation_reg_sliding,sliding_law,
	    calving_rate,flotation_reg_calving,dx,dt,ny,nx,stride,halo,
	    ssa_damping,mc_damping);
}

extern "C" __global__
void vanka_smooth_adjoint_diva(
    float* __restrict__ lambda_u_out,
    float* __restrict__ lambda_v_out,
    float* __restrict__ lambda_H_out,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ H,
    const float* __restrict__ phi,
    const float* __restrict__ mask,
    const float* __restrict__ r_adj_u,  
    const float* __restrict__ r_adj_v,
    const float* __restrict__ r_adj_H,
    const float* __restrict__ bed,
    const float* __restrict__ B,
    const float* __restrict__ beta,
    const float* __restrict__ u_c,
    const float* __restrict__ gamma,
    const float* __restrict__ eta_bar,     // DIVA only
    const float* __restrict__ beta_eff,    // DIVA only
    float n, float eps_reg, float flotation_reg_driving,
    float m, float u_reg, float water_drag, float flotation_reg_sliding, float sliding_law,
    float calving_rate, float flotation_reg_calving,
    float dx, float dt,
    int ny, int nx, int stride, int halo,
    float ssa_damping, float mc_damping
    ) {
    vanka_smooth_adjoint_body<true>(lambda_u_out,lambda_v_out,lambda_H_out,u,v,H,phi,mask,
	    r_adj_u,r_adj_v,r_adj_H,bed,B,beta,u_c,gamma,eta_bar,beta_eff,
	    n,eps_reg,flotation_reg_driving,
	    m,u_reg,water_drag,flotation_reg_sliding,sliding_law,
	    calving_rate,flotation_reg_calving,dx,dt,ny,nx,stride,halo,
	    ssa_damping,mc_damping);
}

// Debug/verification hook: assemble the local blocks and copy them out to host arrays
// without smoothing.  Used to compare an extracted J against an independently formed one.
extern "C" __global__
void vanka_dump(
    float* __restrict__ J_array,
    float* __restrict__ r_array,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ H,
    const float* __restrict__ phi,
    const float* __restrict__ f_u,
    const float* __restrict__ f_v,
    const float* __restrict__ f_H,
    const float* __restrict__ bed,
    const float* __restrict__ B,
    const float* __restrict__ beta,
    const float* __restrict__ u_c,
    const float* __restrict__ gamma,
    float n, float eps_reg, float flotation_reg_driving,
    float m, float u_reg, float water_drag, float flotation_reg_sliding, float sliding_law,
    float calving_rate, float flotation_reg_calving,
    float dx, float dt,
    int ny, int nx, int stride, int halo) 
{
    const int bny = 16;
    const int bnx = 16;

    int bi = threadIdx.y;
    int bj = threadIdx.x;

    int j = blockIdx.x * stride + (threadIdx.x - halo);
    int i = blockIdx.y * stride + (threadIdx.y - halo);

    __shared__ float eta_local[bny][bnx];

    if (i < 0 || i >= ny || j<0 || j >= nx) return;

    populate_viscosity(eta_local, bi, bj, i, j, u, v, B, n, eps_reg, dx, ny, nx);
    __syncthreads();

    bool is_active = (threadIdx.x >= halo && threadIdx.x < blockDim.x - halo) &&
                     (threadIdx.y >= halo && threadIdx.y < blockDim.y - halo);

    if ( is_active ) {
	float dx_inv = 1.0f/dx;

	float u_l = get_vfacet(u, i, j, ny, nx);
	float u_r = get_vfacet(u, i, j + 1, ny, nx);
	float v_t = get_hfacet(v, i, j, ny, nx);
	float v_b = get_hfacet(v, i + 1, j, ny, nx);
	float H_c = get_cell(H, i, j, ny, nx);
	float thklim = get_cell(gamma,i,j,ny,nx);

	float J[25] = {0};
        float r[5] = {0};

        build_5x5_vanka<false>(J, r, nullptr,
	    u_l, u_r, v_t, v_b, H_c,
	    u, v, H, eta_local, phi,
	    bed, B, beta, u_c, nullptr, gamma,
	    n, eps_reg, flotation_reg_driving,
	    m, u_reg, water_drag, flotation_reg_sliding, sliding_law,
	    calving_rate, flotation_reg_calving,
	    dx, dt, ny, nx, i, j, bi, bj);
	
	r[0] -= get_vfacet(f_u,i,j,ny,nx);
	r[1] -= get_vfacet(f_u,i,j+1,ny,nx);
	r[2] -= get_hfacet(f_v,i,j,ny,nx);
	r[3] -= get_hfacet(f_v,i+1,j,ny,nx);
	r[4] -= get_hfacet(f_H,i,j,ny,nx);

	if (j == 0) {
	    for(int k=0; k<5; ++k) J[0 + k] = 0.0f;
	    for(int k=0; k<5; ++k) J[k*5 + 0] = 0.0f;
	    J[0] = 1.0f;
	    r[0] = u_l;
	}

	if (j == (nx - 1)) {
	    for(int k=0; k<5; ++k) J[5 + k] = 0.0f;
	    for(int k=0; k<5; ++k) J[k*5 + 1] = 0.0f;
	    J[6] = 1.0f;
	    r[1] = u_r;
	}

	if (i == 0) {
	    for(int k=0; k<5; ++k) J[10 + k] = 0.0f;
	    for(int k=0; k<5; ++k) J[k*5 + 2] = 0.0f;
	    J[12] = 1.0f;
	    r[2] = v_t;
	}

	if (i == (ny-1)) {
	    for(int k=0; k<5; ++k) J[15 + k] = 0.0f;
	    for(int k=0; k<5; ++k) J[k*5 + 3] = 0.0f;
	    J[18] = 1.0f;
	    r[3] = v_b;
	}
	

	if ((H_c - dt*r[4]) <= (thklim)) {
	    // Active set constraint: Force H = thklim
	    for(int k=0; k<5; ++k) J[20 + k] = 0.0f;
	    for(int k=0; k<5; ++k) J[k*5 + 4] = 0.0f;
	    J[24] = 1.0f;
	    r[4] = H_c - thklim;
	}

        for(int k=0; k<25; ++k) J_array[25*(i * nx + j) + k] = J[k]; 
        for(int k=0; k<5; ++k) r_array[5*(i * nx + j) + k] = r[k]; 
    }
}

