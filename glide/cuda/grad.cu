extern "C" __global__
void compute_gradient_beta(
    float* __restrict__ grad_beta,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ H,
    const float* __restrict__ lambda_u,
    const float* __restrict__ lambda_v,
    const float* __restrict__ lambda_H,
    const float* __restrict__ phi,
    const float* __restrict__ mask,
    const float* __restrict__ bed,
    const float* __restrict__ B,
    const float* __restrict__ beta,
    const float* __restrict__ gamma,
    float n, float eps_reg, float flotation_reg_driving,
    float m, float u_reg, float water_drag, float flotation_reg_sliding,     
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

    bool is_active = (threadIdx.x >= halo && threadIdx.x < blockDim.x - halo) &&
                     (threadIdx.y >= halo && threadIdx.y < blockDim.y - halo);

    bool has_u    = i >= 0 && i <  ny && j >= 0 && j <= nx;
    bool has_v    = i >= 0 && i <= ny && j >= 0 && j <  nx;


    if ( is_active ) {

	// Residual for the u-momentum equation on the left side of the cell
	// the right side residual is handled by the next cell to the right!
	
	if (has_u){

            float u_l    = get_vfacet(u,i,j,ny,nx);
            float v_tl   = get_hfacet(v,i,j-1,ny,nx);
	    float v_tr   = get_hfacet(v,i,j,ny,nx);
	    float v_bl   = get_hfacet(v,i+1,j-1,ny,nx);
	    float v_br   = get_hfacet(v,i+1,j,ny,nx);

	    float H_l    = get_cell(H,i,j-1,ny,nx);
	    float H_c    = get_cell(H,i,j,ny,nx);
	    float phi_l  = get_cell(phi,i,j-1,ny,nx);
	    float phi_c  = get_cell(phi,i,j,ny,nx);
	    float beta_l = get_cell(beta,i,j-1,ny,nx);
	    float beta_c = get_cell(beta,i,j,ny,nx);
	    TauBxJacobian j_tau_bx = get_tau_bx_jac({u_l,v_tl,v_tr,v_bl,v_br,H_l,H_c,phi_l,phi_c,beta_l,beta_c,m,u_reg,water_drag,flotation_reg_sliding});

	    float lambda_u_l = get_vfacet(lambda_u,i,j,ny,nx);

	    if (j>0     )  {atomicAdd(&grad_beta[i * nx + j - 1],lambda_u_l * j_tau_bx.d_beta_l);}
	    if (j<(nx-1))  {atomicAdd(&grad_beta[i * nx + j]    ,lambda_u_l * j_tau_bx.d_beta_r);}
 	}

	if (has_v){

	    float v_t  = get_hfacet(v,i,j,ny,nx);
            float u_tl = get_vfacet(u,i-1,j,ny,nx);
            float u_tr = get_vfacet(u,i-1,j+1,ny,nx);
            float u_bl = get_vfacet(u,i,j,ny,nx);
            float u_br = get_vfacet(u,i,j+1,ny,nx);

	    float H_t    = get_cell(H,i-1,j,ny,nx);
	    float H_c    = get_cell(H,i,j,ny,nx);
	    float phi_t  = get_cell(phi,i-1,j,ny,nx);
	    float phi_c  = get_cell(phi,i,j,ny,nx);
	    float beta_t = get_cell(beta,i-1,j,ny,nx);
	    float beta_c = get_cell(beta,i,j,ny,nx);

	    TauByJacobian j_tau_by = get_tau_by_jac({v_t,u_tl,u_tr,u_bl,u_br,H_t,H_c,phi_t,phi_c,beta_t,beta_c,m,u_reg,water_drag,flotation_reg_sliding});
	    
	    float lambda_v_t = get_hfacet(lambda_v,i,j,ny,nx);
	    
	    if (i>0     ) {atomicAdd(&grad_beta[(i-1) * nx + j],lambda_v_t * j_tau_by.d_beta_t);}
	    if (i<(ny-1)) {atomicAdd(&grad_beta[i * nx + j]    ,lambda_v_t * j_tau_by.d_beta_b);}
	}
    }
}

extern "C" __global__
void compute_gradient_bed(
    float* __restrict__ grad_bed,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ H,
    const float* __restrict__ lambda_u,
    const float* __restrict__ lambda_v,
    const float* __restrict__ lambda_H,
    const float* __restrict__ phi,
    const float* __restrict__ mask,
    const float* __restrict__ bed,
    const float* __restrict__ B,
    const float* __restrict__ beta,
    const float* __restrict__ gamma,
    float n, float eps_reg, float flotation_reg_driving,
    float m, float u_reg, float water_drag, float flotation_reg_sliding,     
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

    bool is_active = (threadIdx.x >= halo && threadIdx.x < blockDim.x - halo) &&
                     (threadIdx.y >= halo && threadIdx.y < blockDim.y - halo);

    bool has_u    = i >= 0 && i <  ny && j >= 0 && j <= nx;
    bool has_v    = i >= 0 && i <= ny && j >= 0 && j <  nx;
    
    float dx_inv = 1.0f / dx;

    if ( is_active ) {

	// Residual for the u-momentum equation on the left side of the cell
	// the right side residual is handled by the next cell to the right!
	
	if (has_u){
	    {
	    float H_l    = get_cell(H,i,j-1,ny,nx);
	    float H_c    = get_cell(H,i,j,ny,nx);
	    
	    float bed_l  = get_cell(bed,i,j-1,ny,nx);
	    float bed_c  = get_cell(bed,i,j,ny,nx);
	    float phi_l  = get_cell(phi,i,j-1,ny,nx);
	    float phi_c  = get_cell(phi,i,j,ny,nx);
	    TauDxJacobian j_tau_dx = get_tau_dx_jac({H_l,H_c,bed_l,bed_c,phi_l,phi_c,flotation_reg_driving},dx_inv,i,j,ny,nx);

            float lambda_u_l    = get_vfacet(lambda_u,i,j,ny,nx);
	    
	    if (j>0     )  {atomicAdd(&grad_bed[i * nx + j - 1],-lambda_u_l * j_tau_dx.d_bed_l);}
	    if (j<(nx-1))  {atomicAdd(&grad_bed[i * nx + j]    ,-lambda_u_l * j_tau_dx.d_bed_r);}
	    }
 	}

	if (has_v){
	    {
	    float H_t    = get_cell(H,i-1,j,ny,nx);
	    float H_c    = get_cell(H,i,j,ny,nx);
	    float bed_t = get_cell(bed,i-1,j,ny,nx);
	    float bed_c = get_cell(bed,i,j,ny,nx);
	    float phi_t  = get_cell(phi,i-1,j,ny,nx);
	    float phi_c  = get_cell(phi,i,j,ny,nx);

	    TauDyJacobian j_tau_dy = get_tau_dy_jac({H_t,H_c,bed_t,bed_c,phi_t,phi_c,flotation_reg_driving},dx_inv,i,j,ny,nx);
            
	    float lambda_v_t    = get_hfacet(lambda_v,i,j,ny,nx);
	    
	    if (i>0     ) {atomicAdd(&grad_bed[(i-1) * nx + j],-lambda_v_t * j_tau_dy.d_bed_t);}
	    if (i<(ny-1)) {atomicAdd(&grad_bed[i * nx + j]    ,-lambda_v_t * j_tau_dy.d_bed_b);}
	    }	    
	}
    }
}

/*=========================================================
  ============ Rheology (B) gradient ======================
  =========================================================
  dJ/dB via the discrete adjoint. The ice-stiffness B enters the
  momentum residual ONLY through the cell viscosity
      eta_ij = 0.5 * B_ij * eps_II_ij^((1-n)/2n),
  which is LINEAR and CELL-LOCAL in B, so  d(eta_ij)/d(B_ij) = eta_ij / B_ij.

  The viscous stress divergence in compute_residual is assembled from
  cell-centered normal stresses (sigma_xx, sigma_yy) and vertex-centered
  shear stress (sigma_xy), each proportional to eta_H = eta * H. Every
  stress Jacobian already exposes d(sigma)/d(eta_H), and the eta_H
  Jacobians expose d(eta_H)/d(eta) (= H for a cell, 0.25*H per corner for
  a vertex). Chaining:
      dJ/dB_k = sum_facets  lambda_facet * dR_facet/d(eta_k) * (eta_k / B_k),
  which this kernel evaluates by mirroring the sigma terms of
  compute_residual EXACTLY (same stencil reads, same signs) and scattering
  each contribution into grad_B. Non-viscous terms (basal drag, driving
  stress, flux, calving) do not depend on B and are omitted.

  Convention matches compute_gradient_beta: grad += lambda * dR/dparam,
  using the residual's own +/- signs (sigma_xx_c: +dx_inv, sigma_xx_l:
  -dx_inv, etc.).
*/
__device__ __forceinline__
void scatter_grad_B(float* __restrict__ grad_B, int i2, int j2,
                    int ny, int nx, float val) {
    if (i2 >= 0 && i2 < ny && j2 >= 0 && j2 < nx) {
        atomicAdd(&grad_B[i2 * nx + j2], val);
    }
}

// Contribution of one cell k to grad_B from a stress term:
//   lambda * coef * (dSigma/dEtaH) * (dEtaH/dEta_k) * (dEta_k/dB_k),
// with dEta_k/dB_k = eta_k / B_k.  coef carries the residual's +/-dx_inv.
__device__ __forceinline__
float gradB_term(float lam, float coef, float d_sigma_d_etaH,
                 float d_etaH_d_eta, float eta_k, float B_k) {
    return lam * coef * d_sigma_d_etaH * d_etaH_d_eta * (eta_k / B_k);
}

extern "C" __global__
void compute_gradient_B(
    float* __restrict__ grad_B,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ H,
    const float* __restrict__ lambda_u,
    const float* __restrict__ lambda_v,
    const float* __restrict__ lambda_H,
    const float* __restrict__ phi,
    const float* __restrict__ mask,
    const float* __restrict__ bed,
    const float* __restrict__ B,
    const float* __restrict__ beta,
    const float* __restrict__ gamma,
    float n, float eps_reg, float flotation_reg_driving,
    float m, float u_reg, float water_drag, float flotation_reg_sliding,
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

    if (i > ny || j > nx) return;

    // Same viscosity tile as compute_residual (fills the full 16x16 incl. halo).
    populate_viscosity(eta_local, bi, bj, i, j, u, v, B, n, eps_reg, dx, ny, nx);

    __syncthreads();

    bool is_active = (threadIdx.x >= halo && threadIdx.x < blockDim.x - halo) &&
                     (threadIdx.y >= halo && threadIdx.y < blockDim.y - halo);

    if (!is_active) return;

    float dx_inv = 1.0f / dx;
    bool has_u = i >= 0 && i <  ny && j >= 0 && j <= nx;
    bool has_v = i >= 0 && i <= ny && j >= 0 && j <  nx;

    /* ---- u-momentum facet residual ru_l at (i,j) ----
       j==0 and j==nx are Dirichlet (ru_l = u), no B dependence -> skip. */
    if (has_u && j > 0 && j < nx) {
        float lam = get_vfacet(lambda_u, i, j, ny, nx);

        // (1) sigma_xx at cell c=(i,j):  ru_l += sigma_xx_c * dx_inv
        {
            float eta_c = eta_local[bi][bj];
            float H_c   = get_cell(H, i, j, ny, nx);
            EtaHCellJacobian eta_H_c = get_eta_H_cell_jac({eta_c, H_c});
            float u_l = get_vfacet(u, i, j, ny, nx);
            float u_r = get_vfacet(u, i, j + 1, ny, nx);
            float v_t = get_hfacet(v, i, j, ny, nx);
            float v_b = get_hfacet(v, i + 1, j, ny, nx);
            SigmaNormalJacobian s = get_sigma_xx_jac({u_l, u_r, v_t, v_b, eta_H_c.res}, dx_inv, i, j, ny, nx);
            scatter_grad_B(grad_B, i, j, ny, nx,
                gradB_term(lam, dx_inv, s.d_eta_H, eta_H_c.d_eta, eta_c, get_cell(B, i, j, ny, nx)));
        }

        // (2) sigma_xx at cell l=(i,j-1):  ru_l -= sigma_xx_l * dx_inv
        {
            float eta_l = eta_local[bi][bj - 1];
            float H_l   = get_cell(H, i, j - 1, ny, nx);
            EtaHCellJacobian eta_H_l = get_eta_H_cell_jac({eta_l, H_l});
            float u_l  = get_vfacet(u, i, j, ny, nx);
            float u_ll = get_vfacet(u, i, j - 1, ny, nx);
            float v_lt = get_hfacet(v, i, j - 1, ny, nx);
            float v_lb = get_hfacet(v, i + 1, j - 1, ny, nx);
            SigmaNormalJacobian s = get_sigma_xx_jac({u_ll, u_l, v_lt, v_lb, eta_H_l.res}, dx_inv, i, j - 1, ny, nx);
            scatter_grad_B(grad_B, i, j - 1, ny, nx,
                gradB_term(lam, -dx_inv, s.d_eta_H, eta_H_l.d_eta, eta_l, get_cell(B, i, j - 1, ny, nx)));
        }

        // (3) sigma_xy at top-left vertex:  ru_l += sigma_xy_tl * dx_inv
        //     corners (tl,t,l,c) = (i-1,j-1),(i-1,j),(i,j-1),(i,j)
        {
            float eta_tl = eta_local[bi - 1][bj - 1];
            float eta_t  = eta_local[bi - 1][bj];
            float eta_l  = eta_local[bi][bj - 1];
            float eta_c  = eta_local[bi][bj];
            float H_tl = get_cell(H, i - 1, j - 1, ny, nx);
            float H_t  = get_cell(H, i - 1, j, ny, nx);
            float H_l  = get_cell(H, i, j - 1, ny, nx);
            float H_c  = get_cell(H, i, j, ny, nx);
            EtaHVertexJacobian eh = get_eta_H_vertex_jac({eta_tl, eta_t, eta_l, eta_c, H_tl, H_t, H_l, H_c});
            float u_tl = get_vfacet(u, i - 1, j, ny, nx);
            float u_l  = get_vfacet(u, i, j, ny, nx);
            float v_lt = get_hfacet(v, i, j - 1, ny, nx);
            float v_t  = get_hfacet(v, i, j, ny, nx);
            SigmaShearJacobian s = get_sigma_xy_jac({u_tl, u_l, v_lt, v_t, eh.res}, dx_inv, i, j, ny, nx);
            scatter_grad_B(grad_B, i - 1, j - 1, ny, nx, gradB_term(lam, dx_inv, s.d_eta_H, eh.d_eta_tl, eta_tl, get_cell(B, i - 1, j - 1, ny, nx)));
            scatter_grad_B(grad_B, i - 1, j,     ny, nx, gradB_term(lam, dx_inv, s.d_eta_H, eh.d_eta_tr, eta_t,  get_cell(B, i - 1, j, ny, nx)));
            scatter_grad_B(grad_B, i,     j - 1, ny, nx, gradB_term(lam, dx_inv, s.d_eta_H, eh.d_eta_bl, eta_l,  get_cell(B, i, j - 1, ny, nx)));
            scatter_grad_B(grad_B, i,     j,     ny, nx, gradB_term(lam, dx_inv, s.d_eta_H, eh.d_eta_br, eta_c,  get_cell(B, i, j, ny, nx)));
        }

        // (4) sigma_xy at bottom-left vertex:  ru_l -= sigma_xy_bl * dx_inv
        //     corners (l,c,bl,b) = (i,j-1),(i,j),(i+1,j-1),(i+1,j)
        {
            float eta_l  = eta_local[bi][bj - 1];
            float eta_c  = eta_local[bi][bj];
            float eta_bl = eta_local[bi + 1][bj - 1];
            float eta_b  = eta_local[bi + 1][bj];
            float H_l  = get_cell(H, i, j - 1, ny, nx);
            float H_c  = get_cell(H, i, j, ny, nx);
            float H_bl = get_cell(H, i + 1, j - 1, ny, nx);
            float H_b  = get_cell(H, i + 1, j, ny, nx);
            EtaHVertexJacobian eh = get_eta_H_vertex_jac({eta_l, eta_c, eta_bl, eta_b, H_l, H_c, H_bl, H_b});
            float u_l  = get_vfacet(u, i, j, ny, nx);
            float u_bl = get_vfacet(u, i + 1, j, ny, nx);
            float v_lb = get_hfacet(v, i + 1, j - 1, ny, nx);
            float v_b  = get_hfacet(v, i + 1, j, ny, nx);
            SigmaShearJacobian s = get_sigma_xy_jac({u_l, u_bl, v_lb, v_b, eh.res}, dx_inv, i + 1, j, ny, nx);
            scatter_grad_B(grad_B, i,     j - 1, ny, nx, gradB_term(lam, -dx_inv, s.d_eta_H, eh.d_eta_tl, eta_l,  get_cell(B, i, j - 1, ny, nx)));
            scatter_grad_B(grad_B, i,     j,     ny, nx, gradB_term(lam, -dx_inv, s.d_eta_H, eh.d_eta_tr, eta_c,  get_cell(B, i, j, ny, nx)));
            scatter_grad_B(grad_B, i + 1, j - 1, ny, nx, gradB_term(lam, -dx_inv, s.d_eta_H, eh.d_eta_bl, eta_bl, get_cell(B, i + 1, j - 1, ny, nx)));
            scatter_grad_B(grad_B, i + 1, j,     ny, nx, gradB_term(lam, -dx_inv, s.d_eta_H, eh.d_eta_br, eta_b,  get_cell(B, i + 1, j, ny, nx)));
        }
    }

    /* ---- v-momentum facet residual rv_t at (i,j) ----
       i==0 and i==ny are Dirichlet (rv_t = v), no B dependence -> skip. */
    if (has_v && i > 0 && i < ny) {
        float lam = get_hfacet(lambda_v, i, j, ny, nx);

        // (5) sigma_yy at cell t=(i-1,j):  rv_t += sigma_yy_t * dx_inv
        {
            float eta_t = eta_local[bi - 1][bj];
            float H_t   = get_cell(H, i - 1, j, ny, nx);
            EtaHCellJacobian eta_H_t = get_eta_H_cell_jac({eta_t, H_t});
            float u_tl = get_vfacet(u, i - 1, j, ny, nx);
            float u_tr = get_vfacet(u, i - 1, j + 1, ny, nx);
            float v_tt = get_hfacet(v, i - 1, j, ny, nx);
            float v_t  = get_hfacet(v, i, j, ny, nx);
            SigmaNormalJacobian s = get_sigma_yy_jac({u_tl, u_tr, v_tt, v_t, eta_H_t.res}, dx_inv, i - 1, j, ny, nx);
            scatter_grad_B(grad_B, i - 1, j, ny, nx,
                gradB_term(lam, dx_inv, s.d_eta_H, eta_H_t.d_eta, eta_t, get_cell(B, i - 1, j, ny, nx)));
        }

        // (6) sigma_yy at cell c=(i,j):  rv_t -= sigma_yy_c * dx_inv
        {
            float eta_c = eta_local[bi][bj];
            float H_c   = get_cell(H, i, j, ny, nx);
            EtaHCellJacobian eta_H_c = get_eta_H_cell_jac({eta_c, H_c});
            float u_l = get_vfacet(u, i, j, ny, nx);
            float u_r = get_vfacet(u, i, j + 1, ny, nx);
            float v_t = get_hfacet(v, i, j, ny, nx);
            float v_b = get_hfacet(v, i + 1, j, ny, nx);
            SigmaNormalJacobian s = get_sigma_yy_jac({u_l, u_r, v_t, v_b, eta_H_c.res}, dx_inv, i, j, ny, nx);
            scatter_grad_B(grad_B, i, j, ny, nx,
                gradB_term(lam, -dx_inv, s.d_eta_H, eta_H_c.d_eta, eta_c, get_cell(B, i, j, ny, nx)));
        }

        // (7) sigma_xy at top-left vertex:  rv_t -= sigma_xy_tl * dx_inv
        //     corners (tl,t,l,c) = (i-1,j-1),(i-1,j),(i,j-1),(i,j)
        {
            float eta_tl = eta_local[bi - 1][bj - 1];
            float eta_t  = eta_local[bi - 1][bj];
            float eta_l  = eta_local[bi][bj - 1];
            float eta_c  = eta_local[bi][bj];
            float H_tl = get_cell(H, i - 1, j - 1, ny, nx);
            float H_t  = get_cell(H, i - 1, j, ny, nx);
            float H_l  = get_cell(H, i, j - 1, ny, nx);
            float H_c  = get_cell(H, i, j, ny, nx);
            EtaHVertexJacobian eh = get_eta_H_vertex_jac({eta_tl, eta_t, eta_l, eta_c, H_tl, H_t, H_l, H_c});
            float u_tl = get_vfacet(u, i - 1, j, ny, nx);
            float u_l  = get_vfacet(u, i, j, ny, nx);
            float v_lt = get_hfacet(v, i, j - 1, ny, nx);
            float v_t  = get_hfacet(v, i, j, ny, nx);
            SigmaShearJacobian s = get_sigma_xy_jac({u_tl, u_l, v_lt, v_t, eh.res}, dx_inv, i, j, ny, nx);
            scatter_grad_B(grad_B, i - 1, j - 1, ny, nx, gradB_term(lam, -dx_inv, s.d_eta_H, eh.d_eta_tl, eta_tl, get_cell(B, i - 1, j - 1, ny, nx)));
            scatter_grad_B(grad_B, i - 1, j,     ny, nx, gradB_term(lam, -dx_inv, s.d_eta_H, eh.d_eta_tr, eta_t,  get_cell(B, i - 1, j, ny, nx)));
            scatter_grad_B(grad_B, i,     j - 1, ny, nx, gradB_term(lam, -dx_inv, s.d_eta_H, eh.d_eta_bl, eta_l,  get_cell(B, i, j - 1, ny, nx)));
            scatter_grad_B(grad_B, i,     j,     ny, nx, gradB_term(lam, -dx_inv, s.d_eta_H, eh.d_eta_br, eta_c,  get_cell(B, i, j, ny, nx)));
        }

        // (8) sigma_xy at top-right vertex:  rv_t += sigma_xy_tr * dx_inv
        //     corners (t,tr,c,r) = (i-1,j),(i-1,j+1),(i,j),(i,j+1)
        {
            float eta_t  = eta_local[bi - 1][bj];
            float eta_tr = eta_local[bi - 1][bj + 1];
            float eta_c  = eta_local[bi][bj];
            float eta_r  = eta_local[bi][bj + 1];
            float H_t  = get_cell(H, i - 1, j, ny, nx);
            float H_tr = get_cell(H, i - 1, j + 1, ny, nx);
            float H_c  = get_cell(H, i, j, ny, nx);
            float H_r  = get_cell(H, i, j + 1, ny, nx);
            EtaHVertexJacobian eh = get_eta_H_vertex_jac({eta_t, eta_tr, eta_c, eta_r, H_t, H_tr, H_c, H_r});
            float u_tr = get_vfacet(u, i - 1, j + 1, ny, nx);
            float u_r  = get_vfacet(u, i, j + 1, ny, nx);
            float v_t  = get_hfacet(v, i, j, ny, nx);
            float v_rt = get_hfacet(v, i, j + 1, ny, nx);
            SigmaShearJacobian s = get_sigma_xy_jac({u_tr, u_r, v_t, v_rt, eh.res}, dx_inv, i, j + 1, ny, nx);
            scatter_grad_B(grad_B, i - 1, j,     ny, nx, gradB_term(lam, dx_inv, s.d_eta_H, eh.d_eta_tl, eta_t,  get_cell(B, i - 1, j, ny, nx)));
            scatter_grad_B(grad_B, i - 1, j + 1, ny, nx, gradB_term(lam, dx_inv, s.d_eta_H, eh.d_eta_tr, eta_tr, get_cell(B, i - 1, j + 1, ny, nx)));
            scatter_grad_B(grad_B, i,     j,     ny, nx, gradB_term(lam, dx_inv, s.d_eta_H, eh.d_eta_bl, eta_c,  get_cell(B, i, j, ny, nx)));
            scatter_grad_B(grad_B, i,     j + 1, ny, nx, gradB_term(lam, dx_inv, s.d_eta_H, eh.d_eta_br, eta_r,  get_cell(B, i, j + 1, ny, nx)));
        }
    }
}
