/*==================================================
  ============ GROUNDING AND VISCOSITY =============
  ==================================================

  Two cell-centred scalar fields that the momentum stencils treat as coefficients:

    grounded (phi)  smoothed grounded fraction in [0,1]; multiplies basal drag
    eta             membrane effective viscosity from Glen's flow law

  plus the eta*H products the stress terms actually consume, which live on cells
  (sigma_xx, sigma_yy) and on vertices (sigma_xy).
  ==================================================*/

// Fill a shared tile with the grounded fraction, one value per cell.  Tiled for the same
// reason as eta: the stencils below need several neighbours' worth.
template <int H, int W>
__device__ void populate_grounded(
    float (&grounded_local)[H][W],
    int bi, int bj,
    int i, int j,
    const float* __restrict__ thk,
    const float* __restrict__ bed,
    float sigmoid_c,
    int ny, int nx){

    float H_c = get_cell(thk,i,j,ny,nx);
    float bed_c = get_cell(bed,i,j,ny,nx);
    grounded_local[bi][bj] = get_grounded(H_c,bed_c,sigmoid_c);
}

// Update the stored grounded fraction, UNDER-RELAXED toward the value implied by the
// current geometry.  Grounding-line migration is the stiffest feedback in the model -- a
// cell that ungrounds loses its drag, speeds up and thins, which ungrounds it further -- so
// the update is lagged rather than solved simultaneously with the momentum balance.
// relaxation_parameter = 0 takes the new value outright, 1 freezes the field.
extern "C" __global__
void compute_grounded(
    float* __restrict__ grounded,
    const float* __restrict__ H,
    const float* __restrict__ depth,
    float sigmoid_c,
    float sigmoid_k,
    float relaxation_parameter,
    int ny, int nx,
    int stride, int halo
    )
{
    int j = blockIdx.x * stride + (threadIdx.x - halo);
    int i = blockIdx.y * stride + (threadIdx.y - halo);    

    if (i < 0 || i >= ny || j<0 || j >= nx) return;

    float H_c = get_cell(H,i,j,ny,nx);
    float depth_c = get_cell(depth,i,j,ny,nx);
    float grounded_old = grounded[i * nx + j];
    grounded[i * nx + j] = (1.0f - relaxation_parameter) * get_grounded(H_c,depth_c,sigmoid_c, sigmoid_k) + relaxation_parameter * grounded_old;
}
// Superseded by compute_grounded above, which takes `depth` rather than `bed` and adds the
// sigmoid_k offset.  Kept for reference only.
/*
extern "C" __global__
void compute_phi(
    float* __restrict__ phi,
    const float* __restrict__ H,
    const float* __restrict__ bed,
    float relaxation_parameter,
    int ny, int nx,
    int stride, int halo
    )
{
    int j = blockIdx.x * stride + (threadIdx.x - halo);
    int i = blockIdx.y * stride + (threadIdx.y - halo);    

    if (i < 0 || i >= ny || j<0 || j >= nx) return;

    float H_c = get_cell(H,i,j,ny,nx);
    float bed_c = get_cell(bed,i,j,ny,nx);
    float phi_old = phi[i * nx + j];
    phi[i * nx + j] = (1.0f - relaxation_parameter) * get_phi(H_c,bed_c) + relaxation_parameter * phi_old;
}
*/
/*==================================================
  ================ VISCOSITY =======================
  ==================================================

  Glen's flow law, depth-independent (SSA).  With n the Glen exponent,

    eta = 0.5 * B * (eps_II + eps_reg) ^ ((1-n)/(2n))

  where eps_II is the second invariant of the horizontal strain rate,

    eps_II = (du/dx)^2 + (dv/dy)^2 + (du/dx)(dv/dy) + eps_xy^2

  (the third term comes from incompressibility, dw/dz = -(du/dx + dv/dy), which removes the
  vertical strain rate as an independent unknown).  The exponent is NEGATIVE for n > 1, so
  eta falls as the ice strains faster -- shear thinning.  eps_reg keeps it finite at rest,
  where the true expression is singular.

  The shear term needs du/dy and dv/dx, which live on VERTICES, so it is evaluated at the
  cell's four corners and averaged.  Each corner carries a mask because the accessors clamp
  rather than zero at the domain edge (see common.cu), and a clamped read there would
  fabricate a strain rate out of a repeated value.

  --------------------------------------------------------------------------------------
  The two functions below are OVERLOADS, not duplicates: they differ in the tile's element
  type, and the dual one additionally takes the perturbation arrays (d_u, d_v).  Overload
  resolution picks the float one for residual_body and the Vanka smoothers, and the dual one
  for jvp_body and vjp_body.  Both are live.
  --------------------------------------------------------------------------------------

  This is one of the few places doing GENUINE dual propagation rather than a hand-derived
  Jacobian: the dual overload is the same expression run in dual arithmetic, so d(eta)/du
  falls out without anyone differentiating Glen's law by hand.  The cost is that the
  strain-rate expression now exists in four near-identical copies -- these two, plus the
  float and dual get_membrane_eps_sq in diva.cu, which needs the invariant UNREGULARIZED so
  it can add the vertical shear terms before forming eta.  A change to the discretisation
  has to be made in all four; templating on the scalar type (as diva_coeffs_cell does) would
  collapse them to one.
  ==================================================*/
template <int H, int W>
__device__ void populate_viscosity(
    DualFloat (&eta_local)[H][W],
    int bi, int bj,
    int i, int j,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ d_u,
    const float* __restrict__ d_v,
    const float* __restrict__ B,
    float n, float eps_reg, float dx,
    int ny, int nx){

    float dx_inv = 1.0f/dx;
    float glen_exp = (1.0f - n)/(2.0f * n);

    // Cell viscosity
    DualFloat u_l = get_vfacet(u, d_u, i, j, ny, nx);
    DualFloat u_r = get_vfacet(u, d_u, i, j + 1, ny, nx);
    DualFloat v_t = get_hfacet(v, d_v, i, j, ny, nx);
    DualFloat v_b = get_hfacet(v, d_v, i + 1, j, ny, nx);

    DualFloat dudx = (u_r - u_l)*dx_inv;
    DualFloat dvdy = (v_t - v_b)*dx_inv;

    float tl_mask = i > 0 && j > 0;
    DualFloat u_tl = get_vfacet(u, d_u, i - 1, j, ny, nx);
    DualFloat v_lt = get_hfacet(v, d_v, i, j - 1, ny, nx);
    DualFloat eps_xy_tl = 0.5f*((u_tl - u_l)*dx_inv + (v_t - v_lt)*dx_inv)*tl_mask;

    float tr_mask = i > 0 && j < (nx - 1);
    DualFloat u_tr = get_vfacet(u, d_u, i - 1, j + 1, ny, nx);
    DualFloat v_rt = get_hfacet(v, d_v, i, j + 1, ny, nx);
    DualFloat eps_xy_tr = 0.5f*((u_tr - u_r)*dx_inv + (v_rt - v_t)*dx_inv)*tr_mask;

    float bl_mask = i < (ny - 1) && j > 0;
    DualFloat u_bl = get_vfacet(u, d_u, i + 1, j, ny, nx);
    DualFloat v_lb = get_hfacet(v, d_v, i + 1, j - 1, ny, nx);
    DualFloat eps_xy_bl = 0.5f*((u_l - u_bl)*dx_inv + (v_b - v_lb)*dx_inv)*bl_mask;

    float br_mask = i < (ny - 1) && j < (nx - 1);
    DualFloat u_br = get_vfacet(u, d_u, i + 1, j + 1, ny, nx);
    DualFloat v_rb = get_hfacet(v, d_v, i + 1, j + 1, ny, nx);
    DualFloat eps_xy_br = 0.5f*((u_r - u_br)*dx_inv + (v_rb - v_b)*dx_inv)*br_mask;

    DualFloat eps_xy2_bar = 0.25f*(eps_xy_tl*eps_xy_tl + eps_xy_tr*eps_xy_tr + eps_xy_bl*eps_xy_bl + eps_xy_br*eps_xy_br);

    DualFloat eps_II_c = dudx*dudx + dvdy*dvdy + dudx*dvdy + eps_xy2_bar + eps_reg;

    DualFloat eta = 0.5f*get_cell(B,i,j,ny,nx)*__powf(eps_II_c,glen_exp);

    eta_local[bi][bj] = eta;


}

template <int H, int W>
__device__ void populate_viscosity(
    float (&eta_local)[H][W],
    int bi, int bj,
    int i, int j,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ B,
    float n, float eps_reg, float dx,
    int ny, int nx){

    float dx_inv = 1.0f/dx;
    float glen_exp = (1.0f - n)/(2.0f * n);

    // Cell viscosity
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

    float eps_II_c = dudx*dudx + dvdy*dvdy + dudx*dvdy + eps_xy2_bar + eps_reg;

    eta_local[bi][bj] = 0.5f*get_cell(B,i,j,ny,nx)*__powf(eps_II_c,glen_exp);


}



/*==================================================
  ========== Viscosity-Thickness Product ===========
  ==================================================

  The stress terms need the product eta*H, not eta alone, because the balance is
  depth-INTEGRATED: sigma = H*eta*(strain rate).  It gets its own term so that the
  derivatives with respect to eta and to H stay separate and explicit, which is what lets
  the thickness couple into the momentum rows.

  Two flavours, matching where the stresses live:
    cell    sigma_xx, sigma_yy -- eta*H at the cell centre
    vertex  sigma_xy           -- the average of eta*H over the four cells sharing a vertex

  The Jacobians are trivial by hand (it is a product), which is exactly why these are
  get_*_jac functions rather than dual arithmetic.  See common.cu for the
  Stencil/Jacobian/dual idiom these follow.
  ==================================================*/

struct EtaHCellStencil {
    float eta;
    float H;
};

struct EtaHCellStencilDual{
    DualFloat eta;
    DualFloat H;

    __device__ __forceinline__
    EtaHCellStencil get_primals() const {
        return {eta.v,H.v};
    }

    __device__ __forceinline__
    EtaHCellStencil get_diffs() const {
        return {eta.d,H.d};
    }
};

struct EtaHCellJacobian {
    float res;
    float d_eta;
    float d_H;

    __device__ __forceinline__
    float apply_jvp(const EtaHCellStencil& dot) const {
        return d_eta * dot.eta + d_H * dot.H;

    }
};

__device__ __forceinline__
EtaHCellJacobian get_eta_H_cell_jac(EtaHCellStencil s) {
    EtaHCellJacobian jac;
    jac.res = s.H * s.eta;
    jac.d_eta = s.H;
    jac.d_H = s.eta;

    return jac;
}

__device__ __forceinline__
DualFloat get_eta_H_cell_dual(EtaHCellStencilDual s) {
    EtaHCellJacobian jac = get_eta_H_cell_jac(s.get_primals());
    return {jac.res,jac.apply_jvp(s.get_diffs())};
}

struct EtaHVertexStencil {
    float eta_tl, eta_tr, eta_bl, eta_br;
    float H_tl, H_tr, H_bl, H_br;
};

struct EtaHVertexStencilDual {
    DualFloat eta_tl, eta_tr, eta_bl, eta_br;
    DualFloat H_tl, H_tr, H_bl, H_br;

    __device__ __forceinline__
    EtaHVertexStencil get_primals() const {
        return {eta_tl.v,eta_tr.v,eta_bl.v,eta_br.v,H_tl.v,H_tr.v,H_bl.v,H_br.v};
    }

    __device__ __forceinline__
    EtaHVertexStencil get_diffs() const {
        return {eta_tl.d,eta_tr.d,eta_bl.d,eta_br.d,H_tl.d,H_tr.d,H_bl.d,H_br.d};
    }
};

struct EtaHVertexJacobian {
    float res;
    float d_eta_tl, d_eta_tr, d_eta_bl, d_eta_br;
    float d_H_tl, d_H_tr, d_H_bl, d_H_br;

    __device__ __forceinline__
    float apply_jvp(const EtaHVertexStencil& dot) const {
        return d_eta_tl * dot.eta_tl + d_H_tl * dot.H_tl +
               d_eta_tr * dot.eta_tr + d_H_tr * dot.H_tr +
               d_eta_bl * dot.eta_bl + d_H_bl * dot.H_bl +
               d_eta_br * dot.eta_br + d_H_br * dot.H_br;

    }
};

__device__ __forceinline__
EtaHVertexJacobian get_eta_H_vertex_jac(EtaHVertexStencil s) {
    EtaHVertexJacobian jac;
    jac.res = 0.25f*(s.eta_tl * s.H_tl + s.eta_tr * s.H_tr + s.eta_bl * s.H_bl + s.eta_br * s.H_br);

    jac.d_eta_tl = 0.25f*s.H_tl;
    jac.d_eta_tr = 0.25f*s.H_tr;
    jac.d_eta_bl = 0.25f*s.H_bl;
    jac.d_eta_br = 0.25f*s.H_br;

    jac.d_H_tl = 0.25f*s.eta_tl;
    jac.d_H_tr = 0.25f*s.eta_tr;
    jac.d_H_bl = 0.25f*s.eta_bl;
    jac.d_H_br = 0.25f*s.eta_br;

    return jac;
}

__device__ __forceinline__
DualFloat get_eta_H_vertex_dual(EtaHVertexStencilDual s) {
    EtaHVertexJacobian jac = get_eta_H_vertex_jac(s.get_primals());
    return {jac.res,jac.apply_jvp(s.get_diffs())};
}


