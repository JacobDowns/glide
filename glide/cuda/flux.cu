/*=========================================================
  ====================== Mass Flux ========================
  =========================================================

  Facet fluxes for the thickness equation, dH/dt + div(u*H) = smb - calving.  One term per
  facet; residual_body differences them to form the divergence.

  The flux is FIRST-ORDER UPWIND, written in the equivalent centred-plus-diffusion form

    q = H_avg*u - 0.5*|u|*(H_downwind - H_upwind)

  which is exact upwinding but stays a single smooth expression.  Advecting thickness with
  a centred flux alone would be unconditionally unstable; the |u| term is the numerical
  diffusion that stabilises it.

  |u| is SMOOTHED to sqrt(u^2 + 10) rather than fabsf(u).  That matters for the adjoint, not
  the forward model: fabsf has a kink at u = 0 where the derivative does not exist, and
  every grounding line and divide has cells sitting near u = 0.  The commented-out fabsf /
  copysignf lines are the unsmoothed version.  The 10 is in (m/a)^2, so the smoothing is
  invisible wherever the ice moves faster than a few m/a.

  Both flux terms return an identically zero Jacobian on the domain boundary, which imposes
  no-flux there.
  =========================================================*/

struct HorizontalFluxStencil {
    float u;
    float H_l, H_r;
};

struct HorizontalFluxStencilDual {
    DualFloat u;
    DualFloat H_l, H_r;

    __device__ __forceinline__
    HorizontalFluxStencil get_primals() const {
        return {u.v,H_l.v,H_r.v};
    }

    __device__ __forceinline__
    HorizontalFluxStencil get_diffs() const {
        return {u.d,H_l.d,H_r.d};
    }
};

struct HorizontalFluxJacobian {
    float res;
    float d_u;
    float d_H_l, d_H_r;

    __device__ __forceinline__
    float apply_jvp(const HorizontalFluxStencil& dot) const {
        return d_u * dot.u +
	       d_H_l * dot.H_l +
	       d_H_r * dot.H_r;
    }

};

__device__
HorizontalFluxJacobian get_horizontal_flux_jac(
    HorizontalFluxStencil s,
    int i, int j,  // Defined on facets
    int ny, int nx
    ) {

    HorizontalFluxJacobian jac = {0};

    // No flux on boundaries
    if (j <= 0 || j >= nx) {
	return jac;
    }

    float H_avg = 0.5f*(s.H_l + s.H_r);
    float u_mag = sqrtf(s.u * s.u + 10.0f);//fabsf(s.u);
    float u_sign = s.u / u_mag;//copysignf(1.0f, s.u);
    //float u_mag = fabsf(s.u);
    //float u_sign = copysignf(1.0f, s.u);
    jac.res = H_avg*s.u - 0.5f*u_mag*(s.H_r - s.H_l);

    jac.d_H_l = 0.5f*(s.u + u_mag);
    jac.d_H_r = 0.5f*(s.u - u_mag);
    jac.d_u   = H_avg - 0.5f*u_sign*(s.H_r - s.H_l);
    return jac;
}

__device__ __forceinline__
DualFloat get_horizontal_flux_dual(
    HorizontalFluxStencilDual s,
    int i, int j,
    int ny, int nx) {
    HorizontalFluxJacobian jac = get_horizontal_flux_jac(s.get_primals(),i,j,ny,nx);
    return {jac.res,jac.apply_jvp(s.get_diffs())};
}


struct VerticalFluxStencil {
    float v;
    float H_t, H_b;
};

struct VerticalFluxStencilDual {
    DualFloat v;
    DualFloat H_t, H_b;

    __device__ __forceinline__
    VerticalFluxStencil get_primals() const {
        return {v.v,H_t.v,H_b.v};
    }

    __device__ __forceinline__
    VerticalFluxStencil get_diffs() const {
        return {v.d,H_t.d,H_b.d};
    }
};

struct VerticalFluxJacobian {
    float res;
    float d_v;
    float d_H_t, d_H_b;

    __device__ __forceinline__
    float apply_jvp(const VerticalFluxStencil& dot) const {
        return d_v * dot.v +
	       d_H_t * dot.H_t +
	       d_H_b * dot.H_b;
    }

};

__device__
VerticalFluxJacobian get_vertical_flux_jac(
    VerticalFluxStencil s,
    int i, int j,  // Defined on facets
    int ny, int nx
    ) {

    VerticalFluxJacobian jac = {0};

    // No flux on boundaries
    if (i <= 0 || i >= ny) {
	return jac;
    }

    float H_avg = 0.5f*(s.H_t + s.H_b);
    float v_mag = sqrtf(s.v * s.v + 10.0f);//fabsf(s.v);
    float v_sign = s.v / v_mag;//copysignf(1.0f, s.v);
    //float v_mag = fabsf(s.v);
    //float v_sign = copysignf(1.0f, s.v);
    jac.res = H_avg*s.v - 0.5f*v_mag*(s.H_t - s.H_b);

    jac.d_H_t = 0.5f*(s.v - v_mag);
    jac.d_H_b = 0.5f*(s.v + v_mag);
    jac.d_v   = H_avg - 0.5f*v_sign*(s.H_t - s.H_b);
    return jac;
}

__device__ __forceinline__
DualFloat get_vertical_flux_dual(
    VerticalFluxStencilDual s,
    int i, int j,
    int ny, int nx) {
    VerticalFluxJacobian jac = get_vertical_flux_jac(s.get_primals(),i,j,ny,nx);
    return {jac.res,jac.apply_jvp(s.get_diffs())};
}

/*==============================================
  ==========  CALVING ==========================
  =============================================

  Two calving formulations, both linear in the local thickness:

    cell   -- a sink over the whole floating cell,  -rate*(1 - phi)*H
    facet  -- a sink at a facet, active only where BOTH sides float, via the product
              (1 - phi_this)*(1 - phi_other)*rate*H_this

  The facet form is the one residual_body uses.  Gating on the product means an interior
  floating cell surrounded by other floating cells still calves on every facet, while a
  facet with grounded ice on either side does not calve at all -- the terminus position
  falls out of the phi field rather than being tracked explicitly.

  NOTE what is NOT differentiated here: phi enters the Dual stencils as a plain float, and
  get_diffs() returns zero for it.  The flotation state is deliberately treated as a FROZEN
  coefficient throughout the derivative path, because it is updated by its own
  under-relaxed outer iteration (compute_grounded in viscosity.cu) rather than solved
  simultaneously.  So d(calving)/d(phi) is absent by design, not by oversight -- but it does
  mean a gradient will not see sensitivity that acts through grounding-line migration.

  H_other and the calving_length / sigmoid_c parameters are carried in the stencils but
  unused by the current expressions; the commented-out lines are smoother gatings that were
  tried and parked.
  =============================================*/

struct CellCalvingStencil {
    float H;
    float grounded;
    float calving_rate;
    float sigmoid_c;
};

struct CellCalvingStencilDual {
    DualFloat H;
    float grounded;
    float calving_rate;
    float sigmoid_c;

    __device__ __forceinline__
    CellCalvingStencil get_primals() const {
        return {H.v,grounded,calving_rate,sigmoid_c};
    }

    __device__ __forceinline__
    CellCalvingStencil get_diffs() const {
        return {H.d,0.0f,0.0f,0.0f};
    }
};

struct CellCalvingJacobian {
    float res;
    float d_H;

    __device__ __forceinline__
    float apply_jvp(const CellCalvingStencil& dot) const {
        return d_H * dot.H;
    }

};

__device__
CellCalvingJacobian get_cell_calving_jac(
    CellCalvingStencil s,
    int i, int j,  // Defined on facets
    int ny, int nx
    ) {

    CellCalvingJacobian jac = {0};

    jac.res = -s.calving_rate*(1.0f - s.grounded)*s.H;
    jac.d_H = -s.calving_rate*(1.0f - s.grounded);

    return jac;
}

__device__ __forceinline__
DualFloat get_cell_calving_dual(
    CellCalvingStencilDual s,
    int i, int j,
    int ny, int nx) {
    CellCalvingJacobian jac = get_cell_calving_jac(s.get_primals(),i,j,ny,nx);
    return {jac.res,jac.apply_jvp(s.get_diffs())};
}

struct FacetCalvingStencil {
    float H_this, H_other;
    float phi_this, phi_other;
    float calving_rate;
    float calving_length;
};

struct FacetCalvingStencilDual {
    DualFloat H_this, H_other;
    float phi_this, phi_other;
    float calving_rate;
    float calving_length;

    __device__ __forceinline__
    FacetCalvingStencil get_primals() const {
        return {H_this.v,H_other.v,phi_this,phi_other,calving_rate,calving_length};
    }

    __device__ __forceinline__
    FacetCalvingStencil get_diffs() const {
        return {H_this.d,H_other.d,0.0f,0.0f,0.0f,0.0f};
    }
};

struct FacetCalvingJacobian {
    float res;
    float d_H_this;

    __device__ __forceinline__
    float apply_jvp(const FacetCalvingStencil& dot) const {
        return d_H_this * dot.H_this;
    }

};

__device__
FacetCalvingJacobian get_facet_calving_jac(
    FacetCalvingStencil s,
    int i, int j,  // Defined on facets
    int ny, int nx
    ) {

    FacetCalvingJacobian jac = {0};


    //float phineg_this = fmaxf(-s.phi_this,0.0f)/s.calving_length;
    //float phineg_other = fmaxf(-s.phi_other,0.0f)/s.calving_length;

    //float phineg2_this = phineg_this * phineg_this;
    //float phineg2_other = phineg_other * phineg_other;

    //float chi_this = phineg2_this/(1.0f + phineg2_this);
    //float chi_other = phineg2_other/(1.0f + phineg2_other);

    //float chi_this = 1.0f - sigmoid(s.phi_this,s.calving_length);
    //float chi_other = 1.0f - sigmoid(s.phi_other,s.calving_length);
    float chi_this = 1.0f - s.phi_this;//sigmoid(s.phi_this,s.calving_length);
    float chi_other = 1.0f - s.phi_other;// sigmoid(s.phi_other,s.calving_length);

    float coeff = chi_this*chi_other;
    jac.res = coeff * s.calving_rate * s.H_this;
    jac.d_H_this = coeff * s.calving_rate;
    
    return jac;
}

__device__ __forceinline__
DualFloat get_facet_calving_dual(
    FacetCalvingStencilDual s,
    int i, int j,
    int ny, int nx) {
    FacetCalvingJacobian jac = get_facet_calving_jac(s.get_primals(),i,j,ny,nx);
    return {jac.res,jac.apply_jvp(s.get_diffs())};
}







