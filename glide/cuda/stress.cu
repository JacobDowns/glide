/*=======================================================
  =============== STRESS TERMS: OVERVIEW ================
  =======================================================

  The four families of term the momentum rows are built from.  residual_body differences
  them; jvp_body and vjp_body contract the same partials forwards and transposed.

    sigma_xx, sigma_yy   membrane normal stress, on CELLS
    sigma_xy             membrane shear stress, on VERTICES
    tau_bx, tau_by       basal drag, on FACETS (SSA and DIVA variants)
    tau_dx, tau_dy       driving stress, on FACETS

  UNITS: rho*g appears NOWHERE in this file, or anywhere else in the kernels.  The whole
  momentum balance is divided through by rho*g, and the caller is expected to have divided
  B and beta by rho*g before handing them in (see how the tests build them).  That is why
  the driving stress below is H*ds/dx with no density or gravity in sight.  Passing a raw,
  un-divided B would produce a silently wrong answer rather than an error.

  SIGN CONVENTION, as assembled in residual_body:

    r_u = d(sigma_xx)/dx + d(sigma_xy)/dy + tau_b - tau_d

  so tau_b is returned NEGATIVE (it resists) and tau_d positive (it drives).

  Every term returns an identically zero Jacobian outside its valid index range, which is
  how the no-stress boundary conditions are imposed -- there is no separate boundary kernel.
  =======================================================*/

/*=======================================================
  ================== Normal Stress ======================
  =======================================================

  sigma_xx = 2*(eta*H)*eps_xx,   eps_xx = 2*du/dx + dv/dy

  so sigma_xx = H*eta*(4*du/dx + 2*dv/dy), the bracket in the momentum equation.  The
  factor 4 and the appearance of dv/dy in an xx-stress both come from incompressibility:
  the deviatoric stress carries -dw/dz = du/dx + dv/dy, which folds the vertical strain
  rate back into the horizontal ones.  sigma_yy is the same expression with x and y
  exchanged.

  d_eta_H is the partial with respect to the eta*H PRODUCT, not eta -- that is the hook
  through which both the viscosity and the thickness sensitivities reach the momentum rows
  (see viscosity.cu for the product term itself).
 ========================================================*/
// Stencil items that require differentiation
struct SigmaNormalStencil {
    float u_l, u_r, v_t, v_b;
    float eta_H;
};

struct SigmaNormalStencilDual {
    DualFloat u_l, u_r, v_t, v_b;
    DualFloat eta_H;

    __device__ __forceinline__
    SigmaNormalStencil get_primals() const {
        return {u_l.v,u_r.v,v_t.v,v_b.v,eta_H.v};
    }

    __device__ __forceinline__
    SigmaNormalStencil get_diffs() const {
        return {u_l.d,u_r.d,v_t.d,v_b.d,eta_H.d};
    }
};

// Return type for sigma_xx,
// containing residual and jacobian row
struct SigmaNormalJacobian {
    float res;
    float d_u_l, d_u_r, d_v_t, d_v_b;
    float d_eta_H;

    __device__ __forceinline__
    float apply_jvp(const SigmaNormalStencil& dot) const {
        return d_u_l * dot.u_l +
	       d_u_r * dot.u_r +
	       d_v_t * dot.v_t +
	       d_v_b * dot.v_b +
	       d_eta_H * dot.eta_H;
    }

};

__device__ __forceinline__
SigmaNormalJacobian get_sigma_xx_jac(
    SigmaNormalStencil s,
    float dx_inv,
    int i, int j,  // Defined on cells - the i,j for the cell
    int ny, int nx) {

    SigmaNormalJacobian jac= {0};

    if (j < 0 || j >= nx) {
	return jac;
    }

    float eps_xx = (2.0f*(s.u_r - s.u_l)*dx_inv + (s.v_t - s.v_b)*dx_inv);
    float jac_prefactor = 2.0f * s.eta_H * dx_inv;

    jac.res = 2.0f * s.eta_H * eps_xx;
    jac.d_u_l = -2.0f * jac_prefactor;
    jac.d_u_r =  2.0f * jac_prefactor;
    jac.d_v_t =  jac_prefactor;
    jac.d_v_b = -jac_prefactor;
    jac.d_eta_H = 2.0f * eps_xx;
    return jac;
}

__device__ __forceinline__
DualFloat get_sigma_xx_dual(
    SigmaNormalStencilDual s,
    float dx_inv,
    int i, int j,
    int ny, int nx) {
    SigmaNormalJacobian jac = get_sigma_xx_jac(s.get_primals(),dx_inv,i,j,ny,nx);
    return {jac.res,jac.apply_jvp(s.get_diffs())};
}

__device__ __forceinline__
SigmaNormalJacobian get_sigma_yy_jac(
    SigmaNormalStencil s,
    float dx_inv,
    int i, int j,  // Defined on cells - the i,j for the cell
    int ny, int nx) {

    SigmaNormalJacobian jac= {0};

    // No normal stress on out-of-domain cells
    if (i < 0 || i >= ny) {
	return jac;
    }

    float eps_yy = ((s.u_r - s.u_l)*dx_inv + 2.0f*(s.v_t - s.v_b)*dx_inv);
    float jac_prefactor = 2.0f * s.eta_H * dx_inv;

    jac.res = 2.0f * s.eta_H * eps_yy;
    jac.d_u_l = -jac_prefactor;
    jac.d_u_r =  jac_prefactor;
    jac.d_v_t =  2.0f*jac_prefactor;
    jac.d_v_b = -2.0f*jac_prefactor;
    jac.d_eta_H = 2.0f * eps_yy;
    return jac;
}

__device__ __forceinline__
DualFloat get_sigma_yy_dual(
    SigmaNormalStencilDual s,
    float dx_inv,
    int i, int j,
    int ny, int nx) {
    SigmaNormalJacobian jac = get_sigma_yy_jac(s.get_primals(),dx_inv,i,j,ny,nx);
    return {jac.res,jac.apply_jvp(s.get_diffs())};
}

/*======================================================
  ==================== Shear Stress ====================
  ======================================================

  sigma_xy = 2*(eta*H)*eps_xy,   eps_xy = 0.5*(du/dy + dv/dx)

  i.e. H*eta*(du/dy + dv/dx).  This one lives on VERTICES, because du/dy needs two u
  facets stacked vertically and dv/dx two v facets side by side, and the natural place
  those meet is the cell corner.  eta*H therefore has to be averaged over the four cells
  sharing the vertex, which is what get_eta_H_vertex_jac does.

  Zeroed on boundary vertices: no shear traction on the domain edge.
  ======================================================*/

// Stencil items that require differentiation
struct SigmaShearStencil {
    float u_t, u_b, v_l, v_r;
    float eta_H;
};

struct SigmaShearStencilDual {
    DualFloat u_t, u_b, v_l, v_r;
    DualFloat eta_H;

    __device__ __forceinline__
    SigmaShearStencil get_primals() const {
        return {u_t.v,u_b.v,v_l.v,v_r.v,eta_H.v};
    }

    __device__ __forceinline__
    SigmaShearStencil get_diffs() const {
        return {u_t.d,u_b.d,v_l.d,v_r.d,eta_H.d};
    }

};

// Return type for sigma_xx,
// containing residual and jacobian row
struct SigmaShearJacobian {
    float res;
    float d_u_t, d_u_b, d_v_l, d_v_r;
    float d_eta_H;

    __device__ __forceinline__
    float apply_jvp(const SigmaShearStencil& dot) const {
        return d_u_t * dot.u_t +
	       d_u_b * dot.u_b +
	       d_v_l * dot.v_l +
	       d_v_r * dot.v_r +
	       d_eta_H * dot.eta_H;
    }

};


__device__ __forceinline__
SigmaShearJacobian get_sigma_xy_jac(
    SigmaShearStencil s,
    float dx_inv,
    int i, int j, // defined on vertices, the i,j for the vertex
    int ny, int nx) {

    SigmaShearJacobian jac = {0};
    // No shear on boundary vertices
    if (i <= 0 || i >= ny || j <= 0 || j >= nx) {
        return jac;
    }

    float eps_xy = 0.5f*((s.u_t - s.u_b)*dx_inv + (s.v_r - s.v_l)*dx_inv);
    float jac_prefactor = s.eta_H * dx_inv;

    jac.res = 2.0f * s.eta_H * eps_xy;
    jac.d_u_t = jac_prefactor;
    jac.d_u_b = -jac_prefactor;
    jac.d_v_l = -jac_prefactor;
    jac.d_v_r = jac_prefactor;
    jac.d_eta_H = 2.0f * eps_xy;
    return jac;

}

__device__ __forceinline__
DualFloat get_sigma_xy_dual(
    SigmaShearStencilDual s,
    float dx_inv,
    int i, int j,
    int ny, int nx) {
    SigmaShearJacobian jac = get_sigma_xy_jac(s.get_primals(),dx_inv,i,j,ny,nx);
    return {jac.res,jac.apply_jvp(s.get_diffs())};
}

/*=========================================================
  ================== Basal Shear Stress ===================
  =========================================================

  SSA basal drag: tau_b = -c(|u|)*u, with the drag coefficient c set by the sliding law,

    Weertman             c = beta_eff*(|u|^2 + u_reg)^((m-1)/2) + water_drag
    regularized Coulomb  c = beta_eff/(|u| + u_c) + water_drag

  Three things about these stencils are easy to miss:

  1. beta_eff here is just 0.5*(beta_l*phi_l + beta_r*phi_r) -- the cell-centred drag
     averaged onto the facet with GROUNDING FOLDED IN.  It is not the DIVA beta_eff, which
     is a different quantity entirely (see the DIVA section below).  These SSA stencils
     apply the grounded factor themselves; the DIVA ones must not.

  2. The drag couples the velocity COMPONENTS.  c depends on the speed, so
     d(tau_bx)/d(v) is nonzero -- hence d_v_tl..d_v_br.  Written out, the Jacobian of an
     isotropic drag is -[c*delta_ij + (c'/|u|)*u_i*u_j], which is symmetric for any scalar
     c whatsoever, and positive definite exactly when the law is monotone (c' >= 0).  That
     is the property the smoother's local solve depends on, and it holds for a learned c
     just as much as for these two.

  3. |u| at a u-facet is reconstructed as sqrt(u^2 + mean of the four neighbouring v^2) --
     the mean of squares, not the square of the mean.  u_reg keeps it and its derivative
     finite at rest, where the power law is singular for m < 1.

  d_m carries the logf factor because m sits in an exponent: d/dm x^((m-1)/2) =
  x^((m-1)/2) * 0.5*log(x).  The water_drag term has no m in it, hence its absence there.
  =========================================================*/

struct TauBxStencil {
    float u;
    float v_tl, v_tr, v_bl, v_br;
    float H_l, H_r;
    float phi_l, phi_r;
    float beta_l, beta_r;
    float m;
    float u_reg;
    float water_drag;
    float flotation_reg_sliding;
    float u_c_l, u_c_r;
    float sliding_law;
};

struct TauBxStencilDual {
    DualFloat u;
    DualFloat v_tl, v_tr, v_bl, v_br;
    DualFloat H_l, H_r;
    float phi_l, phi_r;
    float beta_l, beta_r;
    float m;
    float u_reg;
    float water_drag;
    float flotation_reg_sliding;
    float u_c_l, u_c_r;
    float sliding_law;

    __device__ __forceinline__
    TauBxStencil get_primals() const {
        return {u.v,v_tl.v,v_tr.v,v_bl.v,v_br.v,H_l.v,H_r.v,phi_l,phi_r,beta_l,beta_r,m,u_reg,water_drag,flotation_reg_sliding,u_c_l,u_c_r,sliding_law};
    }

    __device__ __forceinline__
    TauBxStencil get_diffs() const {
        return {u.d,v_tl.d,v_tr.d,v_bl.d,v_br.d,H_l.d,H_r.d,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
    }

};

struct TauBxJacobian {
    float res;
    float d_u;
    float d_v_tl,d_v_tr,d_v_bl,d_v_br;
    float d_H_l, d_H_r;
    float d_beta_l, d_beta_r;
    float d_m;
    float d_u_c_l, d_u_c_r;

    __device__ __forceinline__
    float apply_jvp(const TauBxStencil& dot) const {
        return d_u * dot.u +
	       d_v_tl * dot.v_tl +
	       d_v_tr * dot.v_tr +
	       d_v_bl * dot.v_bl +
	       d_v_br * dot.v_br +
	       d_H_l * dot.H_l +
	       d_H_r * dot.H_r;
    }
};

__device__ __forceinline__
TauBxJacobian get_tau_bx_jac(
   TauBxStencil s )
{
    TauBxJacobian jac = {0};

    float grounded_l = s.phi_l;
    float grounded_r = s.phi_r;

    float beta_eff_l = s.beta_l * grounded_l;
    float beta_eff_r = s.beta_r * grounded_r;
    float beta_eff = 0.5f*(beta_eff_l + beta_eff_r);

    float unorm_sq = s.u * s.u + 0.25f*(s.v_tl * s.v_tl + s.v_tr * s.v_tr + s.v_bl * s.v_bl + s.v_br * s.v_br);

    if (s.sliding_law < 0.5f) {
        // ---- Weertman power law (default; unchanged) ----
        float unorm_sq_pow = __powf(unorm_sq + s.u_reg,(s.m - 1.0f)/2.0f);
        float unorm_sq_deriv = (s.m - 1.0f)/2.0f * __powf(unorm_sq + s.u_reg,(s.m - 1.0f)/2.0f - 1.0f);

        jac.res = -(beta_eff * unorm_sq_pow + s.water_drag)* s.u;
        jac.d_u = -(beta_eff * (2.0f * unorm_sq_deriv * s.u * s.u + unorm_sq_pow) + s.water_drag);
        jac.d_v_tl = -beta_eff * (0.5f * unorm_sq_deriv * s.u * s.v_tl);
        jac.d_v_tr = -beta_eff * (0.5f * unorm_sq_deriv * s.u * s.v_tr);
        jac.d_v_bl = -beta_eff * (0.5f * unorm_sq_deriv * s.u * s.v_bl);
        jac.d_v_br = -beta_eff * (0.5f * unorm_sq_deriv * s.u * s.v_br);
        jac.d_beta_l = -0.5f * grounded_l * unorm_sq_pow * s.u;
        jac.d_beta_r = -0.5f * grounded_r * unorm_sq_pow * s.u;
        // d(tau_bx^slide)/dm; the water_drag term has no m-dependence.
        jac.d_m = -beta_eff * unorm_sq_pow * s.u * 0.5f * logf(unorm_sq + s.u_reg);
    } else {
        // ---- Regularized Coulomb: |tau_b| = tau_max*|u|/(|u|+u_c), tau_max = beta ----
        float speed = sqrtf(unorm_sq + s.u_reg);
        float u_c = 0.5f*(s.u_c_l + s.u_c_r);
        float D = speed + u_c;
        float C = beta_eff / D;
        float f = beta_eff / (D * D * speed);          // shared factor for velocity derivs

        jac.res = -(C + s.water_drag) * s.u;
        jac.d_u = -(C + s.water_drag) + f * s.u * s.u;
        jac.d_v_tl = f * s.u * 0.25f * s.v_tl;
        jac.d_v_tr = f * s.u * 0.25f * s.v_tr;
        jac.d_v_bl = f * s.u * 0.25f * s.v_bl;
        jac.d_v_br = f * s.u * 0.25f * s.v_br;
        jac.d_beta_l = -0.5f * grounded_l * s.u / D;    // d/d(tau_max)
        jac.d_beta_r = -0.5f * grounded_r * s.u / D;
        jac.d_u_c_l = 0.5f * beta_eff * s.u / (D * D);
        jac.d_u_c_r = 0.5f * beta_eff * s.u / (D * D);
    }
    return jac;
}

__device__ __forceinline__
DualFloat get_tau_bx_dual(TauBxStencilDual s) {
    TauBxJacobian jac = get_tau_bx_jac(s.get_primals());
    return {jac.res,jac.apply_jvp(s.get_diffs())};
}

struct TauByStencil {
    float v;
    float u_tl, u_tr, u_bl, u_br;
    float H_t, H_b;
    float phi_t, phi_b;
    float beta_t, beta_b;
    float m;
    float u_reg;
    float water_drag;
    float flotation_reg_sliding;
    float u_c_t, u_c_b;
    float sliding_law;
};

struct TauByStencilDual {
    DualFloat v;
    DualFloat u_tl, u_tr, u_bl, u_br;
    DualFloat H_t, H_b;
    float phi_t, phi_b;
    float beta_t, beta_b;
    float m;
    float u_reg;
    float water_drag;
    float flotation_reg_sliding;
    float u_c_t, u_c_b;
    float sliding_law;

    __device__ __forceinline__
    TauByStencil get_primals() const {
        return {v.v,u_tl.v,u_tr.v,u_bl.v,u_br.v,H_t.v,H_b.v,phi_t,phi_b,beta_t,beta_b,m,u_reg,water_drag,flotation_reg_sliding,u_c_t,u_c_b,sliding_law};
    }

    __device__ __forceinline__
    TauByStencil get_diffs() const {
        return {v.d,u_tl.d,u_tr.d,u_bl.d,u_br.d,H_t.d,H_t.d,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
    }

};

struct TauByJacobian {
    float res;
    float d_v;
    float d_u_tl,d_u_tr,d_u_bl,d_u_br;
    float d_H_t, d_H_b;
    float d_beta_t, d_beta_b;
    float d_m;
    float d_u_c_t, d_u_c_b;

    __device__ __forceinline__
    float apply_jvp(const TauByStencil& dot) const {
        return d_v * dot.v +
	       d_u_tl * dot.u_tl +
	       d_u_tr * dot.u_tr +
	       d_u_bl * dot.u_bl +
	       d_u_br * dot.u_br +
	       d_H_t * dot.H_t +
	       d_H_b * dot.H_b;
    }
};

__device__ __forceinline__
TauByJacobian get_tau_by_jac(
   TauByStencil s )
{
    TauByJacobian jac = {0};

    float grounded_t = s.phi_t;
    float grounded_b = s.phi_b;

    float beta_eff_t = s.beta_t * grounded_t;
    float beta_eff_b = s.beta_b * grounded_b;
    
    float beta_eff = 0.5f*(beta_eff_t + beta_eff_b);

    float unorm_sq = s.v * s.v + 0.25f*(s.u_tl * s.u_tl + s.u_tr * s.u_tr + s.u_bl * s.u_bl + s.u_br * s.u_br);

    if (s.sliding_law < 0.5f) {
        // ---- Weertman power law (default; unchanged) ----
        float unorm_sq_pow = __powf(unorm_sq + s.u_reg,(s.m - 1.0f)/2.0f);
        float unorm_sq_deriv = (s.m - 1.0f)/2.0f * __powf(unorm_sq + s.u_reg,(s.m - 1.0f)/2.0f - 1.0f);

        jac.res = -(beta_eff * unorm_sq_pow + s.water_drag) * s.v;
        jac.d_v = -(beta_eff * (2.0f * unorm_sq_deriv * s.v * s.v + unorm_sq_pow) + s.water_drag);
        jac.d_u_tl = -beta_eff * (0.5f * unorm_sq_deriv * s.v * s.u_tl);
        jac.d_u_tr = -beta_eff * (0.5f * unorm_sq_deriv * s.v * s.u_tr);
        jac.d_u_bl = -beta_eff * (0.5f * unorm_sq_deriv * s.v * s.u_bl);
        jac.d_u_br = -beta_eff * (0.5f * unorm_sq_deriv * s.v * s.u_br);
        jac.d_beta_t = -0.5f * grounded_t * unorm_sq_pow * s.v;
        jac.d_beta_b = -0.5f * grounded_b * unorm_sq_pow * s.v;
        // d(tau_by^slide)/dm; the water_drag term has no m-dependence.
        jac.d_m = -beta_eff * unorm_sq_pow * s.v * 0.5f * logf(unorm_sq + s.u_reg);
    } else {
        // ---- Regularized Coulomb: |tau_b| = tau_max*|u|/(|u|+u_c), tau_max = beta ----
        float speed = sqrtf(unorm_sq + s.u_reg);
        float u_c = 0.5f*(s.u_c_t + s.u_c_b);
        float D = speed + u_c;
        float C = beta_eff / D;
        float f = beta_eff / (D * D * speed);

        jac.res = -(C + s.water_drag) * s.v;
        jac.d_v = -(C + s.water_drag) + f * s.v * s.v;
        jac.d_u_tl = f * s.v * 0.25f * s.u_tl;
        jac.d_u_tr = f * s.v * 0.25f * s.u_tr;
        jac.d_u_bl = f * s.v * 0.25f * s.u_bl;
        jac.d_u_br = f * s.v * 0.25f * s.u_br;
        jac.d_beta_t = -0.5f * grounded_t * s.v / D;
        jac.d_beta_b = -0.5f * grounded_b * s.v / D;
        jac.d_u_c_t = 0.5f * beta_eff * s.v / (D * D);
        jac.d_u_c_b = 0.5f * beta_eff * s.v / (D * D);
    }

    return jac;
}

__device__ __forceinline__
DualFloat get_tau_by_dual(TauByStencilDual s) {
    TauByJacobian jac = get_tau_by_jac(s.get_primals());
    return {jac.res,jac.apply_jvp(s.get_diffs())};
}
/*=========================================================
  ============ Basal Stress: DIVA (stress_balance = 1) ====
  =========================================================*/
/*
  Under DIVA the momentum balance sees the depth-averaged velocity resisted by an
  *effective* drag rather than by the sliding law directly:

      tau_b = -beta_eff * ubar,     beta_eff = c(U_b)/(1 + c(U_b)*F2)

  (Goldberg 2011 eq 41).  beta_eff and the basal speed U_b are diagnosed per cell by
  compute_diva_coeffs (diva.cu), with the grounded factor already folded into
  beta_eff -- so unlike the SSA stencils above, these must NOT apply it again.

  The result is linear in the velocity, so these stencils are far simpler than their
  SSA counterparts: all of the velocity dependence of the real sliding law has moved
  into the closure that produces U_b.  The extra derivative the augmented block needs,
  d(beta_eff)/d(U_b), is get_diva_dbeta_eff_du_b below.
*/

__device__ __forceinline__
DualFloat get_diva_drag_coeff(
    DualFloat U,
    float beta_grounded, float m, float u_reg, float water_drag,
    float u_c, float sliding_law){

    // Basal drag COEFFICIENT c(U), i.e. |tau_b| = c(U)*U, as a function of the basal
    // speed, returned as a dual number so c'(U) -- and hence f'(U) by the product
    // rule -- falls out of the same evaluation instead of being hand-derived per law.
    // The same helper therefore serves the closure Newton in compute_diva_coeffs and
    // the augmented Jacobian and its transpose, for every sliding law.
    //
    // Returning the coefficient rather than the drag keeps the effective drag free of
    // a 0/0: beta_eff = c/(1 + c*F2) needs no division by the speed.
    //
    // Mirrors the coefficients in get_tau_bx_jac / get_tau_by_jac above:
    //   Weertman            c = beta*(U^2 + u_reg)^((m-1)/2) + water_drag
    //   regularized Coulomb c = beta/(sqrt(U^2 + u_reg) + u_c) + water_drag
    DualFloat U_sq_reg = U*U + u_reg;

    DualFloat c;
    if (sliding_law < 0.5f) {
        c = beta_grounded * __powf(U_sq_reg, 0.5f*(m - 1.0f));
    } else {
        c = beta_grounded / (sqrtf(U_sq_reg) + u_c);
    }

    return c + water_drag;
}

__device__ __forceinline__
float get_diva_dbeta_eff_du_b(
    float U_b, float F2,
    float beta_grounded, float m, float u_reg, float water_drag,
    float u_c, float sliding_law){

    // d(beta_eff)/d(U_b) for beta_eff = c/(1 + c*F2): the quotient rule collapses to
    // c'/(1 + c*F2)^2, with c' supplied by the dual evaluation.
    DualFloat c = get_diva_drag_coeff({U_b,1.0f},beta_grounded,m,u_reg,water_drag,u_c,sliding_law);
    float denom = 1.0f + c.v*F2;
    return c.d/(denom*denom);
}

__device__ __forceinline__
float get_diva_dbeta_eff_dbeta(
    float U_b, float F2,
    float beta, float grounded, float m, float u_reg, float water_drag,
    float u_c, float sliding_law){

    // d(beta_eff)/d(beta), INCLUDING the closure's own response to beta -- this is the
    // parameter sensitivity the inversion needs, and it is not simply
    // d/d(beta)[c/(1+c*F2)] at fixed U_b, because changing beta moves U_b too.
    //
    // Implicit differentiation of R(U_b,beta) = U_b + f(U_b,beta)*F2 - Ubar = 0 gives
    //     dU_b/d(beta) = -f_beta*F2 / (1 + f'*F2)
    // and hence, for tau_b = f(U_b,beta),
    //     d(tau_b)/d(beta) = f_beta / (1 + f'*F2).
    // Writing tau_b = beta_eff*Ubar and eliminating Ubar with the closure identity
    // Ubar = U_b(1 + c*F2) -- which avoids a 0/0 where the ice is at rest -- leaves
    //     d(beta_eff)/d(beta) = c_beta / ((1 + c*F2)(1 + f'*F2)).
    //
    // c_beta is evaluated directly from the law rather than as (c - water_drag)/beta,
    // so it stays finite where beta = 0.  Grounding appears because compute_diva_coeffs
    // forms c from beta*grounded.
    DualFloat c = get_diva_drag_coeff({U_b,1.0f},beta*grounded,m,u_reg,water_drag,u_c,sliding_law);
    DualFloat f = c * DualFloat{U_b,1.0f};        // f = c*U, so f.d = f'(U_b)

    float U_sq_reg = U_b*U_b + u_reg;
    float c_beta = (sliding_law < 0.5f)
                 ? grounded*__powf(U_sq_reg, 0.5f*(m - 1.0f))
                 : grounded/(sqrtf(U_sq_reg) + u_c);

    return c_beta/((1.0f + c.v*F2)*(1.0f + f.d*F2));
}

struct TauBxDivaJacobian {
    float res;
    float d_u;
    float d_beta_eff_l, d_beta_eff_r;
};

__device__ __forceinline__
TauBxDivaJacobian get_tau_bx_diva_jac(
    float u, float beta_eff_l, float beta_eff_r)
{
    TauBxDivaJacobian jac;

    float beta_eff = 0.5f*(beta_eff_l + beta_eff_r);

    jac.res = -beta_eff * u;
    jac.d_u = -beta_eff;
    jac.d_beta_eff_l = -0.5f * u;
    jac.d_beta_eff_r = -0.5f * u;

    return jac;
}

struct TauByDivaJacobian {
    float res;
    float d_v;
    float d_beta_eff_t, d_beta_eff_b;
};

__device__ __forceinline__
TauByDivaJacobian get_tau_by_diva_jac(
    float v, float beta_eff_t, float beta_eff_b)
{
    TauByDivaJacobian jac;

    float beta_eff = 0.5f*(beta_eff_t + beta_eff_b);

    jac.res = -beta_eff * v;
    jac.d_v = -beta_eff;
    jac.d_beta_eff_t = -0.5f * v;
    jac.d_beta_eff_b = -0.5f * v;

    return jac;
}

/*=========================================================
  ==================== Driving Stress =====================
  =========================================================

  tau_d = H * ds/dx, with rho*g absorbed into B and beta (see the file header).  The
  gradient is a plain centred difference of the surface elevation across the facet, times
  the facet-averaged thickness.

  The surface elevation is not stored; it is reconstructed from the bed, the thickness and
  the flotation state:

    base = phi*bed - (1 - phi)*0.917*H      s = base + H

  Grounded (phi = 1) that is s = bed + H.  Afloat (phi = 0) the base is the draft -0.917*H
  and s = 0.083*H, the freeboard.  Blending on phi rather than branching is what keeps the
  transition differentiable through the grounding line -- and note dbase/dH is therefore
  nonzero only for the floating part, which is why d_H_l and d_H_r carry the extra
  (1 + dbase_dH) factor while d_bed_l and d_bed_r are gated by phi.

  As in flux.cu, phi itself is NOT differentiated: it enters the dual stencils as a plain
  float and is updated by its own under-relaxed outer iteration.  So no gradient here sees
  sensitivity acting through grounding-line migration.
  =========================================================*/

struct TauDxStencil {
    float H_l, H_r;
    float bed_l, bed_r;
    float phi_l, phi_r;
    float sigmoid_c;
};

struct TauDxStencilDual {
    DualFloat H_l, H_r;
    float bed_l, bed_r;
    float phi_l, phi_r;
    float sigmoid_c;

    __device__ __forceinline__
    TauDxStencil get_primals() const {
        return {H_l.v,H_r.v,bed_l,bed_r,phi_l,phi_r,sigmoid_c};
    }

    __device__ __forceinline__
    TauDxStencil get_diffs() const {
        return {H_l.d,H_r.d,0.0f,0.0f,0.0f,0.0f,0.0f};
    }

};

struct TauDxJacobian {
    float res;
    float d_H_l, d_H_r;
    float d_bed_l, d_bed_r;

    __device__ __forceinline__
    float apply_jvp(const TauDxStencil& dot) const {
        return d_H_l * dot.H_l +
	       d_H_r * dot.H_r;
    }

};

__device__ __forceinline__
TauDxJacobian get_tau_dx_jac(
    TauDxStencil s,
    float dx_inv,
    int i, int j,  // Defined on facets
    int ny, int nx) {

    TauDxJacobian jac = {0};

    // No driving stress on boundaries
    if (j <= 0 || j >= nx) {
        return jac;
    }

    float H_avg = 0.5f*(s.H_l + s.H_r);
    //float grounded_l = sigmoid(0.917f*s.H_l + s.bed_l,s.sigmoid_c);
    //float grounded_r = sigmoid(0.917f*s.H_r + s.bed_r,s.sigmoid_c);
    float grounded_l = s.phi_l;//sigmoid(s.phi_l,s.sigmoid_c);
    float grounded_r = s.phi_r;//sigmoid(s.phi_r,s.sigmoid_c);

    float base_l = grounded_l * s.bed_l - (1.0f - grounded_l)*0.917f*s.H_l;
    float base_r = grounded_r * s.bed_r - (1.0f - grounded_r)*0.917f*s.H_r;

    float dbase_dH_l = -(1.0f - grounded_l)*0.917f;
    float dbase_dH_r = -(1.0f - grounded_r)*0.917f;

    float S_l = base_l + s.H_l;
    float S_r = base_r + s.H_r;

    jac.res = H_avg * (S_r - S_l) * dx_inv;

    jac.d_H_l = 0.5f*(S_r - S_l)*dx_inv - H_avg*(1.0f + dbase_dH_l)*dx_inv;
    jac.d_H_r = 0.5f*(S_r - S_l)*dx_inv + H_avg*(1.0f + dbase_dH_r)*dx_inv;
    jac.d_bed_l = -H_avg*grounded_l*dx_inv;
    jac.d_bed_r =  H_avg*grounded_r*dx_inv;
    return jac;
}

__device__ __forceinline__
DualFloat get_tau_dx_dual(
    TauDxStencilDual s,
    float dx_inv,
    int i, int j,
    int ny, int nx) {
    TauDxJacobian jac = get_tau_dx_jac(s.get_primals(),dx_inv,i,j,ny,nx);
    return {jac.res,jac.apply_jvp(s.get_diffs())};
}

struct TauDyStencil {
    float H_t, H_b;
    float bed_t, bed_b;
    float phi_t, phi_b;
    float sigmoid_c;
};

struct TauDyStencilDual {
    DualFloat H_t, H_b;
    float bed_t, bed_b;
    float phi_t, phi_b;
    float sigmoid_c;

    __device__ __forceinline__
    TauDyStencil get_primals() const {
        return {H_t.v,H_b.v,bed_t,bed_b,phi_t,phi_b,sigmoid_c};
    }

    __device__ __forceinline__
    TauDyStencil get_diffs() const {
        return {H_t.d,H_b.d,0.0f,0.0f,0.0f,0.0f,0.0f};
    }

};

struct TauDyJacobian {
    float res;
    float d_H_t, d_H_b;
    float d_bed_t, d_bed_b;

    __device__ __forceinline__
    float apply_jvp(const TauDyStencil& dot) const {
        return d_H_t * dot.H_t +
	       d_H_b * dot.H_b;
    }
};

__device__ __forceinline__
TauDyJacobian get_tau_dy_jac(
    TauDyStencil s,
    float dx_inv,
    int i, int j,
    int ny, int nx) {

    TauDyJacobian jac = {0};
    if (i <= 0 || i >= ny) {
        return jac;
    }

    float H_avg = 0.5f*(s.H_t + s.H_b);
    //float grounded_t = sigmoid(0.917f*s.H_t + s.bed_t,s.sigmoid_c);
    //float grounded_b = sigmoid(0.917f*s.H_b + s.bed_b,s.sigmoid_c);
    float grounded_t = s.phi_t;//sigmoid(s.phi_t,s.sigmoid_c);
    float grounded_b = s.phi_b;//sigmoid(s.phi_b,s.sigmoid_c);

    float base_t = grounded_t * s.bed_t - (1.0f - grounded_t)*0.917f*s.H_t;
    float base_b = grounded_b * s.bed_b - (1.0f - grounded_b)*0.917f*s.H_b;

    float dbase_dH_t = -(1.0f - grounded_t)*0.917f;
    float dbase_dH_b = -(1.0f - grounded_b)*0.917f;

    float S_t = base_t + s.H_t;
    float S_b = base_b + s.H_b;

    jac.res = H_avg * (S_t - S_b) * dx_inv;

    jac.d_H_t = 0.5f*(S_t - S_b)*dx_inv + H_avg*(1.0f + dbase_dH_t)*dx_inv;
    jac.d_H_b = 0.5f*(S_t - S_b)*dx_inv - H_avg*(1.0f + dbase_dH_b)*dx_inv;
    jac.d_bed_t =  H_avg*grounded_t*dx_inv;
    jac.d_bed_b = -H_avg*grounded_b*dx_inv;
    return jac;

}

__device__ __forceinline__
DualFloat get_tau_dy_dual(
    TauDyStencilDual s,
    float dx_inv,
    int i, int j,
    int ny, int nx) {
    TauDyJacobian jac = get_tau_dy_jac(s.get_primals(),dx_inv,i,j,ny,nx);
    return {jac.res,jac.apply_jvp(s.get_diffs())};
}

