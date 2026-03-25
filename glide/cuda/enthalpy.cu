/* =========================================================
   ================== Enthalpy Kernels =======================
   =========================================================

   Solves the enthalpy advection-diffusion equation in
   terrain-following (sigma) coordinates:

   rho_i (dE/dt + u dE/dx + v dE/dy + sigma_dot dE/dsigma)
       - (1/h^2) d/dsigma(K dE/dsigma) = phi - rho_w L Dw(omega)

   Discretization:
   - Horizontal: finite volume with upwind fluxes on MAC grid
   - Vertical: finite differences on non-uniform sigma nodes
   - Column-wise Newton/Thomas solve as multigrid smoother

   Architecture:
   Each physical term is decomposed into a Stencil/Jacobian pair
   following the same pattern as the momentum balance (flux.cu,
   stress.cu). The get_*_jac() function is the single source of
   truth for both the residual and its derivatives. Both the
   residual kernel and the column smoother call the same functions.
   ========================================================= */

// ---- Physical constants ----
#define RHO_I   910.0f      // Ice density (kg/m^3)
#define RHO_W   1000.0f     // Water density (kg/m^3)
#define C_I     2009.0f     // Heat capacity of ice (J/(kg*K))
#define K_I     2.1f        // Thermal conductivity of ice (W/(m*K))
#define L_HEAT  3.34e5f     // Latent heat of fusion (J/kg)
#define T_REF   223.15f     // Reference temperature (K)
#define T_MELT  273.15f     // Melting point at standard pressure (K)
#define BETA_CC 7.9e-8f     // Clausius-Clapeyron constant (K/Pa)
#define GRAVITY 9.81f       // Gravitational acceleration (m/s^2)
#define K_COLD  (K_I/C_I)   // Diffusivity for cold ice
#define K_TEMP_FACTOR 1e-5f // Temperate conductivity reduction factor

// Maximum number of vertical sigma levels
#define MAX_NZ 64


// ---- Helper: enthalpy at pressure melting point ----
__device__ __forceinline__
float get_E_pmp(float sigma, float H) {
    // depth = (1 - sigma) * H
    float depth = (1.0f - sigma) * H;
    float T_pmp = T_MELT - BETA_CC * RHO_I * GRAVITY * depth;
    return C_I * (T_pmp - T_REF);
}

// ---- Helper: diffusivity K(E) with smooth transition ----
__device__ __forceinline__
float get_K(float E, float E_pmp) {
    // Smooth sigmoid transition from cold to temperate
    // K_cold for E < E_pmp, K_temp for E >= E_pmp
    float z = (E - E_pmp) * 0.01f;  // transition sharpness
    z = fminf(fmaxf(z, -20.0f), 20.0f);
    float s = 1.0f / (1.0f + __expf(-z));
    return K_COLD * (1.0f - s + K_TEMP_FACTOR * s);
}

// ---- Helper: derivative dK/dE ----
__device__ __forceinline__
float get_dK_dE(float E, float E_pmp) {
    float z = (E - E_pmp) * 0.01f;
    z = fminf(fmaxf(z, -20.0f), 20.0f);
    float s = 1.0f / (1.0f + __expf(-z));
    float ds_dE = 0.01f * s * (1.0f - s);
    return K_COLD * (K_TEMP_FACTOR - 1.0f) * ds_dE;
}

// ---- Helper: water content from enthalpy ----
__device__ __forceinline__
float get_omega(float E, float E_pmp) {
    if (E > E_pmp) {
        return (E - E_pmp) / L_HEAT;
    }
    return 0.0f;
}

// ---- Helper: drainage function Dw(omega) ----
__device__ __forceinline__
float get_drainage(float omega, float drain_rate) {
    // Simple linear drainage: Dw = drain_rate * omega
    return drain_rate * omega;
}

// ---- Helper: 3D cell-centered access for E (ny, nx, nz layout) ----
__device__ __forceinline__
float get_E(const float* __restrict__ E, int i, int j, int k,
            int ny, int nx, int nz) {
    i = max(min(i, ny - 1), 0);
    j = max(min(j, nx - 1), 0);
    k = max(min(k, nz - 1), 0);
    return E[(i * nx + j) * nz + k];
}

// ---- Helper: 3D facet access for u (nz, ny, nx+1 layout) ----
__device__ __forceinline__
float get_u3d(const float* __restrict__ u, int k, int i, int j,
              int nz, int ny, int nx) {
    k = max(min(k, nz - 1), 0);
    i = max(min(i, ny - 1), 0);
    j = max(min(j, nx), 0);
    return u[(k * ny + i) * (nx + 1) + j];
}

// ---- Helper: 3D facet access for v (nz, ny+1, nx layout) ----
__device__ __forceinline__
float get_v3d(const float* __restrict__ v, int k, int i, int j,
              int nz, int ny, int nx) {
    k = max(min(k, nz - 1), 0);
    i = max(min(i, ny), 0);
    j = max(min(j, nx - 1), 0);
    return v[(k * (ny + 1) + i) * nx + j];
}


/* =========================================================
   ========== Jacobian Structs for Enthalpy Terms ===========
   =========================================================

   Each physical term is decomposed into:
     1. Stencil  — local inputs needed to evaluate the term
     2. Jacobian — residual + partial derivatives w.r.t. column unknowns
     3. get_*_jac() — single function computing both res and d_E_*

   This mirrors the momentum balance pattern in flux.cu / stress.cu
   (HorizontalFluxStencil/Jacobian, SigmaNormalStencil/Jacobian, etc.)
   ========================================================= */


/* ---------------------------------------------------------
   Vertical Diffusion:
     -(1/h^2) d/dsigma(K dE/dsigma)
   at an interior node k with neighbors k-1, k+1.
   Returns tridiagonal contributions (d_E_km1, d_E_k, d_E_kp1).
   --------------------------------------------------------- */
struct ColumnDiffusionStencil {
    float E_km1, E_k, E_kp1;
    float E_pmp_km1, E_pmp_k, E_pmp_kp1;
    float dsig_m, dsig_p;
    float h2_inv;
};

struct ColumnDiffusionStencilDual {
    DualFloat E_km1, E_k, E_kp1;
    float E_pmp_km1, E_pmp_k, E_pmp_kp1;
    float dsig_m, dsig_p;
    float h2_inv;

    __device__ __forceinline__
    ColumnDiffusionStencil get_primals() const {
        return {E_km1.v, E_k.v, E_kp1.v,
                E_pmp_km1, E_pmp_k, E_pmp_kp1,
                dsig_m, dsig_p, h2_inv};
    }

    __device__ __forceinline__
    ColumnDiffusionStencil get_diffs() const {
        return {E_km1.d, E_k.d, E_kp1.d,
                0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f};
    }
};

struct ColumnDiffusionJacobian {
    float res;
    float d_E_km1, d_E_k, d_E_kp1;

    __device__ __forceinline__
    float apply_jvp(const ColumnDiffusionStencil& dot) const {
        return d_E_km1 * dot.E_km1 + d_E_k * dot.E_k + d_E_kp1 * dot.E_kp1;
    }
};

__device__ __forceinline__
ColumnDiffusionJacobian get_column_diffusion_jac(ColumnDiffusionStencil s) {
    ColumnDiffusionJacobian jac = {0};

    float dsig_avg = 0.5f * (s.dsig_m + s.dsig_p);

    float K_upper = get_K(0.5f * (s.E_k + s.E_kp1),
                          0.5f * (s.E_pmp_k + s.E_pmp_kp1));
    float K_lower = get_K(0.5f * (s.E_k + s.E_km1),
                          0.5f * (s.E_pmp_k + s.E_pmp_km1));

    // Residual: -(1/h^2)/dsig_avg * [K_upper*(E_kp1-E_k)/dsig_p - K_lower*(E_k-E_km1)/dsig_m]
    jac.res = -s.h2_inv / dsig_avg * (K_upper * (s.E_kp1 - s.E_k) / s.dsig_p
                                     - K_lower * (s.E_k - s.E_km1) / s.dsig_m);

    // Jacobian entries (linearized: freeze K at current state)
    float diff_lower = s.h2_inv * K_lower / (dsig_avg * s.dsig_m);
    float diff_upper = s.h2_inv * K_upper / (dsig_avg * s.dsig_p);

    jac.d_E_km1 = -diff_lower;
    jac.d_E_kp1 = -diff_upper;
    jac.d_E_k   =  diff_lower + diff_upper;

    return jac;
}

__device__ __forceinline__
DualFloat get_column_diffusion_dual(ColumnDiffusionStencilDual s) {
    ColumnDiffusionJacobian jac = get_column_diffusion_jac(s.get_primals());
    return {jac.res, jac.apply_jvp(s.get_diffs())};
}


/* ---------------------------------------------------------
   Bed Diffusion (basal flux BC):
     -(1/h^2) K_{1/2} (E_1 - E_0)/dsig_p - (Q_geo + Q_fh)/h
   Returns contributions for k=0 only (d_E_k = d_E_0, d_E_kp1 = d_E_1).
   --------------------------------------------------------- */
struct BedDiffusionStencil {
    float E_k, E_kp1;
    float E_pmp_k, E_pmp_kp1;
    float dsig_p;
    float h2_inv, h_inv;
    float Q_geo, Q_fh;
};

struct BedDiffusionStencilDual {
    DualFloat E_k, E_kp1;
    float E_pmp_k, E_pmp_kp1;
    float dsig_p;
    float h2_inv, h_inv;
    float Q_geo, Q_fh;

    __device__ __forceinline__
    BedDiffusionStencil get_primals() const {
        return {E_k.v, E_kp1.v, E_pmp_k, E_pmp_kp1,
                dsig_p, h2_inv, h_inv, Q_geo, Q_fh};
    }

    __device__ __forceinline__
    BedDiffusionStencil get_diffs() const {
        return {E_k.d, E_kp1.d, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    }
};

struct BedDiffusionJacobian {
    float res;
    float d_E_k, d_E_kp1;

    __device__ __forceinline__
    float apply_jvp(const BedDiffusionStencil& dot) const {
        return d_E_k * dot.E_k + d_E_kp1 * dot.E_kp1;
    }
};

__device__ __forceinline__
BedDiffusionJacobian get_bed_diffusion_jac(BedDiffusionStencil s) {
    BedDiffusionJacobian jac = {0};

    float K_half = get_K(0.5f * (s.E_k + s.E_kp1),
                         0.5f * (s.E_pmp_k + s.E_pmp_kp1));

    // Half-cell control volume width at the bed boundary.
    // The boundary node sits at sigma=0; the control volume extends to
    // the midpoint with the first interior node, dsig_p/2.  Dividing
    // the flux divergence by dsig_half keeps the bed cell consistent
    // with the interior cells (which divide by dsig_avg).
    float dsig_half = 0.5f * s.dsig_p;

    // Residual: -(1/h^2)[K_{1/2}(E_1-E_0)/dsig_p + h(Q_geo+Q_fh)] / dsig_half
    jac.res = (-s.h2_inv * K_half * (s.E_kp1 - s.E_k) / s.dsig_p
               - (s.Q_geo + s.Q_fh) * s.h_inv) / dsig_half;

    // Jacobian (freeze K)
    float coeff = s.h2_inv * K_half / (s.dsig_p * dsig_half);
    jac.d_E_k   =  coeff;
    jac.d_E_kp1 = -coeff;

    return jac;
}

__device__ __forceinline__
DualFloat get_bed_diffusion_dual(BedDiffusionStencilDual s) {
    BedDiffusionJacobian jac = get_bed_diffusion_jac(s.get_primals());
    return {jac.res, jac.apply_jvp(s.get_diffs())};
}


/* ---------------------------------------------------------
   Sigma (Vertical) Advection:
     rho_i * sigma_dot * dE/dsigma  (upwind)
   Returns tridiagonal contributions.
   --------------------------------------------------------- */
struct SigmaAdvectionStencil {
    float E_km1, E_k, E_kp1;
    float sigma_dot;
    float dsig_m, dsig_p;
};

struct SigmaAdvectionStencilDual {
    DualFloat E_km1, E_k, E_kp1;
    float sigma_dot;
    float dsig_m, dsig_p;

    __device__ __forceinline__
    SigmaAdvectionStencil get_primals() const {
        return {E_km1.v, E_k.v, E_kp1.v,
                sigma_dot, dsig_m, dsig_p};
    }

    __device__ __forceinline__
    SigmaAdvectionStencil get_diffs() const {
        return {E_km1.d, E_k.d, E_kp1.d,
                0.0f, 0.0f, 0.0f};
    }
};

struct SigmaAdvectionJacobian {
    float res;
    float d_E_km1, d_E_k, d_E_kp1;

    __device__ __forceinline__
    float apply_jvp(const SigmaAdvectionStencil& dot) const {
        return d_E_km1 * dot.E_km1 + d_E_k * dot.E_k + d_E_kp1 * dot.E_kp1;
    }
};

__device__ __forceinline__
SigmaAdvectionJacobian get_sigma_advection_jac(SigmaAdvectionStencil s) {
    SigmaAdvectionJacobian jac = {0};

    float sd_pos = fmaxf(s.sigma_dot, 0.0f);
    float sd_neg = fminf(s.sigma_dot, 0.0f);

    // Residual: rho_i * [sd+ * (E_k - E_km1)/dsig_m + sd- * (E_kp1 - E_k)/dsig_p]
    jac.res = RHO_I * (sd_pos * (s.E_k - s.E_km1) / s.dsig_m
                      + sd_neg * (s.E_kp1 - s.E_k) / s.dsig_p);

    // Jacobian
    jac.d_E_km1 = -RHO_I * sd_pos / s.dsig_m;
    jac.d_E_kp1 =  RHO_I * sd_neg / s.dsig_p;
    jac.d_E_k   =  RHO_I * sd_pos / s.dsig_m - RHO_I * sd_neg / s.dsig_p;

    return jac;
}

__device__ __forceinline__
DualFloat get_sigma_advection_dual(SigmaAdvectionStencilDual s) {
    SigmaAdvectionJacobian jac = get_sigma_advection_jac(s.get_primals());
    return {jac.res, jac.apply_jvp(s.get_diffs())};
}


/* ---------------------------------------------------------
   Bed Sigma Advection (k=0, one-sided):
     Only downward (sd < 0) can be upwinded; sd > 0 gives zero
     gradient at bed.
   --------------------------------------------------------- */
struct BedSigmaAdvectionStencil {
    float E_k, E_kp1;
    float sigma_dot;
    float dsig_p;
};

struct BedSigmaAdvectionStencilDual {
    DualFloat E_k, E_kp1;
    float sigma_dot;
    float dsig_p;

    __device__ __forceinline__
    BedSigmaAdvectionStencil get_primals() const {
        return {E_k.v, E_kp1.v, sigma_dot, dsig_p};
    }

    __device__ __forceinline__
    BedSigmaAdvectionStencil get_diffs() const {
        return {E_k.d, E_kp1.d, 0.0f, 0.0f};
    }
};

struct BedSigmaAdvectionJacobian {
    float res;
    float d_E_k, d_E_kp1;

    __device__ __forceinline__
    float apply_jvp(const BedSigmaAdvectionStencil& dot) const {
        return d_E_k * dot.E_k + d_E_kp1 * dot.E_kp1;
    }
};

__device__ __forceinline__
BedSigmaAdvectionJacobian get_bed_sigma_advection_jac(BedSigmaAdvectionStencil s) {
    BedSigmaAdvectionJacobian jac = {0};

    if (s.sigma_dot < 0.0f) {
        // Upwind from above
        jac.res = RHO_I * s.sigma_dot * (s.E_kp1 - s.E_k) / s.dsig_p;
        jac.d_E_k   = -RHO_I * s.sigma_dot / s.dsig_p;
        jac.d_E_kp1 =  RHO_I * s.sigma_dot / s.dsig_p;
    }
    // sd >= 0: cannot upwind from below bed, zero contribution

    return jac;
}

__device__ __forceinline__
DualFloat get_bed_sigma_advection_dual(BedSigmaAdvectionStencilDual s) {
    BedSigmaAdvectionJacobian jac = get_bed_sigma_advection_jac(s.get_primals());
    return {jac.res, jac.apply_jvp(s.get_diffs())};
}


/* ---------------------------------------------------------
   Horizontal Enthalpy Flux (per facet, upwind):
     F = u * E_upwind
   The "self" derivative (d_E_here) is needed for the column
   Jacobian diagonal when E_here appears as the upwind donor.
   --------------------------------------------------------- */
struct HorizEnthalpyFluxStencil {
    float u;          // facet velocity
    float E_here;     // cell-center enthalpy of this column
    float E_neighbor; // cell-center enthalpy of neighbor column
    bool outflow;     // true if this column is the upwind donor
};

struct HorizEnthalpyFluxStencilDual {
    float u;
    DualFloat E_here;
    DualFloat E_neighbor;
    bool outflow;

    __device__ __forceinline__
    HorizEnthalpyFluxStencil get_primals() const {
        return {u, E_here.v, E_neighbor.v, outflow};
    }

    __device__ __forceinline__
    HorizEnthalpyFluxStencil get_diffs() const {
        return {0.0f, E_here.d, E_neighbor.d, false};
    }
};

struct HorizEnthalpyFluxJacobian {
    float res;       // flux value
    float d_E_here;  // derivative w.r.t. this column's E (for diagonal)

    __device__ __forceinline__
    float apply_jvp(const HorizEnthalpyFluxStencil& dot) const {
        return d_E_here * dot.E_here;
    }
};

__device__ __forceinline__
HorizEnthalpyFluxJacobian get_horiz_enthalpy_flux_jac(HorizEnthalpyFluxStencil s) {
    HorizEnthalpyFluxJacobian jac = {0};

    if (s.outflow) {
        // This column is the upwind donor: flux = u * E_here
        jac.res = s.u * s.E_here;
        jac.d_E_here = s.u;
    } else {
        // Neighbor is the upwind donor: flux = u * E_neighbor (frozen)
        jac.res = s.u * s.E_neighbor;
        jac.d_E_here = 0.0f;
    }

    return jac;
}

__device__ __forceinline__
DualFloat get_horiz_enthalpy_flux_dual(HorizEnthalpyFluxStencilDual s) {
    HorizEnthalpyFluxJacobian jac = get_horiz_enthalpy_flux_jac(s.get_primals());
    return {jac.res, jac.apply_jvp(s.get_diffs())};
}


/* ---------------------------------------------------------
   Drainage Source Term:
     rho_w * L * drain_rate * omega(E, E_pmp)
   Positive = removes enthalpy from temperate ice.
   --------------------------------------------------------- */
struct DrainageStencil {
    float E_k;
    float E_pmp_k;
    float drain_rate;
};

struct DrainageStencilDual {
    DualFloat E_k;
    float E_pmp_k;
    float drain_rate;

    __device__ __forceinline__
    DrainageStencil get_primals() const {
        return {E_k.v, E_pmp_k, drain_rate};
    }

    __device__ __forceinline__
    DrainageStencil get_diffs() const {
        return {E_k.d, 0.0f, 0.0f};
    }
};

struct DrainageJacobian {
    float res;
    float d_E_k;

    __device__ __forceinline__
    float apply_jvp(const DrainageStencil& dot) const {
        return d_E_k * dot.E_k;
    }
};

__device__ __forceinline__
DrainageJacobian get_drainage_jac(DrainageStencil s) {
    DrainageJacobian jac = {0};

    float omega = get_omega(s.E_k, s.E_pmp_k);
    jac.res = RHO_W * L_HEAT * get_drainage(omega, s.drain_rate);

    // d/dE(rho_w * L * drain_rate * omega) = rho_w * drain_rate when E > E_pmp
    if (s.E_k > s.E_pmp_k) {
        jac.d_E_k = RHO_W * s.drain_rate;
    }

    return jac;
}

__device__ __forceinline__
DualFloat get_drainage_dual(DrainageStencilDual s) {
    DrainageJacobian jac = get_drainage_jac(s.get_primals());
    return {jac.res, jac.apply_jvp(s.get_diffs())};
}


/* ---------------------------------------------------------
   Horizontal advection assembly for one layer.
   Computes the net horizontal advection contribution and
   the diagonal Jacobian term from outflow faces.
   --------------------------------------------------------- */
struct HorizAdvectionResult {
    float res;       // rho_i * (flux_right - flux_left + flux_bottom - flux_top) / dx
    float d_E_here;  // sum of d_E_here from all outflow faces, scaled by rho_i/dx
};

__device__ __forceinline__
HorizAdvectionResult get_horiz_advection(
    const float* __restrict__ E,
    const float* __restrict__ u3d,
    const float* __restrict__ v3d,
    float E_here,
    int i, int j, int k,
    int ny, int nx, int nz,
    float dx_inv)
{
    HorizAdvectionResult result = {0};

    // x-direction
    float u_left  = get_u3d(u3d, k, i, j,   nz, ny, nx);
    float u_right = get_u3d(u3d, k, i, j+1, nz, ny, nx);
    float E_xm = get_E(E, i, j-1, k, ny, nx, nz);
    float E_xp = get_E(E, i, j+1, k, ny, nx, nz);

    HorizEnthalpyFluxJacobian fl = {0};
    HorizEnthalpyFluxJacobian fr = {0};

    if (j > 0) {
        fl = get_horiz_enthalpy_flux_jac({u_left, E_here, E_xm, u_left < 0.0f});
    }
    if (j < nx - 1) {
        fr = get_horiz_enthalpy_flux_jac({u_right, E_here, E_xp, u_right > 0.0f});
    }

    // y-direction
    float v_top    = get_v3d(v3d, k, i,   j, nz, ny, nx);
    float v_bottom = get_v3d(v3d, k, i+1, j, nz, ny, nx);
    float E_ym = get_E(E, i-1, j, k, ny, nx, nz);
    float E_yp = get_E(E, i+1, j, k, ny, nx, nz);

    HorizEnthalpyFluxJacobian ft = {0};
    HorizEnthalpyFluxJacobian fb = {0};

    if (i > 0) {
        ft = get_horiz_enthalpy_flux_jac({v_top, E_here, E_ym, v_top < 0.0f});
    }
    if (i < ny - 1) {
        fb = get_horiz_enthalpy_flux_jac({v_bottom, E_here, E_yp, v_bottom > 0.0f});
    }

    result.res = RHO_I * ((fr.res - fl.res) + (fb.res - ft.res)) * dx_inv;
    result.d_E_here = RHO_I * ((fr.d_E_here - fl.d_E_here)
                              + (fb.d_E_here - ft.d_E_here)) * dx_inv;

    return result;
}


/* =========================================================
   Compute the enthalpy residual at all (i, j, k) points.

   Uses the Jacobian structs above — only reads .res fields.
   This guarantees residual/Jacobian consistency: both kernels
   evaluate the same get_*_jac() functions.
   ========================================================= */
extern "C" __global__
void enthalpy_compute_residual(
    float* __restrict__ r_E,            // (ny, nx, nz) residual output
    const float* __restrict__ E,        // (ny, nx, nz) current enthalpy
    const float* __restrict__ E_prev,   // (ny, nx, nz) previous time step
    const float* __restrict__ f_E,      // (ny, nx, nz) FAS tau correction (0 on finest)
    const float* __restrict__ u3d,      // (nz, ny, nx+1) x-velocity per layer
    const float* __restrict__ v3d,      // (nz, ny+1, nx) y-velocity per layer
    const float* __restrict__ sigma_dot,// (ny, nx, nz) sigma velocity
    const float* __restrict__ H,        // (ny, nx) ice thickness
    const float* __restrict__ phi_strain,// (ny, nx, nz) strain heating
    const float* __restrict__ E_surface,// (ny, nx) surface enthalpy BC
    const float* __restrict__ Q_geo,    // (ny, nx) geothermal heat flux
    const float* __restrict__ Q_fh,     // (ny, nx) frictional heating
    const float* __restrict__ sigma,    // (nz,) sigma node positions
    float dx, float dt, float drain_rate, float h_thin,
    int ny, int nx, int nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_columns = ny * nx;
    if (idx >= n_columns) return;

    int i = idx / nx;
    int j = idx % nx;

    float h = H[i * nx + j];

    // Thin-ice bypass: clamp entire column to surface enthalpy
    // Thin-ice and surface Dirichlet cells are not solver unknowns.
    // Their residual is zero so they don't pollute the convergence norm.
    if (h < h_thin) {
        for (int k = 0; k < nz; k++) {
            r_E[(i * nx + j) * nz + k] = 0.0f;
        }
        return;
    }

    float h_inv = 1.0f / h;
    float h2_inv = h_inv * h_inv;
    float dx_inv = 1.0f / dx;

    for (int k = 0; k < nz; k++) {
        int ijk = (i * nx + j) * nz + k;

        // Surface Dirichlet BC: not a solver unknown, zero residual.
        if (k == nz - 1) {
            r_E[ijk] = 0.0f;
            continue;
        }

        float E_k = E[ijk];
        float sigma_k = sigma[k];
        float E_pmp_k = get_E_pmp(sigma_k, h);

        // --- Time derivative ---
        float r = RHO_I * (E_k - E_prev[ijk]) / dt - f_E[ijk];

        // --- Horizontal advection ---
        HorizAdvectionResult h_adv = get_horiz_advection(
            E, u3d, v3d, E_k, i, j, k, ny, nx, nz, dx_inv);
        r += h_adv.res;

        // --- Vertical advection + diffusion ---
        if (k == 0) {
            // Bed boundary
            float E_pmp_bed = get_E_pmp(0.0f, h);

            float E_kp1 = get_E(E, i, j, 1, ny, nx, nz);
            float dsig_p = sigma[1] - sigma[0];

            BedSigmaAdvectionJacobian adv_jac = get_bed_sigma_advection_jac(
                {E_k, E_kp1, sigma_dot[ijk], dsig_p});
            r += adv_jac.res;

            BedDiffusionJacobian diff_jac = get_bed_diffusion_jac(
                {E_k, E_kp1, E_pmp_bed, get_E_pmp(sigma[1], h),
                 dsig_p, h2_inv, h_inv, Q_geo[i*nx+j], Q_fh[i*nx+j]});
            r += diff_jac.res;

            // Geothermal and frictional heat continue to enter the column
            // after the bed reaches pressure melting. Excess enthalpy is
            // carried as latent heat and may be removed only by drainage.
            float phi_k = phi_strain[ijk];
            DrainageJacobian drain_jac = get_drainage_jac({E_k, E_pmp_bed, drain_rate});
            r -= phi_k;
            r += drain_jac.res;
        } else {
            // Interior layers (k = 1 .. nz-2)
            float dsig_m = sigma[k] - sigma[k-1];
            float dsig_p = sigma[k+1] - sigma[k];
            float E_km1 = get_E(E, i, j, k-1, ny, nx, nz);
            float E_kp1 = (k < nz-1) ? get_E(E, i, j, k+1, ny, nx, nz)
                                      : E_surface[i*nx+j];

            float E_pmp_km1 = get_E_pmp(sigma[max(k-1,0)], h);
            float E_pmp_kp1 = get_E_pmp(sigma[min(k+1,nz-1)], h);

            SigmaAdvectionJacobian adv_jac = get_sigma_advection_jac(
                {E_km1, E_k, E_kp1, sigma_dot[ijk], dsig_m, dsig_p});
            r += adv_jac.res;

            ColumnDiffusionJacobian diff_jac = get_column_diffusion_jac(
                {E_km1, E_k, E_kp1, E_pmp_km1, E_pmp_k, E_pmp_kp1,
                 dsig_m, dsig_p, h2_inv});
            r += diff_jac.res;

            // Source terms
            float phi_k = phi_strain[ijk];
            DrainageJacobian drain_jac = get_drainage_jac({E_k, E_pmp_k, drain_rate});
            r -= phi_k;
            r += drain_jac.res;
        }

        r_E[ijk] = r;
    }
}


/* =========================================================
   Column-wise Newton/Thomas smoother.

   For each column (i,j), freezes horizontal neighbors and
   solves the vertical tridiagonal system using the Thomas
   algorithm. Newton iteration handles the K(E) nonlinearity.

   Uses the same get_*_jac() functions as the residual kernel.
   The Jacobian .d_E_* fields map directly to the tridiagonal
   entries a[k], b[k], c[k], guaranteeing consistency.
   ========================================================= */
extern "C" __global__
void enthalpy_column_smooth(
    float* __restrict__ delta_E,        // (ny, nx, nz) correction output
    const float* __restrict__ E,        // (ny, nx, nz) current enthalpy
    const float* __restrict__ E_prev,   // (ny, nx, nz) previous time step
    const float* __restrict__ f_E,      // (ny, nx, nz) FAS tau correction
    const float* __restrict__ u3d,      // (nz, ny, nx+1) x-velocity
    const float* __restrict__ v3d,      // (nz, ny+1, nx) y-velocity
    const float* __restrict__ sigma_dot,// (ny, nx, nz) sigma velocity
    const float* __restrict__ H,        // (ny, nx) thickness
    const float* __restrict__ phi_strain,// (ny, nx, nz) strain heating
    const float* __restrict__ E_surface,// (ny, nx) surface BC
    const float* __restrict__ Q_geo,    // (ny, nx) geothermal heat flux
    const float* __restrict__ Q_fh,     // (ny, nx) frictional heating
    const float* __restrict__ sigma,    // (nz,) sigma positions
    float dx, float dt, float drain_rate, float h_thin,
    int ny, int nx, int nz,
    int n_newton, float relaxation)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_columns = ny * nx;
    if (idx >= n_columns) return;

    int i = idx / nx;
    int j = idx % nx;

    float h = H[i * nx + j];

    // Thin-ice bypass: correction drives entire column toward E_surface
    if (h < h_thin) {
        float E_s = E_surface[i * nx + j];
        for (int k = 0; k < nz; k++) {
            delta_E[(i * nx + j) * nz + k] = E_s - E[(i * nx + j) * nz + k];
        }
        return;
    }

    float h_inv = 1.0f / h;
    float h2_inv = h_inv * h_inv;
    float dx_inv = 1.0f / dx;

    // Local copy of column enthalpy for Newton iteration
    float E_local[MAX_NZ];
    for (int k = 0; k < nz; k++) {
        E_local[k] = E[(i * nx + j) * nz + k];
    }

    // Precompute horizontal advection per layer (frozen neighbors).
    // We store both the residual and the diagonal Jacobian contribution
    // from outflow faces where E_here is the upwind donor.
    float horiz_adv_res[MAX_NZ];
    float horiz_adv_diag[MAX_NZ];
    for (int k = 0; k < nz; k++) {
        HorizAdvectionResult h_adv = get_horiz_advection(
            E, u3d, v3d, E_local[k], i, j, k, ny, nx, nz, dx_inv);
        horiz_adv_res[k]  = h_adv.res;
        horiz_adv_diag[k] = h_adv.d_E_here;
    }

    // --- Newton iteration ---
    for (int newton = 0; newton < n_newton; newton++) {

        // Build tridiagonal: a[k]*dE[k-1] + b[k]*dE[k] + c[k]*dE[k+1] = -r[k]
        float a[MAX_NZ], b[MAX_NZ], c_arr[MAX_NZ], rhs[MAX_NZ];

        float E_s = E_surface[i * nx + j];

        for (int k = 0; k < nz; k++) {
            a[k] = 0.0f;
            b[k] = 0.0f;
            c_arr[k] = 0.0f;
            rhs[k] = 0.0f;
        }

        // Surface: Dirichlet
        {
            int k = nz - 1;
            b[k] = 1.0f;
            rhs[k] = -(E_local[k] - E_s);
        }

        // Bed: k = 0
        {
            float E_k = E_local[0];
            float E_pmp_bed = get_E_pmp(0.0f, h);

            float dsig_p = sigma[1] - sigma[0];
            float E_kp1 = E_local[1];
            float E_pmp_1 = get_E_pmp(sigma[1], h);

            // Residual
            float r = RHO_I * (E_k - E_prev[(i*nx+j)*nz]) / dt - f_E[(i*nx+j)*nz];
            r += horiz_adv_res[0];

            // Vertical advection at bed
            BedSigmaAdvectionJacobian adv_jac = get_bed_sigma_advection_jac(
                {E_k, E_kp1, sigma_dot[(i*nx+j)*nz], dsig_p});
            r += adv_jac.res;

            // Diffusion with persistent basal heat flux
            BedDiffusionJacobian diff_jac = get_bed_diffusion_jac(
                {E_k, E_kp1, E_pmp_bed, E_pmp_1,
                 dsig_p, h2_inv, h_inv,
                 Q_geo[i*nx+j], Q_fh[i*nx+j]});
            r += diff_jac.res;

            // Source
            float phi_k = phi_strain[(i*nx+j)*nz];
            DrainageJacobian drain_jac = get_drainage_jac({E_k, E_pmp_bed, drain_rate});
            r -= phi_k;
            r += drain_jac.res;

            // Jacobian: assemble from struct derivatives
            b[0] = RHO_I / dt
                 + diff_jac.d_E_k
                 + adv_jac.d_E_k
                 + drain_jac.d_E_k
                 + horiz_adv_diag[0];
            c_arr[0] = diff_jac.d_E_kp1
                     + adv_jac.d_E_kp1;

            rhs[0] = -r;
        }

        // Interior layers: k = 1 .. nz-2
        for (int k = 1; k < nz - 1; k++) {
            float E_k = E_local[k];
            float sigma_k = sigma[k];
            float E_pmp_k = get_E_pmp(sigma_k, h);

            float dsig_m = sigma[k] - sigma[k-1];
            float dsig_p = sigma[k+1] - sigma[k];

            float E_km1 = E_local[k-1];
            float E_kp1 = (k < nz - 1) ? E_local[k+1] : E_s;

            float E_pmp_km1 = get_E_pmp(sigma[k-1], h);
            float E_pmp_kp1 = get_E_pmp(sigma[min(k+1,nz-1)], h);

            // --- Residual and Jacobian via struct functions ---
            float r = RHO_I * (E_k - E_prev[(i*nx+j)*nz + k]) / dt - f_E[(i*nx+j)*nz + k];
            r += horiz_adv_res[k];

            SigmaAdvectionJacobian adv_jac = get_sigma_advection_jac(
                {E_km1, E_k, E_kp1, sigma_dot[(i*nx+j)*nz + k], dsig_m, dsig_p});
            r += adv_jac.res;

            ColumnDiffusionJacobian diff_jac = get_column_diffusion_jac(
                {E_km1, E_k, E_kp1, E_pmp_km1, E_pmp_k, E_pmp_kp1,
                 dsig_m, dsig_p, h2_inv});
            r += diff_jac.res;

            float phi_k = phi_strain[(i*nx+j)*nz + k];
            DrainageJacobian drain_jac = get_drainage_jac({E_k, E_pmp_k, drain_rate});
            r -= phi_k;
            r += drain_jac.res;

            // --- Assemble tridiagonal from Jacobian structs ---
            a[k] = diff_jac.d_E_km1 + adv_jac.d_E_km1;
            c_arr[k] = diff_jac.d_E_kp1 + adv_jac.d_E_kp1;
            b[k] = RHO_I / dt
                 + diff_jac.d_E_k + adv_jac.d_E_k
                 + drain_jac.d_E_k
                 + horiz_adv_diag[k];

            rhs[k] = -r;
        }

        // --- Thomas algorithm (forward elimination) ---
        for (int k = 1; k < nz; k++) {
            if (fabsf(b[k-1]) < 1e-30f) continue;
            float w = a[k] / b[k-1];
            b[k]   -= w * c_arr[k-1];
            rhs[k] -= w * rhs[k-1];
        }

        // --- Back substitution ---
        float dE[MAX_NZ];
        dE[nz-1] = rhs[nz-1] / b[nz-1];
        for (int k = nz - 2; k >= 0; k--) {
            dE[k] = (rhs[k] - c_arr[k] * dE[k+1]) / b[k];
        }

        // --- Apply correction with relaxation ---
        for (int k = 0; k < nz; k++) {
            E_local[k] += relaxation * dE[k];
        }
    }

    // Write out the total correction delta_E = E_local - E_original
    for (int k = 0; k < nz; k++) {
        int ijk = (i * nx + j) * nz + k;
        delta_E[ijk] = E_local[k] - E[ijk];
    }
}


/* =========================================================
   Compute sigma_dot from 3D velocity field.

   h * sigma_dot = uz - ux*(db/dx + sigma*dh/dx)
                      - uy*(db/dy + sigma*dh/dy)
                      - sigma * dh/dt

   where dh/dt is approximated from mass conservation.
   ========================================================= */
extern "C" __global__
void compute_sigma_dot(
    float* __restrict__ sigma_dot,  // (ny, nx, nz) output
    const float* __restrict__ u3d,  // (nz, ny, nx+1)
    const float* __restrict__ v3d,  // (nz, ny+1, nx)
    const float* __restrict__ H,    // (ny, nx)
    const float* __restrict__ bed,  // (ny, nx)
    const float* __restrict__ smb,  // (ny, nx) surface mass balance
    const float* __restrict__ sigma,// (nz,)
    float dx, float dt,
    int ny, int nx, int nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_columns = ny * nx;
    if (idx >= n_columns) return;

    int i = idx / nx;
    int j = idx % nx;

    float h = H[i * nx + j];
    float h_inv = (h > 1e-3f) ? 1.0f / h : 0.0f;
    float dx_inv = 1.0f / dx;

    // Compute horizontal gradients of bed and thickness
    float bed_c = get_cell(bed, i, j, ny, nx);
    float bed_xp = get_cell(bed, i, j+1, ny, nx);
    float bed_xm = get_cell(bed, i, j-1, ny, nx);
    float bed_yp = get_cell(bed, i+1, j, ny, nx);
    float bed_ym = get_cell(bed, i-1, j, ny, nx);

    float db_dx = (bed_xp - bed_xm) * 0.5f * dx_inv;
    float db_dy = (bed_yp - bed_ym) * 0.5f * dx_inv;

    float H_xp = get_cell(H, i, j+1, ny, nx);
    float H_xm = get_cell(H, i, j-1, ny, nx);
    float H_yp = get_cell(H, i+1, j, ny, nx);
    float H_ym = get_cell(H, i-1, j, ny, nx);

    float dH_dx = (H_xp - H_xm) * 0.5f * dx_inv;
    float dH_dy = (H_yp - H_ym) * 0.5f * dx_inv;

    // Approximate dh/dt from mass conservation:
    // dh/dt = SMB - div(h*u_bar)
    // Use depth-averaged velocity (average over layers)
    // For now, use the layer 0 (bed) values as approximation
    // of div(h*u_bar) via finite differences on the flux
    float u_bar_left  = 0.0f;
    float u_bar_right = 0.0f;
    float v_bar_top   = 0.0f;
    float v_bar_bot   = 0.0f;
    float nz_inv = 1.0f / (float)nz;
    for (int k = 0; k < nz; k++) {
        u_bar_left  += get_u3d(u3d, k, i, j,   nz, ny, nx);
        u_bar_right += get_u3d(u3d, k, i, j+1, nz, ny, nx);
        v_bar_top   += get_v3d(v3d, k, i,   j, nz, ny, nx);
        v_bar_bot   += get_v3d(v3d, k, i+1, j, nz, ny, nx);
    }
    u_bar_left  *= nz_inv;
    u_bar_right *= nz_inv;
    v_bar_top   *= nz_inv;
    v_bar_bot   *= nz_inv;

    // Upwind flux divergence for dh/dt estimate
    float H_c = H[i * nx + j];
    float flux_x = u_bar_right * ((u_bar_right > 0.0f) ? H_c : H_xp)
                  - u_bar_left  * ((u_bar_left  > 0.0f) ? H_xm : H_c);
    float flux_y = v_bar_bot * ((v_bar_bot > 0.0f) ? H_c : H_yp)
                  - v_bar_top * ((v_bar_top > 0.0f) ? H_ym : H_c);

    float div_hU = (flux_x + flux_y) * dx_inv;
    float dh_dt = smb[i*nx+j] - div_hU;

    // Compute sigma_dot at each level by integrating incompressibility
    // from the bed. For generality, we compute it directly from the
    // definition rather than assuming SSA.
    for (int k = 0; k < nz; k++) {
        float sig_k = sigma[k];

        // Interpolate velocities to cell center at this layer
        float ux_k = 0.5f * (get_u3d(u3d, k, i, j, nz, ny, nx)
                            + get_u3d(u3d, k, i, j+1, nz, ny, nx));
        float uy_k = 0.5f * (get_v3d(v3d, k, i, j, nz, ny, nx)
                            + get_v3d(v3d, k, i+1, j, nz, ny, nx));

        // For uz: integrate incompressibility from bed
        // uz(sigma) = uz(0) - h * integral_0^sigma (dux/dx + duy/dy) dsigma'
        // For now, approximate uz from the kinematic relation:
        // At the bed: uz(0) = ux*db/dx + uy*db/dy (+ mb, neglected initially)
        // Then uz(sigma) = uz(0) - h*sigma*(dux/dx + duy/dy) for depth-independent u
        // This generalizes: for varying u(sigma), integrate numerically

        // Compute horizontal divergence at this layer
        float dux_dx_k = (get_u3d(u3d, k, i, j+1, nz, ny, nx)
                        - get_u3d(u3d, k, i, j,   nz, ny, nx)) * dx_inv;
        float duy_dy_k = (get_v3d(v3d, k, i+1, j, nz, ny, nx)
                        - get_v3d(v3d, k, i,   j, nz, ny, nx)) * dx_inv;

        // Simple trapezoidal integration of divergence from bed to sigma_k
        // For now, assume divergence is roughly constant with depth (SSA-like)
        float integrated_div = sig_k * (dux_dx_k + duy_dy_k);

        // uz at bed from kinematic BC (neglecting basal melt for now)
        float ux_bed = 0.5f * (get_u3d(u3d, 0, i, j, nz, ny, nx)
                              + get_u3d(u3d, 0, i, j+1, nz, ny, nx));
        float uy_bed = 0.5f * (get_v3d(v3d, 0, i, j, nz, ny, nx)
                              + get_v3d(v3d, 0, i+1, j, nz, ny, nx));
        float uz_bed = ux_bed * db_dx + uy_bed * db_dy;
        float uz_k = uz_bed - h * integrated_div;

        // sigma_dot from definition
        float h_sigma_dot = uz_k
            - ux_k * (db_dx + sig_k * dH_dx)
            - uy_k * (db_dy + sig_k * dH_dy)
            - sig_k * dh_dt;

        sigma_dot[(i*nx+j)*nz + k] = h_sigma_dot * h_inv;
    }
}


/* =========================================================
   Restrict enthalpy from fine to coarse grid.
   Operates horizontally (averages 4 fine cells),
   keeps vertical dimension unchanged.
   ========================================================= */
extern "C" __global__
void restrict_enthalpy(
    const float* __restrict__ E_fine,   // (ny_fine, nx_fine, nz)
    float* __restrict__ E_coarse,       // (ny_coarse, nx_coarse, nz)
    int ny_coarse, int nx_coarse, int nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = ny_coarse * nx_coarse * nz;
    if (idx >= total) return;

    int tmp = idx / nz;
    int k = idx % nz;
    int i_c = tmp / nx_coarse;
    int j_c = tmp % nx_coarse;

    int nx_fine = 2 * nx_coarse;

    // Full-weighting: average 4 fine cells at each sigma level
    int i_f = 2 * i_c;
    int j_f = 2 * j_c;

    float val = 0.25f * (
        E_fine[(i_f     * nx_fine + j_f    ) * nz + k] +
        E_fine[(i_f     * nx_fine + j_f + 1) * nz + k] +
        E_fine[((i_f+1) * nx_fine + j_f    ) * nz + k] +
        E_fine[((i_f+1) * nx_fine + j_f + 1) * nz + k]);

    E_coarse[(i_c * nx_coarse + j_c) * nz + k] = val;
}


/* =========================================================
   Prolongate enthalpy from coarse to fine grid.
   Bilinear interpolation horizontally, identity vertically.
   ========================================================= */
extern "C" __global__
void prolongate_enthalpy(
    const float* __restrict__ E_coarse, // (ny_coarse, nx_coarse, nz)
    float* __restrict__ E_fine,         // (ny_fine, nx_fine, nz)
    int ny_fine, int nx_fine, int nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = ny_fine * nx_fine * nz;
    if (idx >= total) return;

    int tmp = idx / nz;
    int k = idx % nz;
    int i_f = tmp / nx_fine;
    int j_f = tmp % nx_fine;

    int ny_coarse = ny_fine / 2;
    int nx_coarse = nx_fine / 2;

    // Map fine cell center to coarse grid coordinates
    float I_float = (j_f - 0.5f) * 0.5f;
    float J_float = (i_f - 0.5f) * 0.5f;

    I_float = fmaxf(0.0f, fminf(I_float, (float)(nx_coarse - 1)));
    J_float = fmaxf(0.0f, fminf(J_float, (float)(ny_coarse - 1)));

    int I_lo = (int)I_float;
    int J_lo = (int)J_float;
    int I_hi = min(I_lo + 1, nx_coarse - 1);
    int J_hi = min(J_lo + 1, ny_coarse - 1);

    float t_x = I_float - I_lo;
    float t_y = J_float - J_lo;

    float v00 = E_coarse[(J_lo * nx_coarse + I_lo) * nz + k];
    float v01 = E_coarse[(J_lo * nx_coarse + I_hi) * nz + k];
    float v10 = E_coarse[(J_hi * nx_coarse + I_lo) * nz + k];
    float v11 = E_coarse[(J_hi * nx_coarse + I_hi) * nz + k];

    E_fine[idx] = (1.0f-t_y)*((1.0f-t_x)*v00 + t_x*v01)
                + t_y       *((1.0f-t_x)*v10 + t_x*v11);
}
