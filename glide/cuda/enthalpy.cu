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
   Compute the enthalpy residual at all (i, j, k) points.

   r_{i,j,k} = rho_i * (E - E_prev)/dt
             + rho_i * [horizontal advection]
             + rho_i * sigma_dot * dE/dsigma  (upwind)
             - (1/h^2) * d/dsigma(K dE/dsigma) (centered)
             - phi + rho_w * L * Dw(omega)

   Uses (ny, nx, nz) layout for E and r_E.
   ========================================================= */
extern "C" __global__
void enthalpy_compute_residual(
    float* __restrict__ r_E,            // (ny, nx, nz) residual output
    const float* __restrict__ E,        // (ny, nx, nz) current enthalpy
    const float* __restrict__ E_prev,   // (ny, nx, nz) previous time step
    const float* __restrict__ u3d,      // (nz, ny, nx+1) x-velocity per layer
    const float* __restrict__ v3d,      // (nz, ny+1, nx) y-velocity per layer
    const float* __restrict__ sigma_dot,// (ny, nx, nz) sigma velocity
    const float* __restrict__ H,        // (ny, nx) ice thickness
    const float* __restrict__ phi_strain,// (ny, nx, nz) strain heating
    const float* __restrict__ E_surface,// (ny, nx) surface enthalpy BC
    const float* __restrict__ Q_geo,    // (ny, nx) geothermal heat flux
    const float* __restrict__ Q_fh,     // (ny, nx) frictional heating
    const float* __restrict__ sigma,    // (nz,) sigma node positions
    float dx, float dt, float drain_rate,
    int ny, int nx, int nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_columns = ny * nx;
    if (idx >= n_columns) return;

    int i = idx / nx;
    int j = idx % nx;

    float h = H[i * nx + j];
    float h_inv = (h > 1e-3f) ? 1.0f / h : 0.0f;
    float h2_inv = h_inv * h_inv;
    float dx_inv = 1.0f / dx;

    // Surface BC: top layer (k = nz-1) is Dirichlet
    // Bed BC: bottom layer (k = 0) depends on thermal state

    for (int k = 0; k < nz; k++) {
        int ijk = (i * nx + j) * nz + k;

        // --- Surface Dirichlet BC ---
        if (k == nz - 1) {
            float E_s = E_surface[i * nx + j];
            r_E[ijk] = E[ijk] - E_s;
            continue;
        }

        float E_k = E[ijk];
        float sigma_k = sigma[k];
        float E_pmp_k = get_E_pmp(sigma_k, h);

        // --- Time derivative ---
        float r = RHO_I * (E_k - E_prev[ijk]) / dt;

        // --- Horizontal advection (upwind, layer-wise) ---
        // x-direction: flux at vertical facets
        float u_left  = get_u3d(u3d, k, i, j,   nz, ny, nx);
        float u_right = get_u3d(u3d, k, i, j+1, nz, ny, nx);

        float E_here = E_k;
        float E_xm   = get_E(E, i, j-1, k, ny, nx, nz);
        float E_xp   = get_E(E, i, j+1, k, ny, nx, nz);

        // Upwind fluxes
        float flux_left  = (u_left  > 0.0f) ? u_left  * E_xm   : u_left  * E_here;
        float flux_right = (u_right > 0.0f) ? u_right * E_here  : u_right * E_xp;

        // Apply boundary: no flux at domain edges
        if (j == 0)      flux_left  = 0.0f;
        if (j == nx - 1)  flux_right = 0.0f;

        r += RHO_I * (flux_right - flux_left) * dx_inv;

        // y-direction: flux at horizontal facets
        float v_top    = get_v3d(v3d, k, i,   j, nz, ny, nx);
        float v_bottom = get_v3d(v3d, k, i+1, j, nz, ny, nx);

        float E_ym = get_E(E, i-1, j, k, ny, nx, nz);
        float E_yp = get_E(E, i+1, j, k, ny, nx, nz);

        float flux_top    = (v_top    > 0.0f) ? v_top    * E_ym   : v_top    * E_here;
        float flux_bottom = (v_bottom > 0.0f) ? v_bottom * E_here : v_bottom * E_yp;

        if (i == 0)       flux_top    = 0.0f;
        if (i == ny - 1)  flux_bottom = 0.0f;

        r += RHO_I * (flux_bottom - flux_top) * dx_inv;

        // --- Vertical advection (upwind) ---
        float sd_k = sigma_dot[ijk];

        if (k == 0) {
            // One-sided: can only look up
            float dsig_p = sigma[1] - sigma[0];
            float E_kp1 = get_E(E, i, j, 1, ny, nx, nz);
            if (sd_k > 0.0f) {
                // Cannot upwind from below the bed, use zero gradient
                r += 0.0f;
            } else {
                r += RHO_I * sd_k * (E_kp1 - E_k) / dsig_p;
            }
        } else if (k == nz - 2) {
            // Next to surface Dirichlet: k+1 = nz-1 has known E_s
            float dsig_m = sigma[k] - sigma[k-1];
            float dsig_p = sigma[k+1] - sigma[k];
            float E_km1 = get_E(E, i, j, k-1, ny, nx, nz);
            float E_kp1 = E_surface[i * nx + j]; // Dirichlet value

            float sd_pos = fmaxf(sd_k, 0.0f);
            float sd_neg = fminf(sd_k, 0.0f);
            r += RHO_I * (sd_pos * (E_k - E_km1) / dsig_m
                        + sd_neg * (E_kp1 - E_k) / dsig_p);
        } else {
            // Interior
            float dsig_m = sigma[k] - sigma[k-1];
            float dsig_p = sigma[k+1] - sigma[k];
            float E_km1 = get_E(E, i, j, k-1, ny, nx, nz);
            float E_kp1 = get_E(E, i, j, k+1, ny, nx, nz);

            float sd_pos = fmaxf(sd_k, 0.0f);
            float sd_neg = fminf(sd_k, 0.0f);
            r += RHO_I * (sd_pos * (E_k - E_km1) / dsig_m
                        + sd_neg * (E_kp1 - E_k) / dsig_p);
        }

        // --- Vertical diffusion (centered) ---
        if (k == 0) {
            // Bed boundary: check cold vs temperate
            float E_pmp_bed = get_E_pmp(0.0f, h);
            if (E_k < E_pmp_bed) {
                // Cold base, Neumann: -K/h * dE/dsigma = Q_geo + Q_fh
                // Discretize: K/h * (E_1 - E_0)/dsig_0^+ = Q_geo + Q_fh
                // Residual contribution: -K/(h^2) * (E_1 - E_0)/dsig^+ + (Q_geo+Q_fh)/h
                float dsig_p = sigma[1] - sigma[0];
                float K_half = get_K(0.5f * (E_k + get_E(E, i, j, 1, ny, nx, nz)),
                                     0.5f * (E_pmp_k + get_E_pmp(sigma[1], h)));
                float E_kp1 = get_E(E, i, j, 1, ny, nx, nz);
                r -= h2_inv * K_half * (E_kp1 - E_k) / dsig_p;
                r -= (Q_geo[i*nx+j] + Q_fh[i*nx+j]) * h_inv;
            } else {
                // Temperate base: Dirichlet E = E_pmp
                r = E_k - E_pmp_bed;
            }
        } else {
            // Interior diffusion (including k = nz-2)
            float dsig_m = sigma[k] - sigma[k-1];
            float dsig_p = sigma[k+1] - sigma[k];
            float dsig_avg = 0.5f * (dsig_m + dsig_p);

            float E_km1 = (k > 0) ? get_E(E, i, j, k-1, ny, nx, nz) : E_k;
            float E_kp1 = (k < nz-1) ? get_E(E, i, j, k+1, ny, nx, nz)
                                      : E_surface[i*nx+j];

            float E_pmp_km1 = get_E_pmp(sigma[max(k-1,0)], h);
            float E_pmp_kp1 = get_E_pmp(sigma[min(k+1,nz-1)], h);

            float K_upper = get_K(0.5f*(E_k + E_kp1), 0.5f*(E_pmp_k + E_pmp_kp1));
            float K_lower = get_K(0.5f*(E_k + E_km1), 0.5f*(E_pmp_k + E_pmp_km1));

            r -= h2_inv / dsig_avg * (K_upper * (E_kp1 - E_k) / dsig_p
                                    - K_lower * (E_k - E_km1) / dsig_m);
        }

        // --- Source terms ---
        float phi_k = phi_strain[ijk];
        float omega_k = get_omega(E_k, E_pmp_k);
        r -= phi_k;
        r += RHO_W * L_HEAT * get_drainage(omega_k, drain_rate);

        r_E[ijk] = r;
    }
}


/* =========================================================
   Column-wise Newton/Thomas smoother.

   For each column (i,j), freezes horizontal neighbors and
   solves the vertical tridiagonal system using the Thomas
   algorithm. Newton iteration handles the K(E) nonlinearity.

   This is the enthalpy analogue of the Vanka smoother.
   ========================================================= */
extern "C" __global__
void enthalpy_column_smooth(
    float* __restrict__ delta_E,        // (ny, nx, nz) correction output
    const float* __restrict__ E,        // (ny, nx, nz) current enthalpy
    const float* __restrict__ E_prev,   // (ny, nx, nz) previous time step
    const float* __restrict__ u3d,      // (nz, ny, nx+1) x-velocity
    const float* __restrict__ v3d,      // (nz, ny+1, nx) y-velocity
    const float* __restrict__ sigma_dot,// (ny, nx, nz) sigma velocity
    const float* __restrict__ H,        // (ny, nx) thickness
    const float* __restrict__ phi_strain,// (ny, nx, nz) strain heating
    const float* __restrict__ E_surface,// (ny, nx) surface BC
    const float* __restrict__ Q_geo,    // (ny, nx) geothermal heat flux
    const float* __restrict__ Q_fh,     // (ny, nx) frictional heating
    const float* __restrict__ sigma,    // (nz,) sigma positions
    float dx, float dt, float drain_rate,
    int ny, int nx, int nz,
    int n_newton, float relaxation)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_columns = ny * nx;
    if (idx >= n_columns) return;

    int i = idx / nx;
    int j = idx % nx;

    float h = H[i * nx + j];
    float h_inv = (h > 1e-3f) ? 1.0f / h : 0.0f;
    float h2_inv = h_inv * h_inv;
    float dx_inv = 1.0f / dx;

    // Local copy of column enthalpy for Newton iteration
    float E_local[MAX_NZ];
    for (int k = 0; k < nz; k++) {
        E_local[k] = E[(i * nx + j) * nz + k];
    }

    // Precompute frozen horizontal advection contributions per layer
    float horiz_adv[MAX_NZ];
    for (int k = 0; k < nz; k++) {
        float E_here = E_local[k];

        // x-direction
        float u_left  = get_u3d(u3d, k, i, j,   nz, ny, nx);
        float u_right = get_u3d(u3d, k, i, j+1, nz, ny, nx);
        float E_xm = get_E(E, i, j-1, k, ny, nx, nz);
        float E_xp = get_E(E, i, j+1, k, ny, nx, nz);

        float fl = (u_left  > 0.0f) ? u_left  * E_xm  : u_left  * E_here;
        float fr = (u_right > 0.0f) ? u_right * E_here : u_right * E_xp;
        if (j == 0)      fl = 0.0f;
        if (j == nx - 1) fr = 0.0f;

        // y-direction
        float v_top    = get_v3d(v3d, k, i,   j, nz, ny, nx);
        float v_bottom = get_v3d(v3d, k, i+1, j, nz, ny, nx);
        float E_ym = get_E(E, i-1, j, k, ny, nx, nz);
        float E_yp = get_E(E, i+1, j, k, ny, nx, nz);

        float ft = (v_top    > 0.0f) ? v_top    * E_ym  : v_top    * E_here;
        float fb = (v_bottom > 0.0f) ? v_bottom * E_here : v_bottom * E_yp;
        if (i == 0)       ft = 0.0f;
        if (i == ny - 1)  fb = 0.0f;

        // Note: the self-advection terms (where E_here appears in flux)
        // contribute to the diagonal of the column Jacobian, but we
        // handle those through the residual + Jacobian assembly below.
        // Here we store the full horizontal advection for the residual.
        horiz_adv[k] = RHO_I * ((fr - fl) + (fb - ft)) * dx_inv;
    }

    // --- Newton iteration ---
    for (int newton = 0; newton < n_newton; newton++) {

        // Build tridiagonal: a[k]*dE[k-1] + b[k]*dE[k] + c[k]*dE[k+1] = -r[k]
        float a[MAX_NZ], b[MAX_NZ], c[MAX_NZ], rhs[MAX_NZ];

        float E_s = E_surface[i * nx + j];

        for (int k = 0; k < nz; k++) {
            a[k] = 0.0f;
            b[k] = 0.0f;
            c[k] = 0.0f;
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

            if (E_k < E_pmp_bed) {
                // Cold base, Neumann BC
                float dsig_p = sigma[1] - sigma[0];
                float E_kp1 = E_local[1];
                float E_pmp_1 = get_E_pmp(sigma[1], h);

                float K_half = get_K(0.5f*(E_k + E_kp1),
                                     0.5f*(E_pmp_bed + E_pmp_1));

                // Residual
                float r = RHO_I * (E_k - E_prev[(i*nx+j)*nz]) / dt;
                r += horiz_adv[0];

                // Vertical advection at bed
                float sd_k = sigma_dot[(i*nx+j)*nz];
                if (sd_k < 0.0f) {
                    r += RHO_I * sd_k * (E_kp1 - E_k) / dsig_p;
                }

                // Diffusion (Neumann)
                r -= h2_inv * K_half * (E_kp1 - E_k) / dsig_p;
                r -= (Q_geo[i*nx+j] + Q_fh[i*nx+j]) * h_inv;

                // Source
                float phi_k = phi_strain[(i*nx+j)*nz];
                float omega_k = get_omega(E_k, E_pmp_bed);
                r -= phi_k;
                r += RHO_W * L_HEAT * get_drainage(omega_k, drain_rate);

                // Jacobian entries (linearized)
                b[0] = RHO_I / dt + h2_inv * K_half / dsig_p;
                c[0] = -h2_inv * K_half / dsig_p;

                if (sd_k < 0.0f) {
                    b[0] += -RHO_I * sd_k / dsig_p;
                    c[0] +=  RHO_I * sd_k / dsig_p;
                }

                rhs[0] = -r;
            } else {
                // Temperate base: Dirichlet
                b[0] = 1.0f;
                rhs[0] = -(E_k - E_pmp_bed);
            }
        }

        // Interior layers: k = 1 .. nz-2
        for (int k = 1; k < nz - 1; k++) {
            float E_k = E_local[k];
            float sigma_k = sigma[k];
            float E_pmp_k = get_E_pmp(sigma_k, h);

            float dsig_m = sigma[k] - sigma[k-1];
            float dsig_p = sigma[k+1] - sigma[k];
            float dsig_avg = 0.5f * (dsig_m + dsig_p);

            float E_km1 = E_local[k-1];
            float E_kp1 = (k < nz - 1) ? E_local[k+1] : E_s;

            // Diffusivities at half-points
            float E_pmp_km1 = get_E_pmp(sigma[k-1], h);
            float E_pmp_kp1 = get_E_pmp(sigma[min(k+1,nz-1)], h);
            float K_upper = get_K(0.5f*(E_k + E_kp1), 0.5f*(E_pmp_k + E_pmp_kp1));
            float K_lower = get_K(0.5f*(E_k + E_km1), 0.5f*(E_pmp_k + E_pmp_km1));

            // Sigma velocity
            float sd_k = sigma_dot[(i*nx+j)*nz + k];
            float sd_pos = fmaxf(sd_k, 0.0f);
            float sd_neg = fminf(sd_k, 0.0f);

            // --- Residual ---
            float r = RHO_I * (E_k - E_prev[(i*nx+j)*nz + k]) / dt;
            r += horiz_adv[k];
            r += RHO_I * (sd_pos * (E_k - E_km1) / dsig_m
                        + sd_neg * (E_kp1 - E_k) / dsig_p);
            r -= h2_inv / dsig_avg * (K_upper * (E_kp1 - E_k) / dsig_p
                                    - K_lower * (E_k - E_km1) / dsig_m);

            float phi_k = phi_strain[(i*nx+j)*nz + k];
            float omega_k = get_omega(E_k, E_pmp_k);
            r -= phi_k;
            r += RHO_W * L_HEAT * get_drainage(omega_k, drain_rate);

            // --- Jacobian (tridiagonal entries) ---
            // Diffusion contribution
            float diff_lower = h2_inv * K_lower / (dsig_avg * dsig_m);
            float diff_upper = h2_inv * K_upper / (dsig_avg * dsig_p);

            a[k] = -diff_lower - RHO_I * sd_pos / dsig_m;
            c[k] = -diff_upper + RHO_I * sd_neg / dsig_p;
            b[k] = RHO_I / dt + diff_lower + diff_upper
                   + RHO_I * sd_pos / dsig_m - RHO_I * sd_neg / dsig_p;

            // Drainage Jacobian (d/dE of rho_w * L * drain_rate * omega)
            if (E_k > E_pmp_k) {
                b[k] += RHO_W * drain_rate;  // d(L*Dw)/dE = drain_rate * rho_w when temperate
            }

            rhs[k] = -r;
        }

        // --- Thomas algorithm (forward elimination) ---
        for (int k = 1; k < nz; k++) {
            if (fabsf(b[k-1]) < 1e-30f) continue;
            float w = a[k] / b[k-1];
            b[k]   -= w * c[k-1];
            rhs[k] -= w * rhs[k-1];
        }

        // --- Back substitution ---
        float dE[MAX_NZ];
        dE[nz-1] = rhs[nz-1] / b[nz-1];
        for (int k = nz - 2; k >= 0; k--) {
            dE[k] = (rhs[k] - c[k] * dE[k+1]) / b[k];
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
