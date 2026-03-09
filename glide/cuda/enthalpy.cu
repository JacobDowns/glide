extern "C" __global__
void enthalpy_advection_diffusion_step(
    float* enthalpy_new,
    const float* enthalpy_prev,
    const float* thickness,
    const float* surface_enthalpy,
    const float* geothermal_flux,
    const float* sigma_velocity,
    const float* conductivity,
    const float* diffusivity,
    const float* diffusivity_eff,
    float dt,
    float dsigma,
    float thklim,
    int n_columns,
    int nz,
    float* c_prime,
    float* d_prime
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n_columns) {
        return;
    }

    // Each thread owns one independent vertical ice column. The 3D field is
    // flattened so that each column occupies a contiguous segment of length nz.
    int offset = col * nz;
    float H = thickness[col];
    float surface_value = surface_enthalpy[col];

    if (H <= thklim) {
        // Degenerate thin-ice case: collapse the column to the prescribed
        // surface value rather than solving an ill-conditioned diffusion system.
        for (int k = 0; k < nz; ++k) {
            enthalpy_new[offset + k] = surface_value;
            c_prime[offset + k] = 0.0f;
            d_prime[offset + k] = surface_value;
        }
        return;
    }

    float diff_basal = diffusivity[col];
    float cond = conductivity[col];
    float q_geo = geothermal_flux[col];
    float w_sigma = sigma_velocity[col];

    // Terrain-following form of vertical advection-diffusion:
    //   dE/dt + w_sigma dE/dsigma = (1 / H^2) d/dsigma( kappa_eff(E) dE/dsigma )
    // on sigma in [0, 1], with H fixed for this solve.
    float inv_h2 = 1.0f / (H * H);
    float diff_scale = dt * inv_h2 / (dsigma * dsigma);
    float adv = dt * w_sigma / dsigma;

    // Basal Neumann condition:
    //   -(k / H) dE/dsigma = q_geo
    // Rearranged into a ghost-cell contribution for the first row.
    float basal_grad_sigma = -H * q_geo / cond;
    float r_basal = diff_scale * diff_basal;
    float diff_plus = 0.5f * (diffusivity_eff[offset] + diffusivity_eff[offset + 1]);
    float r_plus = diff_scale * diff_plus;
    float basal_rhs = enthalpy_prev[offset] - r_basal * basal_grad_sigma * dsigma;

    // Forward sweep for the first row of the tridiagonal system.
    // This row incorporates the basal flux boundary condition. For upward
    // sigma advection (adv > 0), the upwind stencil touches the basal ghost
    // value, which contributes an additional known term to the RHS.
    float a = 0.0f;
    float b = 1.0f + r_plus;
    float c = -r_plus;

    if (adv >= 0.0f) {
        basal_rhs -= adv * basal_grad_sigma * dsigma;
    } else {
        b -= adv;
        c += adv;
    }

    float inv_b = 1.0f / b;

    c_prime[offset] = c * inv_b;
    d_prime[offset] = basal_rhs * inv_b;

    for (int k = 1; k < nz; ++k) {
        float rhs = enthalpy_prev[offset + k];
        float diff_minus = 0.5f * (diffusivity_eff[offset + k - 1] + diffusivity_eff[offset + k]);
        float r_minus = diff_scale * diff_minus;
        if (k == nz - 1) {
            // Surface Dirichlet condition: E = surface_value.
            // Eliminate the ghost/face value and move the known boundary term
            // to the right-hand side of the final row. For downward sigma
            // advection (adv < 0), the upwind stencil also uses this boundary.
            float r_top = diff_scale * diffusivity_eff[offset + k];
            a = -r_minus;
            b = 1.0f + r_minus + 2.0f * r_top;
            c = 0.0f;
            rhs += 2.0f * r_top * surface_value;
            if (adv >= 0.0f) {
                a -= adv;
                b += adv;
            } else {
                b -= adv;
                rhs -= adv * surface_value;
            }
        } else {
            // Interior backward-Euler advection-diffusion row with first-order
            // upwinding for the advective derivative.
            diff_plus = 0.5f * (diffusivity_eff[offset + k] + diffusivity_eff[offset + k + 1]);
            r_plus = diff_scale * diff_plus;
            a = -r_minus;
            b = 1.0f + r_minus + r_plus;
            c = -r_plus;

            if (adv >= 0.0f) {
                a -= adv;
                b += adv;
            } else {
                b -= adv;
                c += adv;
            }
        }

        // Standard Thomas-algorithm forward elimination.
        float denom = b - a * c_prime[offset + k - 1];
        float inv_denom = 1.0f / denom;
        c_prime[offset + k] = c * inv_denom;
        d_prime[offset + k] = (rhs - a * d_prime[offset + k - 1]) * inv_denom;
    }

    // Back substitution recovers the updated enthalpy profile.
    enthalpy_new[offset + nz - 1] = d_prime[offset + nz - 1];
    for (int k = nz - 2; k >= 0; --k) {
        enthalpy_new[offset + k] =
            d_prime[offset + k] - c_prime[offset + k] * enthalpy_new[offset + k + 1];
    }
}
