extern "C" __global__
void enthalpy_diffusion_step(
    float* enthalpy_new,
    const float* enthalpy_prev,
    const float* thickness,
    const float* surface_enthalpy,
    const float* geothermal_flux,
    const float* conductivity,
    const float* diffusivity,
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

    float diff = diffusivity[col];
    float cond = conductivity[col];
    float q_geo = geothermal_flux[col];

    // Terrain-following form of vertical diffusion:
    //   dE/dt = (kappa / H^2) d^2E/dsigma^2
    // on sigma in [0, 1], with H fixed for this solve.
    // Backward Euler gives a linear tridiagonal system with
    //   r = dt * kappa / (H^2 * dsigma^2).
    float inv_h2 = 1.0f / (H * H);
    float r = dt * diff * inv_h2 / (dsigma * dsigma);

    // Basal Neumann condition:
    //   -(k / H) dE/dsigma = q_geo
    // Rearranged into a ghost-cell contribution for the first row.
    float basal_grad_sigma = -H * q_geo / cond;
    float basal_rhs = enthalpy_prev[offset] - r * basal_grad_sigma * dsigma;

    // Forward sweep for the first row of the tridiagonal system.
    // This row incorporates the basal flux boundary condition.
    float a = 0.0f;
    float b = 1.0f + r;
    float c = -r;
    float inv_b = 1.0f / b;

    c_prime[offset] = c * inv_b;
    d_prime[offset] = basal_rhs * inv_b;

    for (int k = 1; k < nz; ++k) {
        float rhs = enthalpy_prev[offset + k];
        if (k == nz - 1) {
            // Surface Dirichlet condition: E = surface_value.
            // Eliminate the ghost/face value and move the known boundary term
            // to the right-hand side of the final row.
            a = -r;
            b = 1.0f + 3.0f * r;
            c = 0.0f;
            rhs += 2.0f * r * surface_value;
        } else {
            // Interior backward-Euler diffusion row.
            a = -r;
            b = 1.0f + 2.0f * r;
            c = -r;
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
