// =====================================================================
// COMMON UTILITIES: DualFloat, array access helpers, LU solvers
// =====================================================================

// Compile-time stress scheme switch. GLIDE_MOLHO=1 (default) compiles the
// full two-field model; GLIDE_MOLHO=0 (passed as -DGLIDE_MOLHO=0 when
// grid.stress_scheme == 'ssa') compiles out all deformational physics and
// shrinks the Vanka patch solves to the 5 live dofs. The runtime `ssa`
// kernel flag remains the semantic source of truth (see the constraint
// convention below); the SSA build is its specialization and must produce
// the same results as a MOLHO build running with ssa=true.
#ifndef GLIDE_MOLHO
#define GLIDE_MOLHO 1
#endif

// Second compile-time switch, orthogonal to GLIDE_MOLHO. GLIDE_DIVA=1 (passed
// as -DGLIDE_DIVA=1 when grid.stress_scheme == 'diva') selects the depth-
// integrated viscosity approximation. It is only ever set together with
// GLIDE_MOLHO=0: DIVA reuses the SSA 2-field momentum system and 5-DOF patch,
// and differs only by reading two cell-local closure coefficients (the depth-
// averaged viscosity eta_bar and the effective drag beta_eff, produced by
// cuda/diva.cu) in place of the inline SSA viscosity and Weertman drag.
#ifndef GLIDE_DIVA
#define GLIDE_DIVA 0
#endif
struct DualFloat {
    float v; // Primal value
    float d; // Derivative/Perturbation component

    // Addition: (u + v, du + dv)
    __device__ __forceinline__ friend DualFloat operator+(DualFloat a, DualFloat b) {
        return {a.v + b.v, a.d + b.d};
    }

    // Subtraction: (u - v, du - dv)
    __device__ __forceinline__ friend DualFloat operator-(DualFloat a, DualFloat b) {
        return {a.v - b.v, a.d - b.d};
    }

    // Multiplication: (u * v, u * dv + v * du)
    __device__ __forceinline__ friend DualFloat operator*(DualFloat a, DualFloat b) {
        return {a.v * b.v, __fmaf_rn(a.v, b.d, a.d * b.v)};
    }

    // Multiplication by Scalar: (u * s, du * s)
    __device__ __forceinline__ friend DualFloat operator*(DualFloat a, float s) {
        return {a.v * s, a.d * s};
    }

    __device__ __forceinline__ friend DualFloat operator*(float s, DualFloat a) {
        return {a.v * s, a.d * s};
    }

    // Addition with Scalar: (u + s, du)
    __device__ __forceinline__ friend DualFloat operator+(DualFloat a, float s) {
	return {a.v + s, a.d};
    }

    // Commutative version: (s + u, du)
    __device__ __forceinline__ friend DualFloat operator+(float s, DualFloat a) {
	return {s + a.v, a.d};
    }

    // Subtraction with Scalar
    __device__ __forceinline__ friend DualFloat operator-(DualFloat a, float s) {
	return {a.v - s, a.d};
    }

    // Division by Scalar: (u / s, du / s)
    __device__ __forceinline__ friend DualFloat operator/(DualFloat a, float s) {
	float inv_s = 1.0f / s; // Compiler will likely use RCP
	return {a.v * inv_s, a.d * inv_s};
    }

    // Division: (u / v, (du*v - u*dv) / v^2)
    __device__ __forceinline__ friend DualFloat operator/(DualFloat a, DualFloat b) {
        float inv_b = 1.0f / b.v;
        return {a.v * inv_b, (a.d * b.v - a.v * b.d) * inv_b * inv_b};
    }

    // Scalar over dual: (s / v, -s*dv / v^2)
    __device__ __forceinline__ friend DualFloat operator/(float s, DualFloat a) {
        float inv_a = 1.0f / a.v;
        return {s * inv_a, -s * a.d * inv_a * inv_a};
    }

    // Scalar minus dual: (s - u, -du)
    __device__ __forceinline__ friend DualFloat operator-(float s, DualFloat a) {
        return {s - a.v, -a.d};
    }

    // Negation: (-u, -du)
    __device__ __forceinline__ friend DualFloat operator-(DualFloat a) {
        return {-a.v, -a.d};
    }

};

__device__ __forceinline__ DualFloat __powf(DualFloat u, float p) {
    // High-performance hardware intrinsic pow
    float val = __powf(u.v, p);

    // d/dx(u^p) = p * u^(p-1) * du
    // If u.v is zero, derivative is technically singular; eps_reg handles this.
    float deriv = p * __powf(u.v, p - 1.0f) * u.d;

    return {val, deriv};
}

// DualFloat math helpers used by the DIVA closure (diva.cu).  Each applies the plain-float
// intrinsic to the value and carries the derivative analytically; the plain-float calls resolve
// to the builtins (different signature), so there is no recursion.
__device__ __forceinline__ DualFloat fmaxf(DualFloat a, float s) {
    return a.v >= s ? a : DualFloat{s, 0.0f};
}

__device__ __forceinline__ DualFloat fminf(DualFloat a, float s) {
    return a.v <= s ? a : DualFloat{s, 0.0f};
}

__device__ __forceinline__ DualFloat fminf(DualFloat a, DualFloat b) {
    return a.v <= b.v ? a : b;
}

__device__ __forceinline__ DualFloat logf(DualFloat u) {
    // d/dx(log u) = du/u.  Needed for the d/dm parameter derivative of the Weertman power.
    return {logf(u.v), u.d / u.v};
}

__device__ __forceinline__ DualFloat __powf(DualFloat u, DualFloat p) {
    // Both base and exponent carry a perturbation: d(u^p) = u^p*((p/u)du + log(u)dp).
    float val = __powf(u.v, p.v);
    float deriv = val * (p.v * u.d / u.v + logf(u.v) * p.d);
    return {val, deriv};
}

__device__ __forceinline__ DualFloat sqrtf(DualFloat u) {
    // d/dx(sqrt u) = du/(2 sqrt u); callers regularize u.v away from 0 (eps_reg).
    float val = sqrtf(u.v);
    return {val, u.d / (2.0f * val)};
}

__device__ __forceinline__ float sigmoid(const float z, const float c) {
   float scaled_z = fminf(fmaxf(c*z,-20.0f),20.0f);
   return 1.0f/(1.0f + __expf(-scaled_z));
}

// Derivative of sigmoid w.r.t. z: d(sigmoid)/dz = c * sigmoid * (1 - sigmoid)
__device__ __forceinline__ float sigmoid_deriv(const float z, const float c) {
   float s = sigmoid(z, c);
   return c * s * (1.0f - s);
}


//__device__ __forceinline__ float get_grounded(const float H, const float bed, const float sigmoid_c) 
//{
//   float z = bed + 0.917f*H;
//   return sigmoid(z, sigmoid_c);
//}

//__device__ __forceinline__ float get_grounded(const float H, const float bed, const float sigmoid_c) 
//{
//   float depth = fmaxf(-bed,0.0f);
//   float z = 0.917f*H - depth;
//   return fmaxf( fminf(1.0f + sigmoid_c*z,0.99f),0.01f);
//}

__device__ __forceinline__ float get_grounded(const float H, const float depth, const float sigmoid_c, const float sigmoid_k) 
{
   float z = 0.917f*H - depth + sigmoid_k/sigmoid_c;
   return sigmoid(z,sigmoid_c);
}

__device__ __forceinline__ float get_vfacet(const float* __restrict__ u, int i, int j, int ny, int nx) {
    //if (i < 0 || i >= ny || j < 0 || j > nx) return 0.0f;
    i = max(min(i,ny - 1),0);
    j = max(min(j,nx),0);
    return u[i * (nx + 1) + j];
}

__device__ __forceinline__ DualFloat get_vfacet(const float* __restrict__ u, const float* __restrict__ du, int i, int j, int ny, int nx) {
    i = max(min(i,ny - 1),0);
    j = max(min(j,nx),0);
    int idx = i * (nx + 1) + j;
    return {u[idx],du[idx]};
}

__device__ __forceinline__ float get_hfacet(const float* __restrict__ v, int i, int j, int ny, int nx) {
    //if (i < 0 || i > ny || j < 0 || j >= nx) return 0.0f;
    i = max(min(i,ny),0);
    j = max(min(j,nx - 1),0);
    return v[i * nx + j];
}

__device__ __forceinline__ DualFloat get_hfacet(const float* __restrict__ v, const float* __restrict__ dv, int i, int j, int ny, int nx) {
    i = max(min(i,ny),0);
    j = max(min(j,nx - 1),0);
    int idx = i * nx + j;
    return {v[idx],dv[idx]};
}

__device__ __forceinline__ float get_cell(const float* __restrict__ arr, int i, int j, int ny, int nx) {
    //if (i < 0 || i >= ny || j < 0 || j >= nx) return 0.0f;
    i = max(min(i,ny - 1),0);
    j = max(min(j,nx - 1),0);
    return arr[i * nx + j];
}

__device__ __forceinline__ DualFloat get_cell(const float* __restrict__ arr, const float* __restrict__ darr, int i, int j, int ny, int nx) {
    i = max(min(i,ny - 1),0);
    j = max(min(j,nx - 1),0);
    int idx = i * nx + j;
    return {arr[idx],darr[idx]};
}

// Templated facet reads: pick the plain get_* for T=float (perturbation array ignored, may be
// nullptr) and the dual get_* for T=DualFloat.  A body templated on T (membrane_eps_sq, the DIVA
// closure) needs this to choose between the two get_* overloads, which differ only in return type.
template <typename T>
__device__ __forceinline__ T read_vfacet(const float* __restrict__ u, const float* __restrict__ du,
                                         int i, int j, int ny, int nx);
template <>
__device__ __forceinline__ float read_vfacet<float>(const float* __restrict__ u, const float* __restrict__ du,
                                         int i, int j, int ny, int nx) {
    return get_vfacet(u, i, j, ny, nx);
}
template <>
__device__ __forceinline__ DualFloat read_vfacet<DualFloat>(const float* __restrict__ u, const float* __restrict__ du,
                                         int i, int j, int ny, int nx) {
    return get_vfacet(u, du, i, j, ny, nx);
}

template <typename T>
__device__ __forceinline__ T read_hfacet(const float* __restrict__ v, const float* __restrict__ dv,
                                         int i, int j, int ny, int nx);
template <>
__device__ __forceinline__ float read_hfacet<float>(const float* __restrict__ v, const float* __restrict__ dv,
                                         int i, int j, int ny, int nx) {
    return get_hfacet(v, i, j, ny, nx);
}
template <>
__device__ __forceinline__ DualFloat read_hfacet<DualFloat>(const float* __restrict__ v, const float* __restrict__ dv,
                                         int i, int j, int ny, int nx) {
    return get_hfacet(v, dv, i, j, ny, nx);
}

/* =====================================================================
   CONSTRAINT CONVENTION (single source of truth)

   Constrained dofs are Dirichlet velocity facets (u/ud at j in {0,nx},
   v/vd at i in {0,ny}) and active-set thickness cells (mask = 1).
   In SSA mode (the ssa kernel flag / grid.stress_scheme == 'ssa'), EVERY
   ud/vd facet is additionally constrained to zero, which reduces the
   MOLHO momentum balance exactly to the SSA; all of the machinery below
   applies to those dofs unchanged.
   compute_residual defines the convention: constrained dofs have IDENTITY
   residual rows, R_c = x_c - x_bc (r_u = u, r_H = H - thklim), while all
   other rows retain their genuine stencil dependence on constrained dofs.
   Everything else follows verbatim:

   - compute_jvp is the exact derivative: constrained rows return the
     direction component; nothing else is masked.
   - compute_vjp is the exact transpose: the kernel computes the pure
     physics transpose, and the constrained-row structure (project
     multipliers off constrained rows, add the identity part lambda_c) is
     applied ONCE in the Python wrapper (operators.py, _launch_vjp).
   - Parameter gradient kernels project out constrained-row multipliers
     explicitly (dR_c/dp = 0), so they are correct for any lambda.
   - The Vanka patch solves are PRECONDITIONERS and deliberately deviate
     from the true Jacobian at constrained dofs: they use symmetric
     row+column elimination (unit diagonal). Identity-row-only patches
     are exact but unstable when a cell ENTERS the active set - the
     momentum rows then extrapolate the full H -> thklim collapse
     linearly through their d/dH columns, which blows up velocities at
     thin margins. Fixed points are unaffected: the smoother rhs is
     always the exact residual (never zeroed), so the forward smoother
     converges to R = 0 and the adjoint smoother converges lambda_c to
     its true multiplier equation.

   Under this convention lambda at constrained dofs is the constraint
   multiplier (not zero); no consumer may assume it vanishes.
   ===================================================================== */


