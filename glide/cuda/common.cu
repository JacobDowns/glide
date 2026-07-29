// =====================================================================
// COMMON UTILITIES: DualFloat, array access helpers, LU solvers
// =====================================================================
//
// HOW DERIVATIVES WORK IN THIS CODEBASE.  Every term of the residual is packaged the same
// way.  The pattern is worth learning once, because it repeats in viscosity.cu, stress.cu
// and flux.cu without variation:
//
//   struct XStencil      the term's inputs, named by position (_l, _r, _t, _b, _c)
//   struct XStencilDual  the same inputs as DualFloats, with get_primals()/get_diffs()
//   struct XJacobian     the value in .res, one partial per input in .d_<input>, and
//                        apply_jvp(dot) to contract those partials with a direction
//   get_x_jac(s)         computes .res and every partial, by hand
//   get_x_dual(s)        get_x_jac on the primals, then apply_jvp on the diffs
//
// One hand-derived function therefore serves all four consumers: the residual reads .res,
// the JVP contracts the partials forwards, the VJP scatters them transposed, and the Vanka
// smoother picks out the few entries its local block needs.  That is why adding a term
// costs one get_x_jac and nothing else.
//
// IMPORTANT: `_dual` in a name means "returns a DualFloat", NOT "differentiated
// automatically".  DualFloat is genuine forward-mode arithmetic, but only a few functions
// actually propagate through it -- populate_viscosity below and diva_coeffs_cell in diva.cu
// are the notable ones.  The get_*_dual functions are hand-derived Jacobians wearing a dual
// interface.  Change a get_x_jac and you must change its partials by hand; nothing derives
// them for you, and no test will notice a partial that is merely wrong rather than absent
// except the dot-product and FD checks in tests/.
//
// INDEX CONVENTION: get_cell / get_vfacet / get_hfacet CLAMP out-of-range indices to the
// nearest valid one -- they do NOT return zero (see the commented-out early returns).
// Stencils may therefore read past the boundary and will see the edge value, i.e. a
// zero-gradient extension rather than a hole.  Where a true zero is needed the caller
// multiplies by an explicit mask; the *_mask factors in populate_viscosity are that.
// A value paired with its derivative along ONE seeded direction.  Arithmetic on these
// carries (value, derivative) together by the chain rule, so an expression written once
// yields both.  Forward mode only: one direction per pass, and the direction is fixed by
// whatever the .d fields are seeded with at the leaves.
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

// Clamps.  These are not differentiable where the two arguments cross, so the
// convention is the one-sided derivative of whichever branch is selected: the dual
// propagates through when the dual argument wins, and the derivative is zero when the
// constant does.  Callers must keep that in mind -- a clamp that is active at the
// solution contributes no sensitivity.
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
    // d/dx(log u) = du/u.  Needed for parameter derivatives w.r.t. an exponent, e.g.
    // d/dm (x)^((m-1)/2) = (x)^((m-1)/2) * 0.5*log(x), which __powf(dual,float)
    // cannot supply because its exponent is a plain float.
    return {logf(u.v), u.d / u.v};
}

__device__ __forceinline__ DualFloat __powf(DualFloat u, float p) {
    // High-performance hardware intrinsic pow
    float val = __powf(u.v, p);

    // d/dx(u^p) = p * u^(p-1) * du
    // If u.v is zero, derivative is technically singular; eps_reg handles this.
    float deriv = p * __powf(u.v, p - 1.0f) * u.d;

    return {val, deriv};
}

__device__ __forceinline__ DualFloat __powf(DualFloat u, DualFloat p) {
    // Both base and exponent carry a perturbation:
    //     d(u^p) = u^p * ( (p/u) du + log(u) dp )
    // Needed when the exponent is itself a parameter being differentiated, as for the
    // Weertman m.  Reduces exactly to the (dual, float) overload above when dp = 0.
    float val = __powf(u.v, p.v);

    float deriv = val * (p.v * u.d / u.v + logf(u.v) * p.d);

    return {val, deriv};
}

__device__ __forceinline__ DualFloat sqrtf(DualFloat u) {
    // Hardware intrinsic sqrt; d/dx(sqrt(u)) = du / (2*sqrt(u))
    // If u.v is zero the derivative is singular; callers regularize (e.g. eps_reg).
    float val = sqrtf(u.v);

    return {val, u.d / (2.0f * val)};
}

// Flotation is a hard inequality in the physics, but a hard switch would make the
// residual non-differentiable exactly where the adjoint needs it most (the grounding line).
// So it is smoothed: `c` sets the width, and the argument is clamped to +/-20 to keep
// __expf in range.
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

// Grounded fraction in [0,1]: 1 where the ice rests on the bed, 0 where it floats, with a
// smooth transition of width ~1/sigmoid_c across flotation.  z is the flotation excess,
// 0.917*H - depth, offset by sigmoid_k/sigmoid_c so the switch can be biased.  Everything
// downstream multiplies basal drag by this, which is how the drag turns off under shelves.
__device__ __forceinline__ float get_grounded(const float H, const float depth, const float sigmoid_c, const float sigmoid_k) 
{
   float z = 0.917f*H - depth + sigmoid_k/sigmoid_c;
   return sigmoid(z,sigmoid_c);
}

// Staggered-grid accessors.  Three grids, all row-major and all clamped at the edges:
//   cells    (ny,   nx  )   H, beta, eta, phi, ...
//   vfacets  (ny,   nx+1)   u, on the vertical (left/right) cell faces
//   hfacets  (ny+1, nx  )   v, on the horizontal (top/bottom) cell faces
// The dual overloads read the same index out of a second array, which is how a perturbation
// direction is threaded into a stencil without changing the stencil's shape.
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

// Cell read that is zeroed where the mask is set -- used to drop contributions from cells
// whose row has been replaced by an algebraic constraint.
__device__ __forceinline__ float get_masked_cell(const float* __restrict__ arr, const float* __restrict__ mask, int i, int j, int ny, int nx) {
    i = max(min(i,ny - 1),0);
    j = max(min(j,nx - 1),0);
    int idx = i * nx + j;
    return arr[idx]*(1.0f - mask[idx]);
}


