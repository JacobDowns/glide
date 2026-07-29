// =====================================================================
// COMMON UTILITIES: DualFloat, array access helpers, LU solvers
// =====================================================================
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

__device__ __forceinline__ float get_masked_cell(const float* __restrict__ arr, const float* __restrict__ mask, int i, int j, int ny, int nx) {
    i = max(min(i,ny - 1),0);
    j = max(min(j,nx - 1),0);
    int idx = i * nx + j;
    return arr[idx]*(1.0f - mask[idx]);
}


