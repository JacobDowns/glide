"""Finite-difference check for the new dJ/dm (global Weertman exponent) gradient.
Mirrors tests/grad_test.py's beta check, but for the scalar m."""
import cupy as cp

from glide.multigrid import Multigrid, FASCDSolver, FASAdjointSolver

L = 20000.0
dt = cp.float32(1.0)
base_res, y_factr, x_factr = 64, 7, 7
ny, nx = base_res * y_factr, base_res * x_factr
x = cp.linspace(0, x_factr * L, nx, dtype=cp.float32)
y = cp.linspace(0, y_factr * L, ny, dtype=cp.float32)
dx = (x[1] - x[0]).item()
X, Y = cp.meshgrid(x, y)

srf = 1000.0 * cp.ones((ny, nx), dtype=cp.float32) - cp.tan(cp.deg2rad(0.1)) * X + 10000
bed = srf - 1000
thk = srf - bed
rho_i, g = cp.float32(917.0), cp.float32(9.81)
beta = (1000 * cp.sin(2 * cp.pi * X / L) * cp.sin(2 * cp.pi * Y / L) + 1000) / (rho_i * g)
B = cp.ones((ny, nx), dtype=cp.float32); B.fill((1e-16 ** -(1. / 3)) / (rho_i * g))

M0 = cp.float32(0.6)   # nontrivial exponent so the rate-dependence is exercised

mg = Multigrid(6, ny=ny, nx=nx, dx=dx)
grid = mg.levels[0]
mg.geometry.bed.set(bed); mg.rheology.B.set(B); mg.sliding.beta.set(beta)
mg.sliding.m.set(M0); mg.sliding.u_reg.set(1.0)
mg.state.H.set(thk); mg.state.H_prev.set(thk)


def make_solver():
    s = FASCDSolver(mg)
    s.vanka_options.omega.set(0.5)
    s.vanka_options.newton_options.relaxation.set(0.5)
    s.vanka_options.newton_options.steps.set(30)
    s.fas_options.coarsest_steps.set(200); s.fas_options.pre_steps.set(10)
    s.fas_options.post_steps.set(50); s.fas_options.finest_steps.set(150)
    s.fas_options.maximum_vcycles.set(10)
    s.fas_options.relative_tolerance.set(1e-3); s.fas_options.absolute_tolerance.set(0.1)
    return s


solver = make_solver()
solver.solve(dt)
u_obs = cp.array(grid.state.u.data); v_obs = cp.array(grid.state.v.data)

# Linearization point: uniform beta (so there's a real misfit to differentiate).
mg.sliding.beta.set(cp.ones_like(beta) * beta.mean())
solver.solve(dt)
u = cp.array(grid.state.u.data); v = cp.array(grid.state.v.data); H = cp.array(grid.state.H.data)

dJdu = cp.sign(u - u_obs); dJdv = cp.sign(v - v_obs)
grid.adjoint_operators.f_u[:, :] = -dJdu
grid.adjoint_operators.f_v[:, :] = -dJdv

adj = FASAdjointSolver(mg)
adj.fas_options.coarsest_steps.set(200); adj.fas_options.pre_steps.set(10)
adj.fas_options.post_steps.set(50); adj.fas_options.finest_steps.set(150)
adj.fas_options.maximum_vcycles.set(10)
adj.fas_options.absolute_tolerance.set(cp.float32(0.1))
adj.fas_options.relative_tolerance.set(cp.float32(1e-3))
adj.vanka_options.newton_options.ssa_damping.set(cp.float32(0.1))
adj.vanka_options.omega.set(cp.float32(0.5))
adj.solve(dt)

grid.adjoint_operators.compute_gradient_m()
gm_ad = float(grid.sliding.m.grad)


def misfit_at(m_val):
    mg.sliding.m.set(cp.float32(m_val))
    grid.state.u.data[:, :] = u; grid.state.v.data[:, :] = v; grid.state.H.data[:, :] = H
    solver.solve(dt)
    return float((abs(grid.state.u.data - u_obs)).sum() + (abs(grid.state.v.data - v_obs)).sum())


eps = 1e-3
J_plus = misfit_at(float(M0) + eps)
J_minus = misfit_at(float(M0) - eps)
gm_fd = (J_plus - J_minus) / (2 * eps)

rel_err = abs(gm_fd - gm_ad) / abs(gm_ad)
print(f"dJ/dm  FD: {gm_fd:.6g}   Adjoint: {gm_ad:.6g}   Rel.Err: {rel_err:.4g}")
assert rel_err < 5e-2, "dJ/dm gradient check FAILED"
print("PASS: dJ/dm matches finite difference")
