"""Finite-difference checks for the regularized-Coulomb law (sliding_law=1):
d(J)/d(u_c) per-cell field, and d(J)/d(beta=tau_max). Mirrors grad_test.py."""
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

tau_max = 150.0 + 80.0 * cp.sin(2 * cp.pi * X / L) * cp.sin(2 * cp.pi * Y / L)   # = beta slot
u_c0 = 40.0 + 15.0 * cp.cos(2 * cp.pi * X / L)
B = cp.ones((ny, nx), dtype=cp.float32); B.fill((1e-16 ** -(1. / 3)) / (917.0 * 9.81))

mg = Multigrid(6, ny=ny, nx=nx, dx=dx)
grid = mg.levels[0]
mg.geometry.bed.set(bed); mg.rheology.B.set(B)
mg.sliding.sliding_law.set(1.0)            # regularized Coulomb
mg.sliding.beta.set(tau_max); mg.sliding.u_c.set(u_c0); mg.sliding.u_reg.set(1.0)
mg.state.H.set(thk); mg.state.H_prev.set(thk)

solver = FASCDSolver(mg)
solver.vanka_options.omega.set(0.5)
solver.vanka_options.newton_options.relaxation.set(0.5)
solver.vanka_options.newton_options.steps.set(30)
solver.fas_options.coarsest_steps.set(200); solver.fas_options.pre_steps.set(10)
solver.fas_options.post_steps.set(50); solver.fas_options.finest_steps.set(150)
solver.fas_options.maximum_vcycles.set(10)
solver.fas_options.relative_tolerance.set(1e-3); solver.fas_options.absolute_tolerance.set(0.1)
solver.solve(dt)
u_obs = cp.array(grid.state.u.data); v_obs = cp.array(grid.state.v.data)

# linearization point: uniform tau_max
mg.sliding.beta.set(cp.ones_like(tau_max) * float(tau_max.mean()))
solver.solve(dt)
u = cp.array(grid.state.u.data); v = cp.array(grid.state.v.data); H = cp.array(grid.state.H.data)

grid.adjoint_operators.f_u[:, :] = -cp.sign(u - u_obs)
grid.adjoint_operators.f_v[:, :] = -cp.sign(v - v_obs)
adj = FASAdjointSolver(mg)
adj.fas_options.coarsest_steps.set(200); adj.fas_options.pre_steps.set(10)
adj.fas_options.post_steps.set(50); adj.fas_options.finest_steps.set(150)
adj.fas_options.maximum_vcycles.set(10)
adj.fas_options.absolute_tolerance.set(cp.float32(0.1)); adj.fas_options.relative_tolerance.set(cp.float32(1e-3))
adj.vanka_options.newton_options.ssa_damping.set(cp.float32(0.1)); adj.vanka_options.omega.set(cp.float32(0.5))
adj.solve(dt)

grid.adjoint_operators.compute_gradient_u_c()
grid.adjoint_operators.compute_gradient_beta()
grad_u_c = cp.array(grid.sliding.u_c.grad)
grad_beta = cp.array(grid.sliding.beta.grad)


def misfit():
    grid.state.u.data[:, :] = u; grid.state.v.data[:, :] = v; grid.state.H.data[:, :] = H
    solver.solve(dt)
    return float((abs(grid.state.u.data - u_obs)).sum() + (abs(grid.state.v.data - v_obs)).sum())


def fd_check(field_mgr, base, eps, name):
    pert = cp.random.randn(*base.shape, dtype=cp.float32)
    field_mgr.set(base + eps * pert); J1 = misfit()
    field_mgr.set(base - eps * pert); J0 = misfit()
    field_mgr.set(base)
    return (J1 - J0) / (2 * eps), pert


# d/d(u_c)
gfd, pert = fd_check(mg.sliding.u_c, cp.array(u_c0), cp.float32(1e-1), "u_c")
gad = float((grad_u_c * pert).sum())
rel_uc = abs(gfd - gad) / max(abs(gad), 1e-9)
print(f"d/d(u_c)   FD: {gfd:.6g}  Adj: {gad:.6g}  Rel.Err: {rel_uc:.4g}")

# d/d(beta=tau_max), at the uniform linearization point
gfd, pert = fd_check(mg.sliding.beta, cp.ones_like(tau_max) * float(tau_max.mean()),
                     cp.float32(1e-1), "beta")
gad = float((grad_beta * pert).sum())
rel_b = abs(gfd - gad) / max(abs(gad), 1e-9)
print(f"d/d(tau_max) FD: {gfd:.6g}  Adj: {gad:.6g}  Rel.Err: {rel_b:.4g}")

assert rel_uc < 5e-2 and rel_b < 5e-2, "Coulomb gradient check FAILED"
print("PASS: regularized-Coulomb d_u_c and d_tau_max match finite difference")
