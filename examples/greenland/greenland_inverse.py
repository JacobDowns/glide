
"""
Greenland inverse simulation example.

Run interactively or execute as a script. Modify the paths and parameters
below to match your setup.
"""
import cupy as cp
import numpy as np

#from glide import IcePhysics
from glide.io import VTIWriter, write_vti
from glide.data import load_greenland_preprocessed

from glide.model import IceDynamics
from glide.multigrid import Multigrid, FASCDSolver, FASAdjointSolver
from scipy.ndimage import gaussian_filter
from glide.hooks import InverseLogger

### Load a dataset (here a preprocessed greenland dataset)
dataset = load_greenland_preprocessed()

### Initialize grid
# ny and nx must both divide by 2^(n_levels - 1) cleanly!
ny,nx,dx = dataset.ny,dataset.nx,dataset.dx
model = IceDynamics(n_levels=6,ny=ny,nx=nx,dx=dx)
mg = model.mg

grid = mg.levels[0]
dt = cp.float32(10.0)

### Initialize state
thk = gaussian_filter(dataset.thickness.values,1)
mg.state.H.set(thk)
mg.state.H_prev.set(thk)

### Initialize geometry
bed = gaussian_filter(dataset.bed.values,1)
mg.geometry.bed.set(bed)
mg.geometry.flotation_reg_driving.set(0.1)

### Initialize rheology
# Compute B (rate factor - we measure driving stress in units of head, so the rho g factor gets subsumed into definitions of beta and B!)
B = cp.zeros((ny,nx), dtype=cp.float32)
B.fill(1e-17 ** (-1.0 / 3.0) / (917 * 9.81)) 
mg.rheology.B.set(B)
mg.rheology.eps_reg.set(1e-6)
mg.rheology.n.set(3.0)

### Initialize sliding
BETA_PATH = None
#BETA_PATH = "./inverse_output/beta_level_0.p"
if BETA_PATH:
    import pickle
    beta = cp.array(pickle.load(open(BETA_PATH, 'rb')))
else:
    beta = cp.zeros((ny,nx), dtype=cp.float32)
    beta.fill(2.5)

mg.sliding.beta.set(beta)
mg.sliding.m.set(1./3.)
mg.sliding.water_drag.set(1e-4)

### Initialize calving
mg.calving.calving_rate.set(2000.0)

### Initialize forcing
smb = dataset.smb.values
#smb += -1.0
mg.forcing.smb.set(smb)

### Load velocity data ###
u_obs_cell = dataset.vx.values
v_obs_cell = dataset.vy.values

# Interpolate to faces
u_obs = cp.zeros((ny, nx + 1), dtype=cp.float32)
u_obs[:, 1:-1] = cp.array((u_obs_cell[:, 1:] + u_obs_cell[:, :-1]) / 2.0)
v_obs = cp.zeros((ny + 1, nx), dtype=cp.float32)
v_obs[1:-1] = cp.array((v_obs_cell[1:] + v_obs_cell[:-1]) / 2.0)

forward_solver = model.forward_solver
forward_solver.vanka_options.omega.set(0.5)
forward_solver.vanka_options.newton_options.relaxation.set(0.5)
forward_solver.vanka_options.newton_options.steps.set(30)

forward_solver.fas_options.coarsest_steps.set(200)
forward_solver.fas_options.pre_steps.set(10)
forward_solver.fas_options.post_steps.set(50)
forward_solver.fas_options.finest_steps.set(150)
forward_solver.fas_options.maximum_vcycles.set(10)
forward_solver.fas_options.relative_tolerance.set(1e-3)
forward_solver.fas_options.absolute_tolerance.set(10.0)

adjoint_solver = model.adjoint_solver
adjoint_solver.fas_options.coarsest_steps.set(200)
adjoint_solver.fas_options.pre_steps.set(10)
adjoint_solver.fas_options.post_steps.set(50)
adjoint_solver.fas_options.finest_steps.set(150)
adjoint_solver.fas_options.maximum_vcycles.set(10)
adjoint_solver.fas_options.absolute_tolerance.set(cp.float32(10.0))
adjoint_solver.fas_options.relative_tolerance.set(cp.float32(0.01))
adjoint_solver.vanka_options.newton_options.ssa_damping.set(cp.float32(0.1))
adjoint_solver.vanka_options.omega.set(cp.float32(0.5))

def tikhonov_regularization(field,weight=cp.float32(1.0)):
    """
    Compute Tikhonov (gradient smoothness) regularization.

    Parameters
    ----------
    field : cupy.ndarray
        2D field to regularize

    Returns
    -------
    loss : float
        Regularization loss
    grad : cupy.ndarray
        Gradient of loss w.r.t. field
    """
    diff_x = field[:, 1:] - field[:, :-1]
    diff_y = field[1:, :] - field[:-1, :]

    loss = 0.5 * (cp.sum(diff_x**2) + cp.sum(diff_y**2))

    grad = cp.zeros_like(field)
    grad[:, 1:-1] -= (field[:, 2:] - 2 * field[:, 1:-1] + field[:, :-2])
    grad[:, 0] -= (field[:, 1] - field[:, 0])
    grad[:, -1] -= (field[:, -2] - field[:, -1])
    grad[1:-1, :] -= (field[2:, :] - 2 * field[1:-1, :] + field[:-2, :])
    grad[0, :] -= (field[1, :] - field[0, :])
    grad[-1, :] -= (field[-2, :] - field[-1, :])

    return float(weight*loss), weight*grad

obs_hierarchy = [(u_obs,v_obs)]
for j in range(5):
    u_obs_coarse = mg.restrict_vfacet(obs_hierarchy[-1][0])
    v_obs_coarse = mg.restrict_hfacet(obs_hierarchy[-1][1])
    obs_hierarchy.append((u_obs_coarse,v_obs_coarse))
t = cp.float32(0.0)

for level in range(5,-1,-1):
    logger = InverseLogger(mg.levels[level],pvd_directory='./inverse/',pvd_base=f'level_{level}')
    model.set_top_level(level)
    u_obs,v_obs = obs_hierarchy[level]
    u_mask = abs(u_obs) > 0.01
    v_mask = abs(v_obs) > 0.01
    log_beta = cp.log(model.mg.levels[level].sliding.beta.data)
    for i in range(50):
        print("Solving Forward")
        mg.sliding.beta.set(cp.exp(log_beta),start_level=level)    
        model.forward(t,dt,update_geometry=False)

        u = mg.levels[level].state.u.data
        v = mg.levels[level].state.v.data

        n_pts = mg.levels[level].ny*mg.levels[level].nx

        dJdu = cp.sign(u - u_obs)*u_mask#/n_pts
        dJdv = cp.sign(v - v_obs)*v_mask#/n_pts

        print("Solving Adjoint")
        model.backward(t,dt,dJdu=dJdu,dJdv=dJdv)

        J_tikh,grad_log_beta_tikh = tikhonov_regularization(log_beta,weight=cp.float32(10.0))

        grad_log_beta_data = mg.levels[level].sliding.beta.grad * mg.levels[level].sliding.beta.data

        grad_log_beta = grad_log_beta_data + grad_log_beta_tikh


        J = (abs(u - u_obs)*u_mask).sum()/n_pts + (abs(v - v_obs)*v_mask).sum()/n_pts
        log_beta -= 0.02 * grad_log_beta / (abs(grad_log_beta) + 1e-3)

        logger(i)

        print("J",J,J_tikh/n_pts)
    if level>0:
        mg.prolongate_cell(mg.levels[level].sliding.beta.data,mg.levels[level-1].sliding.beta.data,method='bilinear')
"""
obs_hierarchy = [(u_obs, v_obs)]
current_u, current_v = u_obs, v_obs
g = grid
while g.child is not None:
    current_u = restrict_vfacet(current_u, kernels)
    current_v = restrict_hfacet(current_v, kernels)
    obs_hierarchy.append((current_u, current_v))
    g = g.child

for level_idx in [4,4,3,2,1,0]:
    physics.set_grid_level(level_idx)
    current_grid = physics.grid
    u_obs_level, v_obs_level = obs_hierarchy[level_idx]

    writer = VTIWriter(
        f"{OUTPUT_DIR}/level_{level_idx}",
        base="inverse",
        dx=float(current_grid.dx)
    )

    # Write observations
    u_obs_c = 0.5 * (u_obs_level[:, 1:] + u_obs_level[:, :-1])
    v_obs_c = 0.5 * (v_obs_level[1:] + v_obs_level[:-1])
    write_vti(
        f"{OUTPUT_DIR}/level_{level_idx}/u_obs.vti",
        {'vel': [u_obs_c, v_obs_c]},
        float(current_grid.dx)
    )

    counter = [0]
    H0 = cp.array(current_grid.H_prev)
    for i in range(5):
        u, v, H = physics.forward(dt=5.0, n_vcycles=10, verbose=True, rtol=1e-4)
    uref = cp.array(u)
    vref = cp.array(v)
    Href = cp.array(H)

    def objective(log_beta):

        current_grid.beta[:] = cp.exp(log_beta)
        restrict_parameters_to_hierarchy(current_grid)

        current_grid.u.fill(0)
        current_grid.v.fill(0)
        current_grid.H[:] = Href
        u, v, H = physics.forward(dt=1.0, n_vcycles=10, verbose=False,update_geometry=False,rtol=1e-3,atol=10.0)

        # Compute loss
        J_data, dJdu, dJdv = abs_loss(current_grid.u, current_grid.v, u_obs_level, v_obs_level)
        dJdH = cp.zeros_like(H)
    
        physics.adjoint(dJdu,dJdv,dJdH,n_vcycles=1,verbose=False)
        grad_log_beta = current_grid.beta*current_grid.grad_beta

        J_tikh,tikh_grad = tikhonov_regularization(log_beta,weight=cp.float32(REG_WEIGHT))
        J = J_data + J_tikh
        grad_log_beta += tikh_grad

        print(f"Level: {level_idx},  Loss: {J:.4f}, Loss Data: {J_data:.4f}, Loss Tikh: {J_tikh:.4f}")

        return float(J),grad_log_beta

    def callback(log_beta):
        counter[0] += 1

        u_c = 0.5 * (current_grid.u[:, 1:] + current_grid.u[:, :-1])
        v_c = 0.5 * (current_grid.v[1:] + current_grid.v[:-1])

        writer.write_step(counter[0], counter[0], {
            'log_beta': log_beta,
            'vel': [u_c * (1-current_grid.mask), v_c * (1-current_grid.mask)]
        })
        writer.write_pvd()

    log_beta = cp.log(current_grid.beta)

    for i in range(50):
        J,grad_log_beta = objective(log_beta)
        log_beta -= 0.02*np.sign(grad_log_beta)
        callback(log_beta)

    current_grid.beta[:] = cp.exp(log_beta)

    # Save result
    pickle.dump(
        current_grid.beta.get(),
        open(f"{OUTPUT_DIR}/beta_level_{level_idx}.p", 'wb')
    )

    if level_idx > 0:
        parent = physics.grids[level_idx - 1]
        prolongate_cell_centered(current_grid.beta, kernels, H_fine=parent.beta)   
"""
