
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

from glide.multigrid import Multigrid, FASCDSolver
from scipy.ndimage import gaussian_filter
from glide.hooks import TimeLogger

### Load a dataset (here a preprocessed greenland dataset)
dataset = load_greenland_preprocessed()

### Initialize grid
# ny and nx must both divide by 2^(n_levels - 1) cleanly!
ny,nx,dx = dataset.ny,dataset.nx,dataset.dx
mg = Multigrid(n_levels=6,ny=ny,nx=nx,dx=dx)

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

### Initialize solver
solver = FASCDSolver(mg)

solver.vanka_options.omega.set(0.5)
solver.vanka_options.newton_options.relaxation.set(0.5)
solver.vanka_options.newton_options.steps.set(30)

solver.fas_options.coarsest_steps.set(200)
solver.fas_options.pre_steps.set(10)
solver.fas_options.post_steps.set(50)
solver.fas_options.finest_steps.set(150)
solver.fas_options.maximum_vcycles.set(10)
solver.fas_options.relative_tolerance.set(1e-3)
solver.fas_options.absolute_tolerance.set(10.0)

### Load velocity data ###
u_obs_cell = dataset.vx.values
v_obs_cell = dataset.vy.values

# Interpolate to faces
u_obs = cp.zeros((ny, nx + 1), dtype=cp.float32)
u_obs[:, 1:-1] = cp.array((u_obs_cell[:, 1:] + u_obs_cell[:, :-1]) / 2.0)
v_obs = cp.zeros((ny + 1, nx), dtype=cp.float32)
v_obs[1:-1] = cp.array((v_obs_cell[1:] + v_obs_cell[:-1]) / 2.0)






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
        """Callback for visualization."""
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

