"""
Antarctica inverse modeling example.

Infers basal friction (beta) from observed surface velocities using
adjoint-based optimization. Run interactively or as a script.
"""

import pickle
import cupy as cp
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from glide import IcePhysics
from glide.io import VTIWriter, write_vti
from glide.physics import huber_loss, tikhonov_regularization
from glide.data import (
    load_bedmachine,
    load_velocity_mosaic,
    load_smb_racmo,
    prepare_grid,
    interpolate_to_grid,
    load_antarctic_velocity,
    load_antarctica_preprocessed
)
from glide.kernels import restrict_vfacet, restrict_hfacet, prolongate_cell_centered, get_kernels
from glide.solver import restrict_parameters_to_hierarchy,restrict_solution_to_hierarchy
#from glide.kernels import prolongate_cell_centered

# =============================================================================
# Configuration - modify these paths and parameters
# =============================================================================

OUTPUT_DIR = "./inverse_output"

SKIP = 4              # Geometry downsampling factor
DT = 1.0             # Time step (years)
N_LEVELS = 5          # Multigrid levels

# Physical constants
RHO_ICE = 917.0
G = 9.81
N_GLEN = 3.0


REG_WEIGHT = 1e-4     # Tikhonov regularization weight
# =============================================================================
# Load data
# =============================================================================

"""
GEOMETRY_PATH = "./data/BedMachineAntarctica-v3.nc"
SMB_PATH = "./data/smbgl_monthlyS_ANT11_RACMO2.4p1_ERA5_197901_202312.nc"
U_OBS_PATH = "./data/antarctica_ice_velocity_450m_v2.nc"

print("Loading geometry...")
geometry = load_bedmachine(GEOMETRY_PATH, skip=SKIP, thklim=0.1,bbox_pad=[1100,1000,1600,1600])
geometry = prepare_grid(geometry, n_levels=N_LEVELS)

ny, nx = geometry['ny'], geometry['nx']
dx = geometry['dx']
x, y = geometry['x'], geometry['y']

print(f"Grid: {ny} x {nx}, dx = {dx:.1f} m")

print("Loading observed velocities...")
x_vel,y_vel,vx,vy = load_antarctic_velocity(U_OBS_PATH)

u_obs_cell = interpolate_to_grid(vx, x_vel, y_vel, x, y)
v_obs_cell = interpolate_to_grid(vy, x_vel, y_vel, x, y)


# Interpolate to faces
u_obs = cp.zeros((ny, nx + 1), dtype=cp.float32)
u_obs[:, 1:-1] = (u_obs_cell[:, 1:] + u_obs_cell[:, :-1]) / 2.0
v_obs = cp.zeros((ny + 1, nx), dtype=cp.float32)
v_obs[1:-1] = (v_obs_cell[1:] + v_obs_cell[:-1]) / 2.0

smb = load_smb_racmo(SMB_PATH,x,y)
"""
# =============================================================================
# Load data - From prepackaged
# =============================================================================

dataset = load_antarctica_preprocessed()
ny,nx = dataset.ny,dataset.nx
dx = dataset.dx
bed = dataset.bed.values
beta = dataset.beta.values
beta.fill(0.5)
surface = dataset.surface.values
thickness = dataset.thickness.values
smb = dataset.smb.values
smb[surface == 0] = -10.0

u_obs_cell = dataset.vx.values
v_obs_cell = dataset.vy.values

# Interpolate to faces
u_obs = cp.zeros((ny, nx + 1), dtype=cp.float32)
u_obs[:, 1:-1] = cp.array((u_obs_cell[:, 1:] + u_obs_cell[:, :-1]) / 2.0)
u_obs[cp.isnan(u_obs)] = 0.0
v_obs = cp.zeros((ny + 1, nx), dtype=cp.float32)
v_obs[1:-1] = cp.array((v_obs_cell[1:] + v_obs_cell[:-1]) / 2.0)
v_obs[cp.isnan(v_obs)] = 0.0

# =============================================================================
# Initialize physics
# =============================================================================

# Compute B (rate factor)
B_scalar = cp.float32(1e-17 ** (-1.0 / N_GLEN) / (RHO_ICE * G))
B = B_scalar * cp.ones((ny, nx), dtype=cp.float32)

print("Initializing physics...")
physics = IcePhysics(ny, nx, dx, n_levels=N_LEVELS, 
        thklim=0.1,
        n=3.0,eps_reg=1e-6,
        m=1.0/3.0,u_reg=1000.0,
        water_drag=1e-5,
        calving_rate=0.0,sigmoid_c=0.1)

physics.set_geometry(bed, thickness)
physics.set_parameters(B=B, beta=beta, smb=smb)

grid = physics.grid
kernels = get_kernels()

# =============================================================================
# Build observation hierarchy for multi-resolution optimization
# =============================================================================

obs_hierarchy = [(u_obs, v_obs)]
current_u, current_v = u_obs, v_obs
g = grid
while g.child is not None:
    current_u = restrict_vfacet(current_u, kernels)
    current_v = restrict_hfacet(current_v, kernels)
    obs_hierarchy.append((current_u, current_v))
    g = g.child

#for level_idx in range(N_LEVELS - 1, -1, -1):
for level_idx in [4,3,2,1,0]:
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

    def objective(log_beta_flat):

        log_beta = cp.asarray(log_beta_flat.reshape((current_grid.ny, current_grid.nx)), dtype=cp.float32) 
        current_grid.beta[:] = cp.exp(log_beta)
        restrict_parameters_to_hierarchy(current_grid)


        current_grid.u.fill(0)
        current_grid.v.fill(0)
        current_grid.H[:] = current_grid.H_prev
        u, v, H = physics.forward(dt=10.0, n_vcycles=10, verbose=False,update_geometry=False)

        # Compute loss
        J_data, dJdu, dJdv = huber_loss(current_grid.u, current_grid.v, u_obs_level, v_obs_level,epsilon=1.0,mask_threshold=0.001)
        dJdH = cp.zeros_like(H)
    
        physics.adjoint(dJdu,dJdv,dJdH,n_vcycles=3,verbose=False)
        grad_log_beta = current_grid.beta*current_grid.grad_beta

        J_tikh,tikh_grad = tikhonov_regularization(log_beta,weight=cp.float32(REG_WEIGHT))
        J = J_data + J_tikh
        grad_log_beta += tikh_grad

        J_norm = 0.5*(log_beta**2).sum()/log_beta.size
        J_norm_grad = log_beta/log_beta.size

        print(f"Level: {level_idx} {counter},  Loss: {J:.4f}, Loss Data: {J_data:.4f}, Loss Tikh: {J_tikh:.4f}, {J_norm}")

        return float(J),grad_log_beta.ravel().get().astype(np.float64)

    def callback(log_beta_flat):
        """Callback for visualization."""
        log_beta = cp.asarray(log_beta_flat.reshape((current_grid.ny, current_grid.nx)), dtype=cp.float32)
        counter[0] += 1

        u_c = 0.5 * (current_grid.u[:, 1:] + current_grid.u[:, :-1])
        v_c = 0.5 * (current_grid.v[1:] + current_grid.v[:-1])

        writer.write_step(counter[0], counter[0], {
            'log_beta': log_beta,
            'vel': [u_c, v_c]
        })
        writer.write_pvd()

    x_init = cp.log(current_grid.beta).ravel().get().astype(np.float64)
    bounds = [(-8,4)]*current_grid.nh
    result = fmin_l_bfgs_b(
        objective, x_init,
        bounds=bounds,
        callback=callback,
        factr=1e11,
        maxiter=200
    )

    current_grid.beta[:] = cp.exp(cp.array(result[0].reshape((current_grid.ny, current_grid.nx))).astype(cp.float32))     # Prolongate to finer grid for next level
    # Save result
    pickle.dump(
        current_grid.beta.get(),
        open(f"{OUTPUT_DIR}/beta_level_{level_idx}.p", 'wb')
    )

    if level_idx > 0:
        parent = physics.grids[level_idx - 1]
        prolongate_cell_centered(current_grid.beta, kernels, H_fine=parent.beta,smooth=False)   

