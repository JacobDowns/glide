"""
Greenland inverse modeling example.

Infers basal friction (beta) from observed surface velocities using
adjoint-based optimization. Run interactively or as a script.
"""

import xarray as xr
import pickle
import cupy as cp
import numpy as np
from scipy.optimize import fmin_l_bfgs_b


from glide import IcePhysics
from glide.io import VTIWriter, write_vti
from glide.physics import abs_loss, tikhonov_regularization
from glide.data import (
    load_bedmachine,
    load_velocity_mosaic,
    load_smb_mar,
    prepare_grid,
    interpolate_to_grid,
    load_greenland_preprocessed
)


from glide.kernels import restrict_vfacet, restrict_hfacet, prolongate_cell_centered, get_kernels
from glide.solver import adjoint_vcycle_fas, restrict_parameters_to_hierarchy
from scipy.ndimage import gaussian_filter


# =============================================================================
# Configuration - modify these paths and parameters
# =============================================================================

OUTPUT_DIR = "./inverse_output"

SKIP = 6              # Geometry downsampling factor
DT = 10.0             # Time step (years)
N_LEVELS = 5          # Multigrid levels

# Physical constants
RHO_ICE = 917.0
G = 9.81
N_GLEN = 3.0

REG_WEIGHT = 1e-4     # Tikhonov regularization weight
# =============================================================================
# Load data - from source files
# =============================================================================
"""
GEOMETRY_PATH = "./data/BedMachineGreenland-v5.nc"
U_OBS_PATH = "./data/greenland_vel_mosaic250_vx_v1.tif"
V_OBS_PATH = "./data/greenland_vel_mosaic250_vy_v1.tif"
SMB_PATH = "./data/MARv3.9-yearly-MIROC5-rcp85-ltm1995-2014.nc"

print("Loading geometry...")
geometry = load_bedmachine(GEOMETRY_PATH, skip=SKIP, thklim=0.1)
geometry = prepare_grid(geometry, n_levels=N_LEVELS)

ny, nx = geometry['ny'], geometry['nx']
dx = geometry['dx']
x, y = geometry['x'], geometry['y']

print(f"Grid: {ny} x {nx}, dx = {dx:.1f} m")

print("Loading observed velocities...")
vel_data = load_velocity_mosaic(U_OBS_PATH, V_OBS_PATH)

u_obs_cell = interpolate_to_grid(vel_data['u'], vel_data['x'], vel_data['y'], x, y)
v_obs_cell = interpolate_to_grid(vel_data['v'], vel_data['x'], vel_data['y'], x, y)

# Interpolate to faces
u_obs = cp.zeros((ny, nx + 1), dtype=cp.float32)
u_obs[:, 1:-1] = (u_obs_cell[:, 1:] + u_obs_cell[:, :-1]) / 2.0
v_obs = cp.zeros((ny + 1, nx), dtype=cp.float32)
v_obs[1:-1] = (v_obs_cell[1:] + v_obs_cell[:-1]) / 2.0

print("Loading SMB...")
smb_data = load_smb_mar(SMB_PATH)
smb = interpolate_to_grid(smb_data['smb'], smb_data['x'], smb_data['y'], x, y)
"""
# =============================================================================
# Load data - From prepackaged
# =============================================================================

dataset = load_greenland_preprocessed()
ny,nx = dataset.ny,dataset.nx
dx = dataset.dx
bed = dataset.bed.values
bed = gaussian_filter(bed,1)
surface = dataset.surface.values
thickness = dataset.thickness.values
thickness = gaussian_filter(thickness,1)
beta = dataset.beta.values
beta.fill(2.5)
smb = dataset.smb.values

u_obs_cell = dataset.vx.values
v_obs_cell = dataset.vy.values

# Interpolate to faces
u_obs = cp.zeros((ny, nx + 1), dtype=cp.float32)
u_obs[:, 1:-1] = cp.array((u_obs_cell[:, 1:] + u_obs_cell[:, :-1]) / 2.0)
v_obs = cp.zeros((ny + 1, nx), dtype=cp.float32)
v_obs[1:-1] = cp.array((v_obs_cell[1:] + v_obs_cell[:-1]) / 2.0)

# =============================================================================
# Initialize physics
# =============================================================================

# Compute B (rate factor)
B_scalar = cp.float32(1e-17 ** (-1.0 / N_GLEN) / (RHO_ICE * G))
B = B_scalar * cp.ones((ny, nx), dtype=cp.float32)


print("Initializing physics...")
physics = IcePhysics(ny, nx, dx, n_levels=N_LEVELS, 
        thklim=0.1,
        n=3.0,eps_reg=1e-5,
        m=1./3.,u_reg=1.0,
        water_drag=1e-5,
        calving_rate=2000.0,sigmoid_c=0.1)

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

