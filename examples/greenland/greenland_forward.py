"""
Greenland forward simulation example.

Run interactively or execute as a script. Modify the paths and parameters
below to match your setup.
"""

import xarray as xr
import pickle
import cupy as cp
import numpy as np

#from glide import IcePhysics
from glide.io import VTIWriter, write_vti
from glide.data import (
    load_bedmachine,
    load_smb_mar,
    prepare_grid,
    interpolate_to_grid,
    load_greenland_preprocessed
)

from glide.multigrid import Multigrid, FASCDSolver
from scipy.ndimage import gaussian_filter

# =============================================================================
# Configuration - modify these paths and parameters
# =============================================================================

OUTPUT_DIR = "./output_fmg"

SKIP = 6           # Geometry downsampling factor
DT = 20.0          # Time step (years)
N_STEPS = 50      # Number of time steps
N_LEVELS = 5       # Multigrid levels
N_VCYCLES = 20      # V-cycles per time step

# Physical constants
RHO_ICE = 917.0
G = 9.81
N_GLEN = 3.0

# =============================================================================
# Load data - from source files
# =============================================================================

"""
GEOMETRY_PATH = "./data/BedMachineGreenland-v5.nc"
SMB_PATH = "./data/MARv3.9-yearly-MIROC5-rcp85-ltm1995-2014.nc"
BETA_PATH = "./inverse_output/beta_level_0.p"

print("Loading geometry...")
geometry = load_bedmachine(GEOMETRY_PATH, skip=SKIP, thklim=0.1)
geometry = prepare_grid(geometry, n_levels=N_LEVELS)

bed = geometry['bed']
thickness = geometry['thickness']

ny, nx = geometry['ny'], geometry['nx']
dx = geometry['dx']
x, y = geometry['x'], geometry['y']

print(f"Grid: {ny} x {nx}, dx = {dx:.1f} m")

print("Loading SMB...")
smb_data = load_smb_mar(SMB_PATH)
smb = interpolate_to_grid(
    smb_data['smb'], smb_data['x'], smb_data['y'],
    x, y
)

print("Loading beta...")
beta = cp.array(pickle.load(open(BETA_PATH, 'rb')))
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
smb += 1
BETA_PATH = "./inverse_output/beta_level_0.p"
beta = cp.array(pickle.load(open(BETA_PATH, 'rb')))
#beta[beta>5] = 5


# =============================================================================
# Initialize physics
# =============================================================================

# Compute B (rate factor - we measure driving stress in units of head, so the rho g factor gets subsumed into definitions of beta and B!)
B_scalar = cp.float32(1e-17 ** (-1.0 / N_GLEN) / (RHO_ICE * G))
B = B_scalar * cp.ones((ny, nx), dtype=cp.float32)

from glide.grid import Grid
grid = Grid(ny,nx,dx)

grid.state.H_prev.set(thickness)
grid.state.H.set(thickness)
grid.geometry.bed.set(bed)
grid.forcing.smb.set(smb)
grid.rheology.B.set(B)
grid.sliding.beta.set(beta)

grid.sliding.m.set(1./3.)
grid.calving.calving_rate.set(2000.0)
grid.rheology.eps_reg.set(1e-5)
grid.sliding.water_drag.set(1e-5)

dt = cp.float32(25.0)
mg = Multigrid(grid)
mg.create_grid_hierarchy(6)

class VankaLogger:
    def __init__(self,grid,level,write=False):
        self.writer = None
        if write:
            self.writer = VTIWriter(OUTPUT_DIR, base=f"level_{level}", dx=grid.dx)
        self.grid = grid

    def __call__(self,i):
        self.grid.forward_operators.compute_residual(dt,use_mask=True,recompute_phi=False)
        print(
            cp.linalg.norm(self.grid.forward_operators.r_u),
            cp.linalg.norm(self.grid.forward_operators.r_v),
            cp.linalg.norm(self.grid.forward_operators.r_H)
        )

        if self.writer:
            u_c = 0.5*(self.grid.state.u.data[:,1:] 
                + self.grid.state.u.data[:,:-1])
            v_c = 0.5*(self.grid.state.v.data[1:] 
                + self.grid.state.v.data[:-1])
            self.writer.write_step(i, i, {
                'r_H': self.grid.forward_operators.r_H,
                'u': [u_c,v_c],
                'H': self.grid.state.H.data}
            )
            self.writer.write_pvd()

class TimeLogger:
    def __init__(self,grid):
        self.writer = VTIWriter(OUTPUT_DIR, base=f"greenland", dx=grid.dx)
        self.grid = grid
        
    def __call__(self,i,t):
        u_c = 0.5*(self.grid.state.u.data[:,1:] 
            + self.grid.state.u.data[:,:-1])
        v_c = 0.5*(self.grid.state.v.data[1:] 
            + self.grid.state.v.data[:-1])
        self.writer.write_step(i, t, {
            'u': [u_c*(1-self.grid.state.mask.data),v_c*(1-self.grid.state.mask.data)],
            'H': self.grid.state.H.data,
            'S': self.grid.state.H.data + cp.maximum(self.grid.geometry.bed.data,-0.917*self.grid.state.H.data)}
        )
        self.writer.write_pvd()


for g in mg.grids:
    g.forward_operators.vanka_config.omega = cp.float32(0.5)
    g.forward_operators.vanka_config.newton_steps = 30
    g.forward_operators.vanka_config.hook_interval = 1
    g.forward_operators.vanka_config.newton_config.relaxation = cp.float32(0.5)
    g.forward_operators.vanka_config.newton_config.steps=30

for g in mg.grids[1:]:
    g.calving.calving_rate.set(cp.float32(0.0))

cp.random.seed(0)
grid.state.u.data[:,:] = cp.random.randn(grid.ny,grid.nx + 1)
grid.state.v.data[:,:] = cp.random.randn(grid.ny+1,grid.nx)
grid.state.H.data[:,:] += cp.random.randn(grid.ny,grid.nx)**2


u0 = cp.array(grid.state.u.data)
v0 = cp.array(grid.state.v.data)
H0 = cp.array(grid.state.H.data)

grid.forward_operators.set_rhs(dt)
grid.forward_operators.compute_phi()
#grid.forward_operators.vanka_config.hook_func = VankaLogger(grid,0)
#grid.forward_operators.vanka_smooth(dt,recompute_phi=False)
#grid.forward_operators.vanka_sweep(dt,100,recompute_phi=False)




#J,r = grid.forward_operators.vanka_dump(dt)


solver = FASCDSolver(mg)
solver.config.coarse_steps = 200
solver.config.pre_steps = 10
solver.config.post_steps = 20
solver.config.finest_steps = 200


t = 0.0
n_steps = 25
writer = VTIWriter(OUTPUT_DIR, base="greenland", dx=dx)
logger = TimeLogger(grid)
for step in range(n_steps):
    print(t)
    solver.solve(dt,max_iter=10,rel_tol=1e-4,abs_tol=10)
    grid.state.H_prev.data[:,:] = grid.state.H.data[:,:]
    t += dt

    logger(step,t)

"""
for level in range(5,-1,-1):
    grid = mg.grids[level]
    
    logger = VankaLogger(grid,level)

    grid.forward_operators.vanka_config.omega = cp.float32(0.66)
    grid.forward_operators.vanka_config.newton_steps = 100
    grid.forward_operators.vanka_config.hook_interval = 10
    grid.forward_operators.vanka_config.hook_func = VankaLogger(grid,level)
    grid.forward_operators.vanka_config.newton_config.relaxation = cp.float32(0.33)

    grid.forward_operators.set_rhs(dt)
    grid.forward_operators.vanka_sweep(dt,500)
    if level > 0:
        mg.prolongate_vfacet(grid.state.u.data,mg.grids[level-1].state.u.data,method='bilinear')
        mg.prolongate_hfacet(grid.state.v.data,mg.grids[level-1].state.v.data,method='bilinear')
        mg.prolongate_cell(grid.state.H.data,mg.grids[level-1].state.H.data,method='bilinear')
"""
"""
print("Initializing physics...")
physics = IcePhysics(ny, nx, dx, n_levels=N_LEVELS, 
        thklim=0.1,
        n=3.0,eps_reg=1e-5,
        m=1./3.,u_reg=1.0,
        water_drag=1e-5,
        calving_rate=2000.0,sigmoid_c=0.1)
physics.set_geometry(bed, thickness)
physics.set_parameters(B=B, beta=beta, smb=smb)

# Access the grid hierarchy
grid = physics.grid

# =============================================================================
# Set up output
# =============================================================================

writer = VTIWriter(OUTPUT_DIR, base="greenland", dx=dx)
write_vti(f"{OUTPUT_DIR}/bed.vti", {'bed': grid.bed}, dx)

# =============================================================================
# Time stepping
# =============================================================================

print(f"Running {N_STEPS} time steps of {DT} years...")
t = 0.0

for step in range(N_STEPS):
    print(f"Step {step}: t = {t:.1f} yr, H_mean = {float(grid.H.mean()):.1f} m")

    # Forward solve
    u, v, H = physics.forward(dt=DT, n_vcycles=N_VCYCLES, verbose=True,update_geometry=True,rtol=1e-4)
    t += DT

    # Output
    u_c, v_c = physics.get_velocities_cell_centered()
    surface = physics.get_surface()

    writer.write_step(step, t, {
        'thk': H,
        'srf': surface,
        'vel': [u_c * (1-grid.mask), v_c*(1-grid.mask)]
    })
    writer.write_pvd()

print("Done!")
"""

