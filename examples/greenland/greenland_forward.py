"""
Greenland forward simulation example.

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
#BETA_PATH = None
BETA_PATH = "./inverse_output/beta_level_0.p"
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

step = 0
t = cp.float32(0.0)
t_end = cp.float32(1000.0)
dt = cp.float32(25.0)
level = 0
logger = TimeLogger(mg.levels[level],pvd_directory='forward',pvd_base='greenland')
while t < t_end:
    print(t)
    solver.solve(dt,start_level=level)
    mg.levels[level].state.H_prev.data[:,:] = mg.levels[level].state.H.data[:,:]
    #mg.state.H_prev.set(mg.levels[level].state.H.data)
    t += dt
    step += 1
    logger(step,t)

