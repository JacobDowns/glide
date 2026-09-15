"""
Antarctica inverse modeling example.

Infers basal friction (beta) from observed surface velocities using
adjoint-based optimization. Run interactively or as a script.
"""

import cupy as cp
import numpy as np
import torch
import pyproj

from scipy.ndimage import gaussian_filter

from glide.model import IceDynamics
from glide.data import load_antarctica_preprocessed
from glide.field import Field, GridEntity
from glide.torch import glide_step
from glide.io import VTIWriter

### Load a dataset (here a preprocessed greenland dataset)
dataset = load_antarctica_preprocessed()

### Initialize grid
# ny and nx must both divide by 2^(n_levels - 1) cleanly!
n_levels = 6

### Stress balance: 'ssa', 'molho' (MOLHO-MI) or 'diva'.
stress_scheme = 'diva'

# Per-scheme inversion configuration (Antarctica).  All three schemes use the SAME setup: coarse-to-fine
# from level 2 (the upstream Antarctica default), lr=0.01, and Tikhonov smoothness regularization ONLY --
# no lower floor, no off-ice mask -- so beta never collapses (the smoothness term keeps data-void cells
# near their observed neighbours).  The ONLY per-scheme difference is the forward tolerance: DIVA's closure
# benefits from a tighter solve (1e-3 vs 1e-2); it is otherwise identical.  DIVA matches the observed
# SURFACE speed through its closure u_s (n_sigma quadrature points); SSA/MOLHO match u_bar + u_d/(n+1).
INV = {
    'ssa':   dict(lr=1e-2, fwd_rtol=1e-2, coarsest_level=2, epochs={2: 50, 1: 50, 0: 50}),
    'molho': dict(lr=1e-2, fwd_rtol=1e-2, coarsest_level=2, epochs={2: 50, 1: 50, 0: 50}),
    'diva':  dict(lr=1e-2, fwd_rtol=1e-3, coarsest_level=2, epochs={2: 50, 1: 50, 0: 50}),
}[stress_scheme]

ny,nx,dx = dataset.ny,dataset.nx,dataset.dx
model = IceDynamics(n_levels=n_levels,ny=ny,nx=nx,dx=dx,
        x0=dataset.x[0].item(),y0=dataset.y[0].item(),
        crs=pyproj.CRS("EPSG:3031"),stress_scheme=stress_scheme)
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
mg.geometry.depth.set(np.maximum(-bed,0.0))
mg.geometry.sigmoid_c.set(0.1)
mg.geometry.sigmoid_k.set(3.0)

### Initialize rheology
B = cp.zeros((ny,nx), dtype=cp.float32)
B.fill(1e-17 ** (-1.0 / 3.0) / (917 * 9.81)) 
mg.rheology.B.set(B)
mg.rheology.eps_reg.set(1e-6)
mg.rheology.n.set(3.0)
mg.rheology.H_reg.set(25.0)
if stress_scheme == 'diva':
    mg.rheology.n_sigma.set(6.0)          # vertical quadrature points (velocity converged by ~6)
    mg.rheology.eps_reg_shear.set(1e-6)

n_glen = 3.0

### Initialize sliding
beta = cp.zeros((ny,nx), dtype=cp.float32)
beta.fill(2.5)

mg.sliding.beta.set(beta)
mg.sliding.m.set(1./3.)
mg.sliding.u_reg.set(1.0)
mg.sliding.water_drag.set(1e-4)

### Initialize calving
mg.calving.calving_rate.set(0.0)

### Initialize forcing
smb = dataset.smb.values
smb[dataset.surface.values==0] = -20.0
mg.forcing.smb.set(smb)

### Load velocity data ###
u_obs = cp.array(dataset.vx.values,dtype=cp.float32)
u_obs[cp.isnan(u_obs)] = 0.0
v_obs = cp.array(dataset.vy.values,dtype=cp.float32)
v_obs[cp.isnan(v_obs)] = 0.0

# Build hierarchy of observations
observation_levels = [(u_obs,v_obs)]
for j in range(1,n_levels):
    u_obs_coarse = mg.restrict_cell(observation_levels[-1][0])
    v_obs_coarse = mg.restrict_cell(observation_levels[-1][1])
    observation_levels.append((u_obs_coarse,v_obs_coarse))

### Set multigrid solver parameters ###
model.forward_solver.fas_options.set(
        coarsest_steps=200, pre_steps=10, 
        post_steps=150, finest_steps=0,
        relative_tolerance=INV['fwd_rtol'], absolute_tolerance=10.0,
        report_norms=True)

model.forward_solver.vanka_options.omega.set(cp.float32(0.25))
model.forward_solver.vanka_options.newton_options.momentum_damping.set(cp.float32(0.1))
model.forward_solver.vanka_options.newton_options.step_tolerance.set(cp.float32(1e-6))

model.adjoint_solver.fas_options.set(
        coarsest_steps=200, pre_steps=10,
        post_steps=150, finest_steps=0,
        relative_tolerance=1e-2, absolute_tolerance=1e-5, # Note that adjoint var
        report_norms=True)                               # adjoint var is small 
                                                          # in magnitude
model.adjoint_solver.vanka_options.omega.set(cp.float32(0.25))
model.adjoint_solver.vanka_options.newton_options.momentum_damping.set(cp.float32(0.01))
model.adjoint_solver.vanka_options.newton_options.step_tolerance.set(cp.float32(1e-6))

t = cp.float32(0.0) # Dummy time, which we don't use here

# Index of the coarsest grid to start on (per-level epoch counts come from INV['epochs'])
coarsest_level = INV['coarsest_level']
log_beta = torch.log(torch.tensor(mg[coarsest_level].sliding.beta.data,device='cuda'))

# Solve the inverse problem at progressively coarser levels
for level in range(coarsest_level,-1,-1):
    # Examples of different writing utilities - First writes to vti/pvd

    # This is what we're optimizing - Convert from the initial guess, 
    # either defined above or by prolongation from the coarser state
    log_beta.requires_grad_()

    # These can be differentiated wrt - but we don't in this simple problem
    H_prev = torch.tensor(mg[level].state.H_prev.data,device='cuda')
    bed = torch.tensor(mg[level].geometry.bed.data,device='cuda')
    smb = torch.tensor(mg[level].forcing.smb.data,device='cuda')

    u_obs,v_obs = (torch.as_tensor(t) for t in observation_levels[level])
    u_mask = abs(u_obs) > 0.01
    v_mask = abs(v_obs) > 0.01

    # Standard torch optimization loop (RMSprop works very well here)
    optimizer = torch.optim.RMSprop([log_beta],lr=INV['lr'])
    n_level_epochs = INV['epochs'][level]

    # Derived surface velocity fields for output monitoring: with the MOLHO
    # ansatz the surface velocity is u_bar + u_d/(n+1)
    ny_l, nx_l = mg[level].ny, mg[level].nx
    u_s_field = Field(
            data=cp.zeros((ny_l,nx_l+1),dtype=cp.float32),
            grid_entity=GridEntity.VERTICAL_FACET,
            dx=mg[level].dx, grid=mg[level], name='u_s', units='m a^{-1}',
            attrs={'long_name':'Surface velocity (x)'})
    v_s_field = Field(
            data=cp.zeros((ny_l+1,nx_l),dtype=cp.float32),
            grid_entity=GridEntity.HORIZONTAL_FACET,
            dx=mg[level].dx, grid=mg[level], name='v_s', units='m a^{-1}',
            attrs={'long_name':'Surface velocity (y)'})

    # Initialize writer
    vti_writer = VTIWriter(f'inverse/level_{level}/vti', base='antarctica', dx=mg[level].dx,
            static_fields={'U_obs':[u_obs,v_obs]},
            dynamic_fields={'beta':mg[level].sliding.beta,
                            'U':[mg[level].state.u, mg[level].state.v],
                            'U_s':[u_s_field, v_s_field],
                            'xi':mg[level].state.xi}
        )

    vti_writer.initialize(mg[level])
    for j in range(n_level_epochs):
        optimizer.zero_grad()

        # Convert log(beta) to beta
        beta = torch.exp(log_beta)

        # Predict the velocity (and thickness) at t + dt, matching the observed SURFACE velocity.
        # SSA/MOLHO: u_s = u_bar + u_d/(n+1).  DIVA: the closure surface speed u_s, built into a facet
        # vector (scale the depth-averaged facet velocity by the surface/depth-avg ratio) exactly like
        # SSA/MOLHO so speed AND direction are fit the same way across schemes.
        mg.sliding.water_drag.set(1e-5)
        if stress_scheme == 'diva':
            u,v,ud,vd,H,mask,u_s = glide_step(t,dt,model,level,H_prev,bed,beta,smb,return_u_s=True)
            uc = 0.5*(u[:,1:]+u[:,:-1]); vc = 0.5*(v[1:]+v[:-1])
            ratio = u_s / torch.sqrt(uc**2 + vc**2 + 1e-6)
            rx = torch.ones_like(u); rx[:,1:-1] = 0.5*(ratio[:,1:]+ratio[:,:-1]); rx[:,0] = ratio[:,0]; rx[:,-1] = ratio[:,-1]
            ry = torch.ones_like(v); ry[1:-1] = 0.5*(ratio[1:]+ratio[:-1]); ry[0] = ratio[0]; ry[-1] = ratio[-1]
            u_s_f = u*rx; v_s_f = v*ry
            u_cell = 0.5*(u_s_f[:,1:] + u_s_f[:,:-1]); v_cell = 0.5*(v_s_f[1:] + v_s_f[:-1])
        else:
            u,v,ud,vd,H,mask = glide_step(t,dt,model,level,H_prev,bed,beta,smb)
            u_s = u + ud/(n_glen + 1.0); v_s = v + vd/(n_glen + 1.0)
            u_cell = 0.5*(u_s[:,1:] + u_s[:,:-1]); v_cell = 0.5*(v_s[1:] + v_s[:-1])

        # L1 Objective function, masked by valid data
        J_data = (abs(u_cell - u_obs)*u_mask).mean() + (abs(v_cell - v_obs)*v_mask).mean()

        # Compute first differences
        dx = mg[level].dx
        gy = torch.diff(log_beta,dim=0)/dx
        gx = torch.diff(log_beta,dim=1)/dx
        
        # Tikhonov Regularization
        J_L2 = 1e-4*((gy**2).sum() + (gx**2).sum())*dx**2
        
        # TV Regularization
        gy_ = gy[:,:-1]
        gx_ = gx[:-1]
        eps = 1e-6
        J_L1 = 0*(torch.sqrt(gy_**2 + gx_**2 + eps**2).sum())*dx**2
        
        # Combined objective - elastic net regularization
        J = J_data + J_L1 + J_L2

        # Backpropagate
        mg.sliding.water_drag.set(1e-4)
        J.backward()

        # Update parameter
        optimizer.step()
        log_beta.data[log_beta.data > 3.5] = 3.5
        
        print(f"Level {level}, Iter. {j}/{n_level_epochs} | J: {J.item():.2f}, J_data: {J_data.item():.2f}, J_L1: {J_L1.item():.2f}, J_L2: {J_L2.item():.2f}")
        su = mg[level].state.u.data; sv = mg[level].state.v.data
        if stress_scheme == 'diva':
            ubar = cp.hypot(0.5*(su[:,1:]+su[:,:-1]), 0.5*(sv[1:]+sv[:-1]))
            r = mg[level].state.u_s.data/(ubar + 1e-6)         # DIVA surface/mean speed ratio (cells)
            rx = cp.empty_like(su); rx[:,1:-1] = 0.5*(r[:,1:]+r[:,:-1]); rx[:,0] = r[:,0]; rx[:,-1] = r[:,-1]
            ry = cp.empty_like(sv); ry[1:-1] = 0.5*(r[1:]+r[:-1]); ry[0] = r[0]; ry[-1] = r[-1]
            u_s_field.data[:,:] = su*rx; v_s_field.data[:,:] = sv*ry
        else:
            u_s_field.data[:,:] = su + mg[level].state.ud.data/(n_glen + 1.0)
            v_s_field.data[:,:] = sv + mg[level].state.vd.data/(n_glen + 1.0)
        vti_writer.append(mg[level],time=j)
        vti_writer.write_pvd()

    if level>0:
        log_beta = torch.tensor(mg.prolongate_cell(cp.asarray(log_beta.detach()),method='bilinear'))

    beta_xr = mg[level].sliding.beta.to_dataarray()
    beta_xr.to_netcdf(f'./inverse/level_{level}/beta_opt.nc')

