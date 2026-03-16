import cupy as cp
from dataclasses import dataclass
from pathlib import Path
from .grid import Grid

#from .kernels import (
#    restrict_vfacet, restrict_hfacet, restrict_cell_centered,
#    restrict_max_pool, restrict_min_pool, prolongate_vfacet, prolongate_hfacet,
#    prolongate_cell_centered
#)

class Multigrid:
    def __init__(self,finest_grid,use_fast_math=True):
        self.finest_grid = finest_grid
        self.grids = [finest_grid]
        self.initialized = False

        cuda_dir = Path(__file__).parent / "cuda"

        # Concatenate ice kernel files in dependency order
        cuda_files = ['common.cu', 'transfer.cu']
        cuda_source = '\n'.join((cuda_dir / f).read_text() for f in cuda_files)
        
        if use_fast_math:
            options=("--use_fast_math",)
        else:
            options=()

        self.kernels = cp.RawModule(code=cuda_source, options=options)

    def create_grid_hierarchy(self,n_grids,restrict_fields=True):
        self.grids = [self.finest_grid]
        for i in range(1,n_grids):
            coarse_grid = self.create_coarse_grid(self.grids[-1],
                restrict_fields=restrict_fields)
            self.grids.append(coarse_grid)
        return self.grids


    def create_coarse_grid(self,parent_grid,restrict_fields=True):
        child_grid = Grid(
            parent_grid.ny // 2, parent_grid.nx // 2,
            parent_grid.dx * 2, parent=parent_grid
        )
        parent_grid.child = child_grid
        if restrict_fields == True:
            self.restrict_state(parent_grid,child_grid)
            self.restrict_geometry(parent_grid,child_grid)
            self.restrict_rheology(parent_grid,child_grid)
            self.restrict_sliding(parent_grid,child_grid)
            self.restrict_calving(parent_grid,child_grid)
            self.restrict_forcing(parent_grid,child_grid)
        return child_grid

    def restrict_state(self,fine_grid,coarse_grid):
        self.restrict_vfacet(fine_grid.state.u.data,coarse_grid.state.u.data)
        self.restrict_hfacet(fine_grid.state.v.data,coarse_grid.state.v.data)
        self.restrict_cell(fine_grid.state.H.data,coarse_grid.state.H.data)
        self.restrict_cell(fine_grid.state.H_prev.data,coarse_grid.state.H_prev.data)
        self.restrict_cell(fine_grid.state.phi.data,coarse_grid.state.phi.data)
        self.restrict_cell(fine_grid.state.mask.data,coarse_grid.state.mask.data,method='max')

    def restrict_geometry(self,fine_grid,coarse_grid):
        self.restrict_cell(fine_grid.geometry.bed.data,coarse_grid.geometry.bed.data)
        coarse_grid.geometry.thklim.set(fine_grid.geometry.thklim.value)
        coarse_grid.geometry.flotation_reg_driving.set(fine_grid.geometry.flotation_reg_driving.value)

    def restrict_rheology(self,fine_grid,coarse_grid):
        self.restrict_cell(fine_grid.rheology.B.data,coarse_grid.rheology.B.data)
        coarse_grid.rheology.n.set(fine_grid.rheology.n.value)
        coarse_grid.rheology.eps_reg.set(fine_grid.rheology.eps_reg.value)
    
    def restrict_sliding(self,fine_grid,coarse_grid):
        self.restrict_cell(fine_grid.sliding.beta.data,coarse_grid.sliding.beta.data)
        coarse_grid.sliding.m.set(fine_grid.sliding.m.value)
        coarse_grid.sliding.u_reg.set(fine_grid.sliding.u_reg.value)
        coarse_grid.sliding.water_drag.set(fine_grid.sliding.water_drag.value)
        coarse_grid.sliding.flotation_reg_sliding.set(fine_grid.sliding.flotation_reg_sliding.value)

    def restrict_calving(self,fine_grid,coarse_grid):
        coarse_grid.calving.calving_rate.set(fine_grid.calving.calving_rate.value)
        coarse_grid.calving.flotation_reg_calving.set(fine_grid.calving.flotation_reg_calving.value)
    
    def restrict_forcing(self,fine_grid,coarse_grid):
        self.restrict_cell(fine_grid.forcing.smb.data,coarse_grid.forcing.smb.data)

    def restrict_residual(self,fine_grid,coarse_grid):
        self.restrict_vfacet(fine_grid.forward_operators.r_u,coarse_grid.forward_operators.r_u)
        self.restrict_hfacet(fine_grid.forward_operators.r_v,coarse_grid.forward_operators.r_v)
        self.restrict_cell(fine_grid.forward_operators.r_H,coarse_grid.forward_operators.r_H)
   
    def restrict_vfacet(self,fine_field,coarse_field=None):
        """Restrict u-velocity (vertical face) field to coarse grid."""
        kernel = self.kernels.get_function('restrict_vfacet')
        ny, nx_plus_1 = fine_field.shape
        nx = nx_plus_1 - 1
        ny_coarse = ny // 2
        nx_coarse = nx // 2

        if coarse_field is None:
            coarse_field = cp.empty((ny_coarse, nx_coarse + 1), dtype=cp.float32)

        total_work = ny_coarse * (nx_coarse + 1)
        block_size = 256
        grid_size = (total_work + block_size - 1) // block_size

        kernel((grid_size,), (block_size,),
               (fine_field, coarse_field, ny_coarse, nx_coarse))

        return coarse_field

    def restrict_hfacet(self,fine_field,coarse_field=None):
        """Restrict u-velocity (vertical face) field to coarse grid."""
        kernel = self.kernels.get_function('restrict_hfacet')
        ny_plus_1, nx = fine_field.shape
        ny = ny_plus_1 - 1
        ny_coarse = ny // 2
        nx_coarse = nx // 2

        if coarse_field is None:
            coarse_field = cp.empty((ny_coarse + 1, nx_coarse), dtype=cp.float32)

        total_work = (ny_coarse + 1) * nx_coarse
        block_size = 256
        grid_size = (total_work + block_size - 1) // block_size

        kernel((grid_size,), (block_size,),
               (fine_field, coarse_field, ny_coarse, nx_coarse))

        return coarse_field

    def restrict_cell(self,fine_field,coarse_field=None,method='avg'):
        """Restrict u-velocity (vertical face) field to coarse grid."""
        if method == 'avg':
            kernel = self.kernels.get_function('restrict_cell_avg')
        elif method == 'max':
            kernel = self.kernels.get_function('restrict_cell_max')
        elif method == 'min':
            kernel = self.kernels.get_function('restrict_cell_min')
        elif method == 'var':
            kernel = self.kernels.get_function('restrict_cell_var')
        else:
            raise TypeError('Valid restriction methods: [avg,max,min]')

        ny, nx = fine_field.shape
        ny_coarse = ny // 2
        nx_coarse = nx // 2

        if coarse_field is None:
            coarse_field = cp.empty((ny_coarse, nx_coarse), dtype=cp.float32)

        total_work = ny_coarse * nx_coarse
        block_size = 256
        grid_size = (total_work + block_size - 1) // block_size

        kernel((grid_size,), (block_size,),
               (fine_field, coarse_field, ny_coarse, nx_coarse))

        return coarse_field

    def prolongate_vfacet(self,coarse_field, fine_field=None, method='injection'):
        """Prolongate u-velocity (vertical face) field to fine grid."""
        if method == 'injection':
            kernel = self.kernels.get_function('prolongate_vfacet_injection')
        elif method == 'bilinear':
            kernel = self.kernels.get_function('prolongate_vfacet_bilinear')
        else:
            raise TypeError('Valid prolongation methods: [injection, bilinear]')

        ny, nx_plus_1 = coarse_field.shape
        nx = nx_plus_1 - 1
        ny_fine = ny * 2
        nx_fine = nx * 2

        if fine_field is None:
            fine_field = cp.empty((ny_fine, nx_fine + 1), dtype=cp.float32)

        total_work = ny_fine * (nx_fine + 1)
        block_size = 256
        grid_size = (total_work + block_size - 1) // block_size

        kernel((grid_size,), (block_size,),
               (coarse_field, fine_field, ny_fine, nx_fine))
        return fine_field

    def prolongate_hfacet(self,coarse_field, fine_field=None, method='injection'):
        """Prolongate u-velocity (vertical face) field to fine grid."""
        if method == 'injection':
            kernel = self.kernels.get_function('prolongate_vfacet_injection')
        elif method == 'bilinear':
            kernel = self.kernels.get_function('prolongate_vfacet_bilinear')
        else:
            raise TypeError('Valid prolongation methods: [injection, bilinear]')

        ny_plus_1, nx = coarse_field.shape
        ny = ny_plus_1 - 1
        ny_fine = ny * 2
        nx_fine = nx * 2

        if fine_field is None:
            fine_field = cp.empty((ny_fine + 1, nx_fine), dtype=cp.float32)

        total_work = (ny_fine + 1) * nx_fine
        block_size = 256
        grid_size = (total_work + block_size - 1) // block_size

        kernel((grid_size,), (block_size,),
               (coarse_field, fine_field, ny_fine, nx_fine))
        return fine_field

    def prolongate_cell(self,coarse_field, fine_field=None, method='injection'):
        """Prolongate cell-centered field to fine grid."""
        if method == 'injection':
            kernel = self.kernels.get_function('prolongate_cell_injection')
        elif method == 'bilinear':
            kernel = self.kernels.get_function('prolongate_cell_bilinear')
        else:
            raise TypeError('Valid prolongation methods: [injection, bilinear]')

        ny, nx = coarse_field.shape
        ny_fine = ny * 2
        nx_fine = nx * 2

        if fine_field is None:
            fine_field = cp.empty((ny_fine, nx_fine), dtype=cp.float32)

        total_work = ny_fine * nx_fine
        block_size = 256
        grid_size = (total_work + block_size - 1) // block_size

        kernel((grid_size,), (block_size,),
               (coarse_field, fine_field, ny_fine, nx_fine))
        return fine_field    


class FASCDScratch:
    def __init__(self,grid):
        ny,nx = grid.ny,grid.nx
        
        self.w_u = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.w_v = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.w_H = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)

        self.y_u = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.y_v = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.y_H = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)
        
        self.z_u = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.z_v = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.z_H = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)

        self.chi = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)
        self.phi = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)

@dataclass
class FASCDLevel:
    grid: Grid
    scratch: FASCDScratch

@dataclass
class FASCDConfig:
    freeze_coarse_calving: bool = True
    freeze_coarse_phi: bool = True
    coarsest_steps: int = 200
    pre_steps: int = 10
    post_steps: int = 20
    finest_steps: int = 50

class FASCDSolver:
    def __init__(self,multigrid):
        self.multigrid = multigrid
        self.levels = [FASCDLevel(grid, FASCDScratch(grid)) for grid in multigrid.grids]
        self.config = FASCDConfig()
        self.n_levels = len(self.levels)
        self.dt = None

    def solve(self,dt,max_iter=10,rel_tol=1e-3,abs_tol=10.0):
        self.dt = cp.float32(dt)
        self.multigrid.grids[0].forward_operators.set_rhs(dt)
        ru_0,rv_0,rH_0 = self.multigrid.grids[0].forward_operators.compute_residual(dt,return_norms=True)
        rel_err = 1.0
        abs_err_0 = abs_err = cp.sqrt(ru_0**2 + rv_0**2 + rH_0**2)
        iteration = 0
        print(ru_0,rv_0,rH_0)
        while rel_err > rel_tol and abs_err > abs_tol and iteration < max_iter:
            self.vcycle(0)
            ru,rv,rH = self.multigrid.grids[0].forward_operators.compute_residual(dt,freeze_phi=True,return_norms=True)
            abs_err = cp.sqrt(ru**2 + rv**2 + rH**2)
            rel_err = abs_err / abs_err_0
            iteration += 1
            print(iteration,rel_err,abs_err,ru,rv,rH)

        
    def vcycle(self, l):
        """
        FASCD V-cycle for the coupled SSA + mass conservation system.

        Full Approximation Scheme with Constrained Descent handles the
        thickness inequality constraint H >= gamma via an active set method.

        Parameters
        ----------
        l : Current grid level
        """
        finest = l == 0
        coarse = l > 0
        mg = self.multigrid
        dt = self.dt
        level = self.levels[l]
        
        if l == 0:
            level.scratch.w_u[:,:] = level.grid.state.u.data[:,:]
            level.scratch.w_v[:,:] = level.grid.state.v.data[:,:]
            level.scratch.w_H[:,:] = level.grid.state.H.data[:,:]
            level.scratch.chi[:,:] = level.grid.geometry.thklim.value - level.grid.state.H.data

        if l == self.n_levels - 1:
            # Coarsest level: direct solve
            level.grid.forward_operators.gamma[:,:] = level.scratch.w_H[:,:] + level.scratch.chi[:,:]
            level.grid.forward_operators.vanka_sweep(self.dt,self.config.coarsest_steps,freeze_calving=self.config.freeze_coarse_calving,freeze_phi=self.config.freeze_coarse_phi)
            level.grid.forward_operators.gamma.fill(level.grid.geometry.thklim.value)
            return

        next_level = self.levels[l+1]

        # Restrict constraint defect
        mg.restrict_cell(level.scratch.chi, next_level.scratch.chi, method='max')

        # Prolongate and compute local constraint adjustment
        mg.prolongate_cell(-next_level.scratch.chi, level.scratch.phi, method='injection')
        level.scratch.phi[:,:] += level.scratch.chi

        # Pre-smooth with local constraint
        level.grid.forward_operators.gamma[:, :] = level.scratch.w_H + level.scratch.phi
        level.grid.forward_operators.vanka_sweep(self.dt,self.config.pre_steps,
                freeze_calving=coarse and self.config.freeze_coarse_calving,
                freeze_phi=coarse and self.config.freeze_coarse_phi)
        level.grid.forward_operators.gamma.fill(level.grid.geometry.thklim.value)

        # Compute coarse grid correction
        level.scratch.y_u[:,:] = level.grid.state.u.data - level.scratch.w_u
        level.scratch.y_v[:,:] = level.grid.state.v.data - level.scratch.w_v
        level.scratch.y_H[:,:] = level.grid.state.H.data - level.scratch.w_H

        # Restrict solution to child
        mg.restrict_state(level.grid,next_level.grid)
        next_level.scratch.w_u[:,:] = next_level.grid.state.u.data[:,:]
        next_level.scratch.w_v[:,:] = next_level.grid.state.v.data[:,:]
        next_level.scratch.w_H[:,:] = next_level.grid.state.H.data[:,:]

        # Compute and restrict residual
        level.grid.forward_operators.compute_residual(dt, use_mask=False, 
                freeze_calving=coarse and self.config.freeze_coarse_calving,
                freeze_phi=True)
        mg.restrict_residual(level.grid,next_level.grid)

        # Form coarse grid RHS: f_c = F_c(I_h^H u_h) - I_h^H r_h
        next_level.grid.forward_operators.compute_residual(dt, use_mask=False, 
                operator_only=True, 
                freeze_calving=self.config.freeze_coarse_calving,
                freeze_phi=self.config.freeze_coarse_phi)

        next_level.grid.forward_operators.f_u[:,:] = next_level.grid.forward_operators.F_u[:,:] - next_level.grid.forward_operators.r_u[:,:]
        next_level.grid.forward_operators.f_v[:,:] = next_level.grid.forward_operators.F_v[:,:] - next_level.grid.forward_operators.r_v[:,:]
        next_level.grid.forward_operators.f_H[:,:] = next_level.grid.forward_operators.F_H[:,:] - next_level.grid.forward_operators.r_H[:,:]

        # Recursive call
        self.vcycle(l+1)

        # Compute coarse correction
        next_level.scratch.z_u[:] = next_level.grid.state.u.data - next_level.scratch.w_u
        next_level.scratch.z_v[:] = next_level.grid.state.v.data - next_level.scratch.w_v
        next_level.scratch.z_H[:] = next_level.grid.state.H.data - next_level.scratch.w_H

        # Prolongate correction
        mg.prolongate_vfacet(next_level.scratch.z_u,level.scratch.z_u,method='bilinear')
        mg.prolongate_hfacet(next_level.scratch.z_v,level.scratch.z_v,method='bilinear')
        mg.prolongate_cell(next_level.scratch.z_H,level.scratch.z_H,method='injection')

        # Apply correction
        level.scratch.z_u[:,:] += level.scratch.y_u[:,:]
        level.scratch.z_v[:,:] += level.scratch.y_v[:,:]
        level.scratch.z_H[:,:] += level.scratch.y_H[:,:]

        level.grid.state.u.data[:,:] = level.scratch.w_u + level.scratch.z_u
        level.grid.state.v.data[:,:] = level.scratch.w_v + level.scratch.z_v
        level.grid.state.H.data[:,:] = level.scratch.w_H + level.scratch.z_H

        # Post-smooth
        level.grid.forward_operators.gamma[:, :] = level.scratch.w_H + level.scratch.chi
        level.grid.forward_operators.vanka_sweep(self.dt,self.config.post_steps,
                freeze_calving=coarse and self.config.freeze_coarse_calving,
                freeze_phi=coarse and self.config.freeze_coarse_phi)
        level.grid.forward_operators.gamma.fill(level.grid.geometry.thklim.value)

        if not coarse:
            level.grid.forward_operators.vanka_sweep(self.dt,self.config.finest_steps,freeze_phi=False,freeze_calving=False)

"""
    def adjoint_vcycle_fas(grid,
                           verbose=False,
                           finest=False,
                           omega=cp.float32(1.0),
                           pre_steps=10,
                           post_steps=30,
                           final_steps=100,
                           coarse_steps=200):
        kernels = grid.kernels

        # --- Coarsest level ---
        if grid.child is None:
            grid.vanka_sweep_adjoint(coarse_steps, omega=omega,verbose=verbose)
            return

        # =========================
        # 1) Pre-smooth on fine
        # =========================
        grid.vanka_sweep_adjoint(pre_steps, omega=omega,verbose=verbose)
        restrict_adjoint_solution(grid)
        grid.child.Lambda_0[:] = grid.child.Lambda[:]

        # =========================
        # 4) Build coarse RHS via tau-correction:
        #    f_2h = R f_h + tau_2h
        #    tau_2h = N_2h(R lambda_h) - R N_h(lambda_h)
        # =========================

        grid.compute_residual_adjoint(use_mask=False)
        restrict_adjoint_residual(grid)
        
        grid.child.compute_F_adjoint(use_mask=False)
        grid.child.f_adj[:] = grid.child.r_adj - grid.child.F_adj


        # =========================
        # 5) Recurse on coarse: solve N_2h(lambda_2h) = f_2h
        # =========================
        adjoint_vcycle_fas(grid.child,
                           verbose=verbose,
                           omega=omega,
                           pre_steps=pre_steps,
                           post_steps=post_steps,
                           coarse_steps=coarse_steps)


        # =========================
        # 6) Prolongate CORRECTION (difference) and apply:
        #    lambda_h <- lambda_h + P( lambda_2h - R lambda_h )
        # =========================

        # Form coarse correction delta_2h = lambda_2h(new) - lambda_2h^0
        # Need coarse scratch arrays for delta_* (or reuse existing)
        grid.child.delta_u[:] = grid.child.lambda_u - grid.child.lambda_u_0
        grid.child.delta_v[:] = grid.child.lambda_v - grid.child.lambda_v_0
        grid.child.delta_H[:] = grid.child.lambda_H - grid.child.lambda_H_0

        # Prolongate delta_2h to fine into grid.z_*
        grid.z_u.fill(0.0); grid.z_v.fill(0.0); grid.z_H.fill(0.0)
        prolongate_vfacet(grid.child.delta_u, kernels, u_fine=grid.z_u, smooth=True)
        prolongate_hfacet(grid.child.delta_v, kernels, v_fine=grid.z_v, smooth=True)
        prolongate_cell_centered(grid.child.delta_H, kernels, H_fine=grid.z_H, smooth=True)

        # Apply fine correction
        grid.lambda_u[:] += grid.z_u
        grid.lambda_v[:] += grid.z_v
        grid.lambda_H[:] += grid.z_H

        # =========================
        # 7) Post-smooth
        # =========================
        grid.vanka_sweep_adjoint(post_steps, omega=omega, verbose=verbose)

        if finest:
            grid.vanka_sweep_adjoint(final_steps,omega=omega,verbose=verbose)
"""
