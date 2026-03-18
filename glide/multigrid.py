import cupy as cp
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from .grid import Grid
from .operators import VankaConfig,NewtonConfig
from .field import LocalOption,BroadcastOption

class Multigrid:
    def __init__(self,n_levels: int,finest_grid=None,ny=None,nx=None,dx=None,use_fast_math=True):

        cuda_dir = Path(__file__).parent / "cuda"

        # Concatenate ice kernel files in dependency order
        cuda_files = ['common.cu', 'transfer.cu']
        cuda_source = '\n'.join((cuda_dir / f).read_text() for f in cuda_files)
        
        if use_fast_math:
            options=("--use_fast_math",)
        else:
            options=()

        self.kernels = cp.RawModule(code=cuda_source, options=options)

        if finest_grid is not None:
            print("Instantiating multigrid from existing grid")
            self.finest_grid = finest_grid
        else:
            print("Instantiating multigrid from new grid")
            self.finest_grid = Grid(ny,nx,dx)

        self.n_levels = n_levels

        if n_levels is not None:
            self.create_grid_hierarchy(n_levels,restrict_fields=True)

        self.state = MGStateManager(self)
        self.geometry = MGGeometryManager(self)
        self.rheology = MGRheologyManager(self)
        self.sliding = MGSlidingManager(self)
        self.calving = MGCalvingManager(self)
        self.forcing = MGForcingManager(self)

        self._adjoint = None

    @property
    def adjoint(self):
        if self._adjoint is None:
            self._adjoint = MGAdjointManager(self)
        return self._adjoint

    def create_grid_hierarchy(self,n_levels,restrict_fields=True):
        self.levels = [self.finest_grid]
        for i in range(1,n_levels):
            coarse_grid = self.create_coarse_grid(self.levels[-1],
                restrict_fields=restrict_fields)
            self.levels.append(coarse_grid)
        return self.levels

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

class HierarchyFieldManager:
    def __init__(self, levels, getter, restrict,name=None):
        self._levels = levels
        self._getter = getter
        self._restrict = restrict
        self._name = name

    def set(self, value):
        finest = self._getter(self._levels[0])
        finest.set(value)
        self.restrict_down()

    def restrict_down(self):
        for l in range(len(self._levels) - 1):
            fine = self._getter(self._levels[l])
            coarse = self._getter(self._levels[l + 1])
            self._restrict(fine, coarse)

    def set_level(self, level, value):
        self._getter(self._levels[level].grid).set(value)

class MGStateManager:
    def __init__(self, mg):
        self.mg = mg
        self.u = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.state.u,
            restrict=lambda f,c: mg.restrict_vfacet(f.data,c.data),
            name="u",
        )

        self.v = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.state.v,
            restrict=lambda f,c: mg.restrict_hfacet(f.data,c.data),
            name="v",
        )

        self.H = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.state.H,
            restrict=lambda f,c: mg.restrict_cell(f.data,c.data,method='avg'),
            name="H",
        )
        
        self.H_prev = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.state.H_prev,
            restrict=lambda f,c: mg.restrict_cell(f.data,c.data,method='avg'),
            name="H_prev",
        )

        self.phi = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.state.phi,
            restrict=lambda f,c: mg.restrict_cell(f.data,c.data,method='avg'),
            name="phi",
        )

        self.mask = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.state.mask,
            restrict=lambda f,c: mg.restrict_cell(f.data,c.data,method='max'),
            name="mask",
        )

    def __repr__(self):
        return f'Top-level ({self.mg.n_levels} levels): \n'+self.mg.levels[0].state.__repr__()

class MGAdjointManager:
    def __init__(self, mg):
        self.mg = mg
        self.lambda_u = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.adjoint.lambda_u,
            restrict=lambda f,c: mg.restrict_vfacet(f.data,c.data),
            name="lambda_u",
        )

        self.lambda_v = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.adjoint.lambda_v,
            restrict=lambda f,c: mg.restrict_hfacet(f.data,c.data),
            name="lambda_v",
        )

        self.lambda_H = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.adjoint.lambda_H,
            restrict=lambda f,c: mg.restrict_cell(f.data,c.data,method='avg'),
            name="lambda_H",
        )
        
    def __repr__(self):
        return f'Top-level ({self.mg.n_levels} levels): \n'+self.mg.levels[0].adjoint.__repr__()

class MGGeometryManager:
    def __init__(self, mg):
        self.mg = mg
        self.bed = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.geometry.bed,
            restrict=lambda f,c: mg.restrict_cell(f.data,c.data,method='avg'),
            name="bed",
        )

        self.flotation_reg_driving = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.geometry.flotation_reg_driving,
            restrict=lambda f,c: c.set(f.value),
            name="flotation_reg_driving",
        )

        self.thklim = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.geometry.thklim,
            restrict=lambda f,c: c.set(f.value),
            name="thklim",
        )
    def __repr__(self):
        return f'Top-level ({self.mg.n_levels} levels): \n'+self.mg.levels[0].geometry.__repr__()

class MGRheologyManager:
    def __init__(self, mg):
        self.mg = mg
        self.B = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.rheology.B,
            restrict=lambda f,c: mg.restrict_cell(f.data,c.data,method='avg'),
            name="B",
        )

        self.n = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.rheology.n,
            restrict=lambda f,c: c.set(f.value),
            name="n",
        )

        self.eps_reg = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.rheology.eps_reg,
            restrict=lambda f,c: c.set(f.value),
            name="eps_reg",
        )
    def __repr__(self):
        return f'Top-level ({self.mg.n_levels} levels): \n'+self.mg.levels[0].rheology.__repr__()

class MGSlidingManager:
    def __init__(self, mg):
        self.mg = mg
        self.beta = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.sliding.beta,
            restrict=lambda f,c: mg.restrict_cell(f.data,c.data,method='avg'),
            name="beta",
        )

        self.m = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.sliding.m,
            restrict=lambda f,c: c.set(f.value),
            name="m",
        )
        
        self.u_reg = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.sliding.u_reg,
            restrict=lambda f,c: c.set(f.value),
            name="u_reg",
        )
        
        self.water_drag = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.sliding.water_drag,
            restrict=lambda f,c: c.set(f.value),
            name="water_drag",
        )
        
        self.flotation_reg_sliding = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.sliding.flotation_reg_sliding,
            restrict=lambda f,c: c.set(f.value),
            name="flotation_reg_sliding",
        )

    def __repr__(self):
        return f'Top-level ({self.mg.n_levels} levels): \n'+self.mg.levels[0].sliding.__repr__()

class MGCalvingManager:
    def __init__(self, mg):
        self.mg = mg
        self.calving_rate = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.calving.calving_rate,
            restrict=lambda f,c: c.set(f.value),
            name="calving_rate",
        )
        
        self.flotation_reg_calving = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.calving.flotation_reg_calving,
            restrict=lambda f,c: c.set(f.value),
            name="flotation_reg_calving",
        )

    def __repr__(self):
        return f'Top-level ({self.mg.n_levels} levels): \n'+self.mg.levels[0].calving.__repr__()

class MGForcingManager:
    def __init__(self, mg):
        self.mg = mg
        self.smb = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.forcing.smb,
            restrict=lambda f,c: mg.restrict_cell(f.data,c.data,method='avg'),
            name="smb",
        )

    def __repr__(self):
        return f'Top-level ({self.mg.n_levels} levels): \n'+self.mg.levels[0].forcing.__repr__()

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



class FASCDSolver:
    def __init__(self,multigrid):
        self.multigrid = multigrid
        self.levels = [FASCDLevel(grid, FASCDScratch(grid)) for grid in multigrid.levels]

        self._fas_config = FASCDConfig()
        self.fas_options = FASCDOptions(self._fas_config)
        
        self.vanka_options = VankaOptions(
            self.levels,
            getter=lambda lev: lev.grid.forward_operators.vanka_config,
        )

        self.n_levels = len(self.levels)
        self.dt = None

    def solve(self,dt,start_level=0,report_norms=True):
        self.dt = cp.float32(dt)
        
        start_level_ = self.multigrid.levels[start_level]
        start_level_.forward_operators.set_rhs(dt)
        
        ru_init,rv_init,rH_init = start_level_.forward_operators.compute_residual(dt,return_norms=True)
        initial_residual_norm = cp.sqrt(ru_init**2 + rv_init**2 + rH_init**2)
        relative_residual_norm = cp.float32(1.0)

        if report_norms:
            print(f"  Initial: |r| = {initial_residual_norm:.4e}, "
                  f"|r_u| = {float(ru_init):.4e}, "
                  f"|r_v| = {float(rv_init):.4e}, "
                  f"|r_H| = {float(rH_init):.4e}")

        absolute_residual_norm = initial_residual_norm
        iteration = 0
        while (relative_residual_norm > self._fas_config.relative_tolerance 
                and absolute_residual_norm > self._fas_config.absolute_tolerance
                and iteration < self._fas_config.maximum_vcycles):
            self.vcycle(start_level,finest=True)
            ru,rv,rH = start_level_.forward_operators.compute_residual(dt,freeze_phi=True,return_norms=True)

            absolute_residual_norm = cp.sqrt(ru**2 + rv**2 + rH**2)
            relative_residual_norm = absolute_residual_norm / initial_residual_norm
            if report_norms:
                print(f"  V-cycle {iteration}: |r|/|r0| = {relative_residual_norm:.4e}, "
                      f"|r_u| = {float(ru):.4e}, "
                      f"|r_v| = {float(rv):.4e}, "
                      f"|r_H| = {float(rH):.4e}")
            iteration += 1

        
    def vcycle(self, l, finest=False):
        """
        FASCD V-cycle for the coupled SSA + mass conservation system.

        Full Approximation Scheme with Constrained Descent handles the
        thickness inequality constraint H >= gamma via an active set method.

        Parameters
        ----------
        l : Current grid level
        """
        coarse = not finest
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
            level.grid.forward_operators.vanka_sweep(self.dt,
                self._fas_config.coarsest_steps,
                freeze_calving=self._fas_config.freeze_coarse_calving,
                freeze_phi=self._fas_config.freeze_coarse_phi
            )
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
        level.grid.forward_operators.vanka_sweep(self.dt,self._fas_config.pre_steps,
                freeze_calving=coarse and self._fas_config.freeze_coarse_calving,
                freeze_phi=coarse and self._fas_config.freeze_coarse_phi)
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
                freeze_calving=coarse and self._fas_config.freeze_coarse_calving,
                freeze_phi=True)
        mg.restrict_residual(level.grid,next_level.grid)

        # Form coarse grid RHS: f_c = F_c(I_h^H u_h) - I_h^H r_h
        next_level.grid.forward_operators.compute_residual(dt, use_mask=False, 
                operator_only=True, 
                freeze_calving=self._fas_config.freeze_coarse_calving,
                freeze_phi=self._fas_config.freeze_coarse_phi)

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
        level.grid.forward_operators.vanka_sweep(self.dt,self._fas_config.post_steps,
                freeze_calving=coarse and self._fas_config.freeze_coarse_calving,
                freeze_phi=coarse and self._fas_config.freeze_coarse_phi)
        level.grid.forward_operators.gamma.fill(level.grid.geometry.thklim.value)

        if not coarse:
            level.grid.forward_operators.vanka_sweep(self.dt,
                self._fas_config.finest_steps,
                freeze_phi=False,
                freeze_calving=False)

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
    maximum_vcycles: int = 10
    relative_tolerance: cp.float32 = cp.float32(1e-3)
    absolute_tolerance: cp.float32 = cp.float32(5.0)

def dataclass_field_names(cls):
    return [f.name for f in fields(cls)]

def validate_kwargs(config_cls, kwargs):
    valid = set(dataclass_field_names(config_cls))
    unknown = set(kwargs) - valid
    if unknown:
        raise AttributeError(
            f"Unknown {config_cls.__name__} option(s): {sorted(unknown)}. "
            f"Valid options are: {sorted(valid)}"
        )

class FASCDOptions:
    """
    User-facing wrapper around a single FASCDConfig.
    """

    def __init__(self, config: FASCDConfig):
        self._config = config

        self.options = ['freeze_coarse_calving',
            'freeze_coarse_phi',
            'coarsest_steps',
            'pre_steps',
            'post_steps',
            'finest_steps']

        self.freeze_coarse_calving = LocalOption(
            getter=lambda: self._config.freeze_coarse_calving,
            setter=lambda v: setattr(self._config, "freeze_coarse_calving", v),
            name="freeze_coarse_calving",
        )
        self.freeze_coarse_phi = LocalOption(
            getter=lambda: self._config.freeze_coarse_phi,
            setter=lambda v: setattr(self._config, "freeze_coarse_phi", v),
            name="freeze_coarse_phi",
        )
        self.coarsest_steps = LocalOption(
            getter=lambda: self._config.coarsest_steps,
            setter=lambda v: setattr(self._config, "coarsest_steps", v),
            name="coarsest_steps",
        )
        self.pre_steps = LocalOption(
            getter=lambda: self._config.pre_steps,
            setter=lambda v: setattr(self._config, "pre_steps", v),
            name="pre_steps",
        )
        self.post_steps = LocalOption(
            getter=lambda: self._config.post_steps,
            setter=lambda v: setattr(self._config, "post_steps", v),
            name="post_steps",
        )
        self.finest_steps = LocalOption(
            getter=lambda: self._config.finest_steps,
            setter=lambda v: setattr(self._config, "finest_steps", v),
            name="finest_steps",
        )

        self.maximum_vcycles = LocalOption(
            getter=lambda: self._config.maximum_vcycles,
            setter=lambda v: setattr(self._config, "maximum_vcycles", v),
            name="maximum_vcyles",
        )
        
        self.relative_tolerance = LocalOption(
            getter=lambda: self._config.relative_tolerance,
            setter=lambda v: setattr(self._config, "relative_tolerance", v),
            name="relative_tolerance",
        )
        
        self.absolute_tolerance = LocalOption(
            getter=lambda: self._config.absolute_tolerance,
            setter=lambda v: setattr(self._config, "absolute_tolerance", v),
            name="absolute_tolerance",
        )



    def set(self, **kwargs):
        validate_kwargs(FASCDConfig, kwargs)
        for k, v in kwargs.items():
            setattr(self._config, k, v)

    def __repr__(self):
        return 'fas_options={' + ', '.join([f"{getattr(self,o)}" for o in self.options]) + '}'

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(self.options))



class VankaOptions:
    """
    Broadcast wrapper around VankaConfig across all solver levels.
    """

    def __init__(self, levels, getter):
        self._levels = levels
        self._getter = getter  # level -> VankaConfig

        self.options = ['omega','relax_phi']

        self.omega = BroadcastOption(self._levels, self._getter, "omega")
        self.relax_phi = BroadcastOption(self._levels, self._getter, "relax_phi")

        self.newton_options = NewtonOptions(
            self._levels,
            getter=lambda lev: self._getter(lev).newton_config,
        )

    def set(self, **kwargs):
        validate_kwargs(VankaConfig, kwargs)
        for lev in self._levels:
            cfg = self._getter(lev)
            for k, v in kwargs.items():
                setattr(cfg, k, v)

    def set_level(self, level_index: int, **kwargs):
        validate_kwargs(VankaConfig, kwargs)
        cfg = self._getter(self._levels[level_index])
        for k, v in kwargs.items():
            setattr(cfg, k, v)

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(self.options))

    def __repr__(self):
        return 'vanka_options={' + ', '.join([f"{getattr(self,o)}" for o in self.options]) + '}'



class NewtonOptions:
    """
    Broadcast wrapper around NewtonConfig across all solver levels.
    """

    def __init__(self, levels, getter):
        self._levels = levels
        self._getter = getter  # level -> NewtonConfig
        self.options = ['steps','relaxation','ssa_damping','mc_damping']

        self.steps = BroadcastOption(self._levels, self._getter, "steps")
        self.relaxation = BroadcastOption(self._levels, self._getter, "relaxation")
        self.ssa_damping = BroadcastOption(self._levels, self._getter, "ssa_damping")
        self.mc_damping = BroadcastOption(self._levels, self._getter, "mc_damping")

    def set(self, **kwargs):
        validate_kwargs(NewtonConfig, kwargs)
        for lev in self._levels:
            cfg = self._getter(lev)
            for k, v in kwargs.items():
                setattr(cfg, k, v)

    def set_level(self, level_index: int, **kwargs):
        validate_kwargs(NewtonConfig, kwargs)
        cfg = self._getter(self._levels[level_index])
        for k, v in kwargs.items():
            setattr(cfg, k, v)

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(self.options))

    def __repr__(self):
        return 'newton_options={' + ', '.join([f"{getattr(self,o)}" for o in self.options]) + '}'



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
