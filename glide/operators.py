import cupy as cp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from .closure import eval_grounded_fraction_kernel

class ForwardOperators:
    def __init__(self,grid,
            use_fast_math=True,
            use_subgrid_bed_closure=False):

        self.grid = grid

        cuda_dir = Path(__file__).parent / "cuda"

        # Concatenate ice kernel files in dependency order
        cuda_files = ['common.cu', 'viscosity.cu', 'stress.cu', 'flux.cu',
                          'residuals.cu', 'vanka.cu', 'grad.cu']
        cuda_source = '\n'.join((cuda_dir / f).read_text() for f in cuda_files)
        
        if use_fast_math:
            options=("--use_fast_math",)
        else:
            options=()

        self.kernels = cp.RawModule(code=cuda_source, options=options)

        self.r_u = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.r_v = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.r_H = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)
        
        self.f_u = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.f_v = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.f_H = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)
        
        self.F_u = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.F_v = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.F_H = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)

        self.delta_u = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.delta_v = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.delta_H = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)

        self.gamma = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)
        self.gamma.fill(grid.geometry.thklim.value)

        self.vanka_config = VankaConfig()
        self.use_subgrid_bed_closure = use_subgrid_bed_closure

    @property
    def _kernel_config(self):
        block_size = (16, 16)
        stride = 14
        halo = 1
        grid_size = (self.grid.nx // stride + 1, self.grid.ny // stride + 1)
        return grid_size, block_size, stride, halo

    def compute_residual(self, dt, 
            use_mask=True, 
            operator_only=False, 
            freeze_calving=False, 
            freeze_phi=False,
            return_norms=False):

        kernel = self.kernels.get_function('compute_residual')
        grid_size, block_size, stride, halo = self._kernel_config
  
        grid = self.grid
        state = grid.state
        geometry = grid.geometry        
        rheology = grid.rheology
        sliding = grid.sliding
        calving = grid.calving
        forcing = grid.forcing

        if freeze_calving:
            calving_rate = cp.float32(0.0)
        else:
            calving_rate = calving.calving_rate.value

        if not freeze_phi:
            self.compute_phi()

        if operator_only:
            out_u = self.F_u
            out_v = self.F_v
            out_H = self.F_H
            use_forcing = False
        else:
            out_u = self.r_u
            out_v = self.r_v
            out_H = self.r_H
            use_forcing = True

        kernel(grid_size, block_size,
               (out_u, out_v, out_H,
                state.u.data, state.v.data, state.H.data, 
                state.phi.data, state.mask.data,
                self.f_u, self.f_v, self.f_H,
                geometry.bed.data, 
                rheology.B.data, 
                sliding.beta.data,
                self.gamma,
                use_forcing,use_mask,
                rheology.n.value, rheology.eps_reg.value, 
                geometry.flotation_reg_driving.value,
                sliding.m.value, sliding.u_reg.value, 
                sliding.water_drag.value, sliding.flotation_reg_sliding.value,
                calving_rate, calving.flotation_reg_calving.value,
                grid.dx, dt,
                grid.ny, grid.nx, stride, halo)) 
        if return_norms:
            return cp.linalg.norm(out_u),cp.linalg.norm(out_v),cp.linalg.norm(out_H)

    def compute_phi(self, relaxation=cp.float32(0.0)):
        if self.use_subgrid_bed_closure:
            self.grid.state.phi.data[:,:] = eval_grounded_fraction_kernel(
                self.grid.state.H.data,
                self.grid.geometry.bed.quantiles,
                cp.float32(0.0)
                )

        else:
            kernel = self.kernels.get_function('compute_grounded')
            grid_size, block_size, stride, halo = self._kernel_config
            
            grid = self.grid
            kernel(grid_size, block_size,
                   (grid.state.phi.data,
                    grid.state.H.data, grid.geometry.bed.data, 
                    grid.geometry.flotation_reg_driving.value,
                    relaxation,
                    grid.ny, grid.nx, 
                    stride, halo))

    def vanka_smooth(self, dt,
            freeze_calving=False,
            freeze_phi=False):

        kernel = self.kernels.get_function('vanka_smooth')
        grid_size, block_size, stride, halo = self._kernel_config

        grid = self.grid
        state = grid.state
        geometry = grid.geometry        
        rheology = grid.rheology
        sliding = grid.sliding
        calving = grid.calving
        forcing = grid.forcing

        if freeze_calving:
            calving_rate = cp.float32(0.0)
        else:
            calving_rate = calving.calving_rate.value

        if not freeze_phi:
            self.compute_phi(relaxation=self.vanka_config.relax_phi)

        self.delta_u.fill(0.0)
        self.delta_v.fill(0.0)
        self.delta_H.fill(0.0)
        kernel(grid_size, block_size,
               (self.delta_u, self.delta_v, self.delta_H, 
                state.mask.data,
                state.u.data, state.v.data, state.H.data, 
                state.phi.data,
                self.f_u, self.f_v, self.f_H,
                geometry.bed.data, rheology.B.data, sliding.beta.data, 
                self.gamma,
                rheology.n.value, rheology.eps_reg.value, 
                geometry.flotation_reg_driving.value,
                sliding.m.value, sliding.u_reg.value, 
                sliding.water_drag.value, 
                sliding.flotation_reg_sliding.value,
                calving_rate, 
                calving.flotation_reg_calving.value,
                grid.dx, dt,
                grid.ny, grid.nx, stride, halo,
                self.vanka_config.newton_config.steps,
                self.vanka_config.newton_config.relaxation,
                self.vanka_config.newton_config.ssa_damping,
                self.vanka_config.newton_config.mc_damping)
        )

    def vanka_sweep(self, dt, n_iter, 
            freeze_calving=False,
            freeze_phi=False):
        for i in range(n_iter):
            self.vanka_smooth(dt,freeze_calving=freeze_calving,freeze_phi=freeze_phi)
            self.grid.state.u.data[:] += self.vanka_config.omega * self.delta_u
            self.grid.state.v.data[:] += self.vanka_config.omega * self.delta_v
            self.grid.state.H.data[:] += self.vanka_config.omega * self.delta_H
            self.vanka_config.hook_func(i)

    def vanka_dump(self,dt):
        kernel = self.kernels.get_function('vanka_dump')
        grid_size, block_size, stride, halo = self._kernel_config
        grid = self.grid

        J = cp.zeros((self.grid.ny*self.grid.nx,25),dtype=cp.float32)
        r = cp.zeros((self.grid.ny*self.grid.nx,5),dtype=cp.float32)
        kernel(grid_size, block_size,
               (J,r,
                grid.state.u.data, grid.state.v.data, grid.state.H.data, grid.state.phi.data,
                self.f_u, self.f_v, self.f_H,
                grid.geometry.bed.data, grid.rheology.B.data, grid.sliding.beta.data, self.gamma,
                grid.rheology.n.value, grid.rheology.eps_reg.value, grid.geometry.flotation_reg_driving.value,
                grid.sliding.m.value, grid.sliding.u_reg.value, grid.sliding.water_drag.value, grid.sliding.flotation_reg_sliding.value,
                grid.calving.calving_rate.value, grid.calving.flotation_reg_calving.value,
                grid.dx, dt,
                grid.ny, grid.nx, stride, halo,
                )
        )

        return J,r
            
    def set_rhs(self,dt):
        self.f_H[:,:] = self.grid.state.H_prev.data/dt + self.grid.forcing.smb.data

@dataclass
class NewtonConfig:
    steps: int = 30
    relaxation: cp.float32 = cp.float32(0.5)
    ssa_damping: cp.float32 = cp.float32(0.01)
    mc_damping: cp.float32 = cp.float32(1.0)

@dataclass
class VankaConfig:
    omega: cp.float32 = cp.float32(0.5)
    newton_config: NewtonConfig = field(default_factory = lambda: NewtonConfig())
    relax_phi: cp.float32 = cp.float32(0.0)
    hook_interval: int = 1
    hook_func: Callable[[int],None] = field(default_factory = lambda: lambda i:None)


