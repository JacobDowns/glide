import cupy as cp
from .io import VTIWriter


class VankaLogger:
    def __init__(self,grid,level,pvd_directory=None,pvd_base='forward'):
        self.writer = pvd_directory
        if pvd_directory:
            self.writer = VTIWriter(pvd_directory, base=pvd_base, dx=grid.dx)
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
    def __init__(self, grid, pvd_directory=None, pvd_base='forward'):
        self.i = 0
        self.writer = None
        if pvd_directory is not None:
            self.writer = VTIWriter(pvd_directory, base=pvd_base, dx=grid.dx)
        self.grid = grid
        
    def __call__(self,t):
        u_c = 0.5*(self.grid.state.u.data[:,1:] 
            + self.grid.state.u.data[:,:-1])
        v_c = 0.5*(self.grid.state.v.data[1:] 
            + self.grid.state.v.data[:-1])
        self.writer.write_step(self.i, t, {
            'u': [u_c*(1-self.grid.state.mask.data),v_c*(1-self.grid.state.mask.data)],
            'H': self.grid.state.H.data,
            'S': self.grid.state.H.data + cp.maximum(self.grid.geometry.bed.data,-0.917*self.grid.state.H.data)}
        )
        self.writer.write_pvd()
        self.i += 1

class InverseLogger:
    def __init__(self, grid, pvd_directory=None, pvd_base='inverse'):
        self.writer = None
        if pvd_directory is not None:
            self.writer = VTIWriter(pvd_directory, base=pvd_base, dx=grid.dx)
        self.grid = grid
        
    def __call__(self,i):
        u_c = 0.5*(self.grid.state.u.data[:,1:] 
            + self.grid.state.u.data[:,:-1])
        v_c = 0.5*(self.grid.state.v.data[1:] 
            + self.grid.state.v.data[:-1])
        self.writer.write_step(i, i, {
            'u': [u_c*(1-self.grid.state.mask.data),v_c*(1-self.grid.state.mask.data)],
            'beta': self.grid.sliding.beta.data}
        )
        self.writer.write_pvd()

