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
    def __init__(self,grid, pvd_directory=None,pvd_base='forward'):
        self.writer = None
        if pvd_directory is not None:
            self.writer = VTIWriter(pvd_directory, base=pvd_base, dx=grid.dx)
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
