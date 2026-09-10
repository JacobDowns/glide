"""
Core ice physics API.

Provides the IcePhysics class that wraps the forward model and adjoint
computations into a clean interface.
"""

import cupy as cp
from .grid import Grid
from .multigrid import Multigrid, FASCDSolver,FASAdjointSolver

class IceDynamics:
    def __init__(self,mg=None,
            n_levels=None,grid=None,
            ny=None,nx=None,dx=None,
            x0=cp.float32(0.0),y0=cp.float32(0.0),crs=None,
            stress_scheme='molho'):
        # stress_scheme: 'molho' (two-field, shear-resolving) or 'ssa'
        # (deformational components pinned to zero). When an existing mg or
        # grid is supplied, its stress_scheme governs.
        if mg is not None:
            self.mg = mg
        elif grid is not None and n_levels is not None:
            self.mg = Multigrid(n_levels,finest_grid=grid)
        elif ny and nx and dx and n_levels:
            self.mg = Multigrid(n_levels,ny=ny,nx=nx,dx=dx,
                   x0=x0,y0=y0,crs=crs,stress_scheme=stress_scheme)
        else:
            raise ValueError('Must supply either (a) a multigrid object \
                              (b) a grid and number of levels \
                              (c) ny/nx/dx and number of levels')

        self._forward_solver = None
        self._adjoint_solver = None
        self.top_level = 0

        self._post_forward_hooks = []

    @property
    def forward_solver(self):
        if self._forward_solver is None:
            self._forward_solver = FASCDSolver(self.mg)
        return self._forward_solver
    
    @property
    def adjoint_solver(self):
        if self._adjoint_solver is None:
            self._adjoint_solver = FASAdjointSolver(self.mg)
        return self._adjoint_solver

    def set_top_level(self,level):
        self.top_level = level

    def register_post_forward_hook(self,hook):
        self._post_forward_hooks.append(hook)

    def forward(self,t,dt,update_geometry=True):
        self.forward_solver.solve(dt,start_level=self.top_level)
        if update_geometry:
            self.mg.levels[self.top_level].state.H_prev.data[:,:] = (
                self.mg.levels[self.top_level].state.H.data[:,:]
            )
        for f in self._post_forward_hooks:
            f(t+dt)

    def backward(self,t,dt,dJdu=None,dJdv=None,dJdud=None,dJdvd=None,dJdH=None,dJdu_s=None,
            compute_beta_grad=True,compute_bed_grad=True,
            compute_H_prev_grad=True,compute_smb_grad=True):
        """Solve the adjoint and reduce it onto the parameter gradients.

        ``dJdu``/``dJdv``/``dJdud``/``dJdvd``/``dJdH`` are cotangents on the solved state.

        ``dJdu_s`` is the cotangent on the DIVA SURFACE speed (per cell), and enters differently:
        u_s is not a state variable, so its cotangent is scattered to the velocity/thickness rows
        through the closure (diva_surface_misfit_rhs), and u_s depends on the sliding parameters
        DIRECTLY, so an explicit sum_c cot_c d(u_s)_c/dp term is added after the reduction.  DIVA
        only.  See notes/diva_numerics.md 6.5, notes/diva_adjoint_map.md."""
        ao = self.mg.levels[self.top_level].adjoint_operators

        # Surface term first: diva_surface_misfit_rhs FILLS f_u/f_v/f_H with -dJ/d(u,v,H), so the
        # depth-averaged cotangents accumulate on top of it rather than the reverse.
        if dJdu_s is not None:
            if ao.diva_surface_misfit_rhs(dJdu_s)[0] is None:
                raise ValueError(
                    "dJdu_s requires DIVA (stress_scheme='diva'): under SSA/MOLHO the surface and "
                    "depth-averaged velocities coincide, so pass the misfit through dJdu/dJdv.")
        else:
            ao.f_u.fill(0.0); ao.f_v.fill(0.0); ao.f_H.fill(0.0)
        if dJdu is not None: ao.f_u -= dJdu
        if dJdv is not None: ao.f_v -= dJdv
        if dJdH is not None: ao.f_H -= dJdH

        # Deformational rows (MOLHO only); u_s has no deformational analogue.
        if dJdud is not None: ao.f_ud[:,:] = -dJdud
        else: ao.f_ud.fill(0.0)
        if dJdvd is not None: ao.f_vd[:,:] = -dJdvd
        else: ao.f_vd.fill(0.0)

        converged = self.adjoint_solver.solve(dt,start_level=self.top_level)
        ao.compute_gradient_beta()
        ao.compute_gradient_bed()
        ao.compute_gradient_H_prev(dt)
        ao.compute_gradient_smb()

        # The explicit parameter dependence of u_s, which only a surface objective has.
        if dJdu_s is not None:
            sliding = self.mg.levels[self.top_level].sliding
            sliding.beta.grad[:] += ao.diva_surface_param_gradient(dJdu_s,'beta')

        return converged
        




        

