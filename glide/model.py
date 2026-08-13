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
            x0=cp.float32(0.0),y0=cp.float32(0.0),crs=None):
        if mg is not None:
            self.mg = mg
        elif grid is not None and n_levels is not None:
            self.mg = Multigrid(n_levels,finest_grid=grid)
        elif ny and nx and dx and n_levels:
            self.mg = Multigrid(n_levels,ny=ny,nx=nx,dx=dx,
                   x0=x0,y0=y0,crs=crs)
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

    def backward(self,t,dt,dJdu=None,dJdv=None,dJdH=None,dJdu_s=None,
            compute_beta_grad=True,compute_bed_grad=True,
            compute_H_prev_grad=True,compute_smb_grad=True,
            compute_m_grad=False,compute_u_c_grad=False):
        """Solve the adjoint and reduce it onto the parameter gradients.

        ``dJdu``/``dJdv``/``dJdH`` are cotangents on the solved state, as before.

        ``dJdu_s`` is the cotangent on the DIVA SURFACE speed, per cell.  It enters
        differently from the others and requires DIVA:

          * u_s is not a state variable, so its cotangent cannot simply be dropped into
            the right-hand side.  It is scattered to the velocity facets through the
            closure's stored dus_deps/dus_dU (diva_surface_misfit_rhs), then ADDED to
            whatever dJdu/dJdv contributed.
          * u_s depends on the sliding parameters DIRECTLY through the closure, not only
            through the state, so lambda^T dr/dp is no longer the whole gradient.  The
            explicit term sum_c cot_c * d(u_s)_c/dp is added after the reduction.

        See notes/diva_adjoint_map.md 7.3 and tests/diva_surface_gradient_test.py.
        """
        ao = self.mg.levels[self.top_level].adjoint_operators

        # Surface term first: diva_surface_misfit_rhs FILLS f_u/f_v with -dJ/d(u,v), so the
        # depth-averaged cotangents are accumulated on top of it rather than the reverse.
        if dJdu_s is not None:
            # Fills f_u, f_v AND f_H: u_s depends on thickness through the shear moments,
            # so a surface objective has a thickness right-hand side of its own.
            if ao.diva_surface_misfit_rhs(dJdu_s)[0] is None:
                raise ValueError(
                    "dJdu_s requires DIVA: under SSA the surface and depth-averaged "
                    "velocities coincide, so pass the misfit through dJdu/dJdv instead "
                    "(set mg.rheology.stress_balance to 1.0 for DIVA).")
        else:
            ao.f_u.fill(0.0)
            ao.f_v.fill(0.0)
            ao.f_H.fill(0.0)

        if dJdu is not None:
            ao.f_u -= dJdu
        if dJdv is not None:
            ao.f_v -= dJdv
        if dJdH is not None:
            ao.f_H -= dJdH

        converged = self.adjoint_solver.solve(dt,start_level=self.top_level)
        ao.compute_gradient_beta()
        ao.compute_gradient_bed()
        ao.compute_gradient_H_prev(dt)
        ao.compute_gradient_smb()
        if compute_m_grad:
            ao.compute_gradient_m()
        if compute_u_c_grad:
            ao.compute_gradient_u_c()

        # The explicit parameter dependence of u_s, which only a surface objective has.
        if dJdu_s is not None:
            sliding = self.mg.levels[self.top_level].sliding
            # Field.grad is read-only (it hands back the array), so accumulate in place.
            sliding.beta.grad[:] += ao.diva_surface_param_gradient(dJdu_s,'beta')
            if compute_u_c_grad:
                sliding.u_c.grad[:] += ao.diva_surface_param_gradient(dJdu_s,'u_c')
            if compute_m_grad:
                sliding.m.grad = sliding.m.grad + ao.diva_surface_param_gradient(dJdu_s,'m')

        return converged
        




        

