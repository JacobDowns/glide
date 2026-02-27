"""
Grid hierarchy for multigrid ice sheet solver.

Implements a MAC (marker-and-cell) staggered grid with:
- u velocities on vertical faces: shape (ny, nx+1)
- v velocities on horizontal faces: shape (ny+1, nx)
- H thickness and other scalars on cell centers: shape (ny, nx)
"""

import cupy as cp
from .kernels import make_physics_params


class Grid:
    """
    Single level of the multigrid hierarchy.

    Manages state vectors, parameters, and kernel dispatch for one grid level.
    Supports both forward simulation and adjoint-based inverse modeling.

    Parameters
    ----------
    ny, nx : int
        Grid dimensions (number of cells in y and x)
    dx : float
        Grid spacing (assumed isotropic)
    dt : float
        Time step
    kernels : module
        Compiled CUDA kernel module
    parent : Grid, optional
        Parent (finer) grid in hierarchy
    n : float
        Glen's flow law exponent (default 3.0)
    eps_reg : float
        Strain rate regularization (default 1e-5)
    water_drag : float
        Drag coefficient for floating ice (default 0.001)
    calving_rate : float
        Calving rate for mass loss at margins (default 1.0)
    """

    def __init__(self, ny, nx, dx, dt, kernels, parent=None,
                 n=3.0, eps_reg=1e-5, 
                 m=1.0, u_reg=1.0,
                 water_drag=0.001, calving_rate=1.0,
                 sigmoid_c=0.1):

        self.parent = parent
        self.child = None
        self.kernels = kernels

        self.dx = cp.float32(dx)
        self.dt = cp.float32(dt)
        self.ny = ny
        self.nx = nx

        # Degrees of freedom
        self.nu = ny * (nx + 1)
        self.nv = (ny + 1) * nx
        self.nU = self.nu + self.nv
        self.nh = ny * nx
        self.n_total = self.nu + self.nv + self.nh

        # Physics parameters (passed to CUDA kernels as struct)
        self._n = cp.float32(n)
        self._eps_reg = cp.float32(eps_reg)
        self._m = cp.float32(m)
        self._u_reg = cp.float32(u_reg)
        self._water_drag = cp.float32(water_drag)
        self._calving_rate = cp.float32(calving_rate)
        self._sigmoid_c = cp.float32(sigmoid_c)

        # Allocate state and work arrays
        self._allocate_arrays()

    def _allocate_arrays(self):
        """Allocate GPU arrays for state, residuals, and work vectors."""
        ny, nx = self.ny, self.nx

        # Previous thickness (for time stepping)
        self.H_prev = cp.zeros((ny, nx), dtype=cp.float32)

        # Primary state vector [u, v, H]
        self.U = cp.zeros(self.n_total, dtype=cp.float32)
        self.u, self.v, self.H = self._vec_to_fields(self.U)

        # Perturbation vector (for JVP)
        self.d_U = cp.zeros(self.n_total, dtype=cp.float32)
        self.d_u, self.d_v, self.d_H = self._vec_to_fields(self.d_U)

        # Update/correction vector
        self.delta_U = cp.zeros(self.n_total, dtype=cp.float32)
        self.delta_u, self.delta_v, self.delta_H = self._vec_to_fields(self.delta_U)

        # Adjoint state vector
        self.Lambda = cp.zeros(self.n_total, dtype=cp.float32)
        self.lambda_u, self.lambda_v, self.lambda_H = self._vec_to_fields(self.Lambda)
        
        self.Lambda_0 = cp.zeros(self.n_total, dtype=cp.float32)
        self.lambda_u_0, self.lambda_v_0, self.lambda_H_0 = self._vec_to_fields(self.Lambda_0)

        self.delta_Lambda = cp.zeros(self.n_total, dtype=cp.float32)
        self.delta_lambda_u, self.delta_lambda_v, self.delta_lambda_H = self._vec_to_fields(self.delta_Lambda)

        # RHS vector
        self.f = cp.zeros(self.n_total, dtype=cp.float32)
        self.f_u, self.f_v, self.f_H = self._vec_to_fields(self.f)

        # Adjoint RHS
        self.f_adj = cp.zeros(self.n_total, dtype=cp.float32)
        self.f_adj_u, self.f_adj_v, self.f_adj_H = self._vec_to_fields(self.f_adj)

        # Operator evaluation F(U)
        self.F = cp.zeros(self.n_total, dtype=cp.float32)
        self.F_u, self.F_v, self.F_H = self._vec_to_fields(self.F)

        self.F_adj = cp.zeros(self.n_total, dtype=cp.float32)
        self.F_adj_u, self.F_adj_v, self.F_adj_H = self._vec_to_fields(self.F_adj)


        # Zero vector for residual computation
        self.Z = cp.zeros(self.n_total, dtype=cp.float32)
        self.Z_u, self.Z_v, self.Z_H = self._vec_to_fields(self.Z)

        # Residual vector r = f - F(U)
        self.r = cp.zeros(self.n_total, dtype=cp.float32)
        self.r_u, self.r_v, self.r_H = self._vec_to_fields(self.r)

        # Adjoint residual
        self.r_adj = cp.zeros(self.n_total, dtype=cp.float32)
        self.r_adj_u, self.r_adj_v, self.r_adj_H = self._vec_to_fields(self.r_adj)

        # JVP output
        self.j = cp.zeros(self.n_total, dtype=cp.float32)
        self.j_u, self.j_v, self.j_H = self._vec_to_fields(self.j)

        # VJP output
        self.l = cp.zeros(self.n_total, dtype=cp.float32)
        self.l_u, self.l_v, self.l_H = self._vec_to_fields(self.l)

        # Work vectors for multigrid
        self.w = cp.zeros(self.n_total, dtype=cp.float32)
        self.w_u, self.w_v, self.w_H = self._vec_to_fields(self.w)

        self.y = cp.zeros(self.n_total, dtype=cp.float32)
        self.y_u, self.y_v, self.y_H = self._vec_to_fields(self.y)

        self.z = cp.zeros(self.n_total, dtype=cp.float32)
        self.z_u, self.z_v, self.z_H = self._vec_to_fields(self.z)

        # Constraint work arrays
        self.chi = cp.zeros((ny, nx), dtype=cp.float32)
        self.phi = cp.zeros((ny, nx), dtype=cp.float32)

        # Physical parameters (cell-centered)
        self.bed = cp.zeros((ny, nx), dtype=cp.float32)
        self.beta = cp.zeros((ny, nx), dtype=cp.float32)
        self.grad_beta = cp.zeros((ny,nx),dtype=cp.float32)
        self.B = cp.zeros((ny, nx), dtype=cp.float32)
        self.smb = cp.zeros((ny, nx), dtype=cp.float32)
        self.mask = cp.zeros((ny, nx), dtype=cp.float32)
        self.error_mask = cp.zeros((ny, nx), dtype=cp.float32)
        self.error_mask.fill(1)
        self.gamma = cp.zeros((ny, nx), dtype=cp.float32)
        self.grounded = cp.zeros((ny,nx),dtype=cp.float32)

    def _vec_to_fields(self, x):
        """Create field views into a monolithic state vector."""
        u = x[:self.nu].reshape(self.ny, self.nx + 1)
        v = x[self.nu:self.nU].reshape(self.ny + 1, self.nx)
        H = x[self.nU:].reshape(self.ny, self.nx)
        return u, v, H

    def spawn_child(self):
        """Create a coarser child grid (2x coarsening)."""
        child = Grid(
            self.ny // 2, self.nx // 2,
            self.dx * 2, self.dt,
            self.kernels,
            parent=self,
            n=self._n,
            eps_reg=self._eps_reg,
            m=self._m,
            u_reg=self._u_reg,
            water_drag=self._water_drag,
            calving_rate=self._calving_rate,
            sigmoid_c=self._sigmoid_c
        )
        self.child = child
        return child

    def _kernel_config(self):
        """Return (grid_size, block_size, stride, halo) for kernels."""
        block_size = (16, 16)
        stride = 14
        halo = 1
        grid_size = (self.nx // stride + 1, self.ny // stride + 1)
        return grid_size, block_size, stride, halo

    def compute_residual(self, use_mask=True, enable_calving=True, recompute_grounded=True):
        """Compute residual r = f - F(U)."""
        kernel = self.kernels.ice.get_function('compute_residual')
        grid_size, block_size, stride, halo = self._kernel_config()

        if use_mask:
            mask = self.mask
        else:
            mask = self.Z_H     

        if enable_calving:
            calving_rate = self._calving_rate
        else:
            calving_rate = cp.float32(0.0)

        if recompute_grounded:
            self.compute_grounded()

        kernel(grid_size, block_size,
               (self.r_u, self.r_v, self.r_H,
                self.u, self.v, self.H, self.grounded,
                self.f_u, self.f_v, self.f_H,
                self.bed, self.B, self.beta,
                mask, self.gamma,
                self._n, self._eps_reg, 
                self._m, self._u_reg,
                self._water_drag,
                calving_rate, self._sigmoid_c,
                self.dx, self.dt,
                self.ny, self.nx, stride, halo))



    def compute_F(self, use_mask=True, enable_calving=True, recompute_grounded=True):
        """Compute F(U) (operator evaluation without RHS)."""
        kernel = self.kernels.ice.get_function('compute_residual')
        grid_size, block_size, stride, halo = self._kernel_config()

        if use_mask:
            mask = self.mask
        else:
            mask = self.Z_H  

        if enable_calving:
            calving_rate = self._calving_rate
        else:
            calving_rate = cp.float32(0.0)

        if recompute_grounded:
            self.compute_grounded()


        kernel(grid_size, block_size,
               (self.F_u, self.F_v, self.F_H,
                self.u, self.v, self.H, self.grounded,
                self.Z_u, self.Z_v, self.Z_H,
                self.bed, self.B, self.beta,
                mask, self.gamma,
                self._n, self._eps_reg, 
                self._m, self._u_reg,
                self._water_drag,
                calving_rate, self._sigmoid_c,
                self.dx, self.dt,
                self.ny, self.nx, stride, halo))

    def compute_jvp(self, use_mask=True, enable_calving=True, recompute_grounded=True):
        """Compute Jacobian-vector product J @ d_U."""
        kernel = self.kernels.ice.get_function('compute_jvp')
        grid_size, block_size, stride, halo = self._kernel_config()

        if use_mask:
            mask = self.mask
        else:
            mask = self.Z_H  


        if enable_calving:
            calving_rate = self._calving_rate
        else:
            calving_rate = cp.float32(0.0)

        if recompute_grounded:
            self.compute_grounded()


        kernel(grid_size, block_size,
               (self.j_u, self.j_v, self.j_H,
                self.u, self.v, self.H, self.grounded,
                self.d_u, self.d_v, self.d_H,
                self.bed, self.B, self.beta,
                mask, self.gamma,
                self._n, self._eps_reg,
                self._m, self._u_reg,
                self._water_drag,
                calving_rate, self._sigmoid_c,
                self.dx, self.dt,
                self.ny, self.nx, stride, halo))

    def compute_residual_adjoint(self,use_mask=True,enable_calving=True):
        self.compute_vjp(use_mask=use_mask,enable_calving=enable_calving)
        self.r_adj[:] = self.f_adj - self.l
        if use_mask:
            self.r_adj_H[self.mask>1e-3] = 0.0

    def compute_F_adjoint(self,use_mask=True,enable_calving=True):
        self.compute_vjp(use_mask=use_mask,enable_calving=enable_calving)
        self.F_adj[:] = -self.l

    def compute_vjp(self,use_mask=True,enable_calving=True,recompute_grounded=False):
        """Compute vector-Jacobian product Lambda^T @ J."""
        kernel = self.kernels.ice.get_function('compute_vjp')
        grid_size, block_size, stride, halo = self._kernel_config()

        if use_mask:
            mask = self.mask
        else:
            mask = self.Z_H  

        if enable_calving:
            calving_rate = self._calving_rate
        else:
            calving_rate = cp.float32(0.0)

        if recompute_grounded:
            self.compute_grounded()

        self.l.fill(0.0)
        kernel(grid_size, block_size,
               (self.l_u, self.l_v, self.l_H,
                self.u, self.v, self.H, self.grounded,
                self.lambda_u, self.lambda_v, self.lambda_H,
                self.bed, self.B, self.beta,
                mask, self.gamma,
                self._n, self._eps_reg,
                self._m, self._u_reg,
                self._water_drag,
                calving_rate, self._sigmoid_c,
                self.dx, self.dt,
                self.ny, self.nx, stride, halo))

    def vanka_smooth(self, n_inner=1, enable_calving=True, recompute_grounded=True, relax_grounded=cp.float32(0.5)):
        """Apply one Vanka smoother pass (red-black)."""
        kernel = self.kernels.ice.get_function('vanka_smooth')
        grid_size, block_size, stride, halo = self._kernel_config()

        if enable_calving:
            calving_rate = self._calving_rate
        else:
            calving_rate = cp.float32(0.0)

        if recompute_grounded:
            self.compute_grounded(relaxation=relax_grounded)

        self.delta_U.fill(0.0)
        kernel(grid_size, block_size,
               (self.delta_u, self.delta_v, self.delta_H, self.mask,
                self.u, self.v, self.H, self.grounded,
                self.f_u, self.f_v, self.f_H,
                self.bed, self.B, self.beta, self.gamma,
                self._n, self._eps_reg, 
                self._m, self._u_reg,
                self._water_drag,
                calving_rate, self._sigmoid_c,
                self.dx, self.dt,
                self.ny, self.nx, stride, halo,
                n_inner))

    def vanka_smooth_adjoint(self, use_mask=True, enable_calving=True, recompute_grounded=False):
        """Apply adjoint Vanka smoother pass."""
        kernel = self.kernels.ice.get_function('vanka_smooth_adjoint')
        grid_size, block_size, stride, halo = self._kernel_config()

        if use_mask:
            mask = self.mask
        else:
            mask = self.Z_H  

        if enable_calving:
            calving_rate = self._calving_rate
        else:
            calving_rate = cp.float32(0.0)

        if recompute_grounded:
            self.compute_grounded()

        self.delta_Lambda.fill(0.0)
        kernel(grid_size, block_size,
               (self.delta_lambda_u, self.delta_lambda_v, self.delta_lambda_H,
                self.u, self.v, self.H, self.grounded,
                mask,
                self.r_adj_u, self.r_adj_v, self.r_adj_H,
                self.bed, self.B, self.beta, self.gamma,
                self._n, self._eps_reg,
                self._m, self._u_reg,
                self._water_drag,
                calving_rate, self._sigmoid_c,
                self.dx, self.dt,
                self.ny, self.nx, stride, halo
                ))

    def vanka_sweep(self, n_iter, verbose=False,n_inner=30, omega=cp.float32(0.5),enable_calving=True,recompute_grounded=True):
        """Perform n_iter red-black Vanka smoothing sweeps."""

        for _ in range(n_iter):
            self.vanka_smooth(n_inner=n_inner,enable_calving=enable_calving,recompute_grounded=recompute_grounded)
            self.U[:] += omega * self.delta_U
        if verbose:
            self.compute_residual(use_mask=True,recompute_grounded=False)
            print(self.dx,cp.linalg.norm(self.r_u),
                      cp.linalg.norm(self.r_v),
                      cp.linalg.norm(self.r_H))

    def vanka_sweep_adjoint(self, n_iter, verbose=True,omega=cp.float32(0.5), use_mask=True, enable_calving=True):
        """Perform n_iter adjoint Vanka smoothing sweeps."""
        for _ in range(n_iter):
            self.compute_residual_adjoint(use_mask=use_mask,enable_calving=enable_calving)
            self.vanka_smooth_adjoint(use_mask=use_mask,enable_calving=enable_calving)
            self.Lambda[:] += omega * self.delta_Lambda
        if verbose:
            print(self.dx,cp.linalg.norm(self.r_adj),cp.linalg.norm(self.r_adj_u),cp.linalg.norm(self.r_adj_v),cp.linalg.norm(self.r_adj_H))

    def compute_grad_beta(self):
        """Compute gradient of objective w.r.t. beta via adjoint."""
        kernel = self.kernels.ice.get_function('compute_grad_beta')
        grid_size, block_size, stride, halo = self._kernel_config()

        self.grad_beta.fill(0)
        kernel(grid_size, block_size,
               (self.grad_beta,
                self.u, self.v, self.H, self.grounded,
                self.lambda_u, self.lambda_v, self.lambda_H,
                self.bed, self.B, self.beta,
                self.mask, self.gamma,
                self._n, self._eps_reg,
                self._m, self._u_reg,
                self._water_drag,
                self._calving_rate, self._sigmoid_c,
                self.dx, self.dt,
                self.ny, self.nx, stride, halo))

        return self.grad_beta

    def compute_grounded(self, relaxation=cp.float32(0.0)):
        kernel = self.kernels.ice.get_function('compute_grounded')
        grid_size, block_size, stride, halo = self._kernel_config()

        kernel(grid_size, block_size,
               (self.grounded,
                self.H, self.bed, 
                self._sigmoid_c,
                relaxation,
                self.ny, self.nx, 
                stride, halo))

    def set_rhs(self):
        self.f_H[:,:] = self.H_prev/self.dt + self.smb

    def vanka_dump(self, enable_calving=True):
        """Apply one Vanka smoother pass (red-black)."""
        kernel = self.kernels.ice.get_function('vanka_dump')
        grid_size, block_size, stride, halo = self._kernel_config()

        if enable_calving:
            calving_rate = self._calving_rate
        else:
            calving_rate = cp.float32(0.0)


        J = cp.zeros((self.nh,25),dtype=cp.float32)
        r = cp.zeros((self.nh,5),dtype=cp.float32)
        kernel(grid_size, block_size,
               (J,r,
                self.u, self.v, self.H, self.grounded,
                self.f_u, self.f_v, self.f_H,
                self.bed, self.B, self.beta, self.gamma,
                self._n, self._eps_reg, 
                self._m, self._u_reg,
                self._water_drag,
                calving_rate, self._sigmoid_c,
                self.dx, self.dt,
                self.ny, self.nx, stride, halo
                ))

        return J,r
