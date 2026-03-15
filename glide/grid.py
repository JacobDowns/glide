from dataclasses import dataclass, field, fields
import cupy as cp
from cupy.typing import NDArray
from .field import Field, Constant
from .operators import ForwardOperators

@dataclass
class State:
    u: Field | None = None
    v: Field | None = None
    H: Field | None = None
    H_prev: Field | None = None
    phi: Field | None = None
    mask: Field | None = None

    def __repr__(self):
        return f'{self.u.compact_string}\n{self.v.compact_string}\n{self.H.compact_string}\n{self.H_prev.compact_string}\n{self.phi.compact_string}\n{self.mask.compact_string}'

@dataclass
class AdjointState:
    lambda_u: Field | None = None
    lambda_v: Field | None = None
    lambda_H: Field | None = None

    def __repr__(self):
        return f'{self.lambda_u.compact_string}\n{self.lambda_v.compact_string}\n{self.lambda_H.compact_string}'

@dataclass
class Geometry:
    bed: Field | None = None
    thklim: Constant = field(
        default_factory=lambda: Constant(
            value=cp.float32(0.1),
            name='thklim',
            units='m',
            attrs={'long_name':'minimum thickness'})
        )
    flotation_reg_driving: Constant = field(
        default_factory=lambda: Constant(
            value=cp.float32(0.1),
            name='flotation_reg_driving',
            units='m^{-1}',
            attrs={'long_name':("smoothing factor for sigmoidal \
                                  grounding flag used in driving stress. \
                                  Lower values imply a smoother transition \
                                  from grounded to floating physics")})
        )

    def __repr__(self):
        return f'{self.bed.compact_string}\n{self.thklim}\n{self.flotation_reg_driving}'

@dataclass
class Rheology:
    B: Field | None = None
    n: Constant = field(
        default_factory = lambda: Constant(
            value=cp.float32(3.0),
            name='n',
            units='',
            attrs={'long_name':'Glens law n'})
        )
    eps_reg: Constant = field(
        default_factory = lambda: Constant(
            value=cp.float32(1e-6),
            name='eps_reg',
            units='s^{-2}',
            attrs={'long_name':'Strain invariant squared regularizer'})
        )
    
    def __repr__(self):
        return f'{self.B.compact_string}\n{self.n}\n{self.eps_reg}'
    
@dataclass
class Sliding:
    beta: Field | None = None
    m: Constant = field(
        default_factory = lambda: Constant(
            value=cp.float32(1.0),
            name='m',
            units='',
            attrs={'long_name':'Weertman law m'})
        )
    u_reg: Constant = field(
        default_factory = lambda: Constant(
            value=cp.float32(1.0),
            name='u_reg',
            units='m a^{-1}',
            attrs={'long_name':'Weertman law regularization'})
        )
    water_drag: Constant = field(
        default_factory = lambda: Constant(
            value=cp.float32(1e-5),
            name='water_drag',
            units='',
            attrs={'long_name':'basal traction exerted by water'})
        )

    flotation_reg_sliding: Constant = field(
        default_factory=lambda: Constant(
            value=cp.float32(0.1),
            name='flotation_reg_sliding',
            units='m',
            attrs={'long_name':("smoothing factor for pseudo-sigmoidal \
                                  grounding flag used in basal stress. \
                                  Larger values imply a smoother transition \
                                  from grounded to floating physics")})
        )


    def __repr__(self):
        return f'{self.beta.compact_string}\n{self.m}\n{self.u_reg}\n{self.water_drag}\n{self.flotation_reg_sliding}'

@dataclass
class Calving:
    calving_rate: Constant = field(
        default_factory = lambda: Constant(
            value=cp.float32(0.0),
            name='calving_rate',
            units='m a^{-1}',
            attrs={'long_name':"The speed at which ice \
                        nonconservatively fluxes through \
                        a facet when both cells are floating"})
        )

    flotation_reg_calving: Constant = field(
        default_factory=lambda: Constant(
            value=cp.float32(0.1),
            name='flotation_reg_calving',
            units='m',
            attrs={'long_name':("smoothing factor for pseudo-sigmoidal \
                                  grounding flag used in calving. \
                                  Larger values imply a smoother transition \
                                  from grounded to floating physics")})
        )

    def __repr__(self):
        return f'{self.calving_rate}\n{self.flotation_reg_calving}'

@dataclass
class Forcing:
    smb: Field = None
    
    def __repr__(self):
        return f'{self.smb.compact_string}'

class Grid:
    """
    Single level of the multigrid hierarchy.

    Parameters
    ----------
    ny, nx : int
        Grid dimensions (number of cells in y and x)
    dx : float
        Grid spacing (assumed isotropic)
    parent : Grid, optional
        Parent (finer) grid in hierarchy

    Note: If any of the optional dataclasses are passed in,
    the Fields and Constants contained therein are *not*
    copied and will mutate if the original data is 
    mutated externally.  This may or may not be 
    desirable behavior.
    """

    def __init__(self, ny: int, nx: int, dx: cp.float32, 
            parent = None,
            state: State = None,
            adjoint: AdjointState = None,
            geometry: Geometry = None,
            rheology: Rheology = None,
            sliding: Sliding = None,
            calving: Calving = None,
            forcing: Forcing = None
            ):

        self.parent = parent
        self.child = None
        
        self.ny = ny
        self.nx = nx
        
        self.dx = cp.float32(dx)

        # Degrees of freedom
        self.nu = ny * (nx + 1)
        self.nv = (ny + 1) * nx
        self.nh = ny * nx
        self.n_total = self.nu + self.nv + self.nh

        self.state    = state    if state    is not None else self._allocate_state()
        self.geometry = geometry if geometry is not None else self._allocate_geometry()
        self.rheology = rheology if rheology is not None else self._allocate_rheology()
        self.sliding  = sliding  if sliding  is not None else self._allocate_sliding()
        self.calving  = calving  if calving  is not None else self._allocate_calving()
        self.forcing  = forcing  if forcing  is not None else self._allocate_forcing()
        
        # Adjoint fields are initialized lazily
        self._adjoint  = adjoint

        self._forward_operators = None
        self._adjoint_operators = None

    @property
    def forward_operators(self):
        if self._forward_operators is None:
            self._forward_operators = ForwardOperators(self)
        return self._forward_operators

    @property
    def adjoint(self):
        if self._adjoint is None:
            self._adjoint = self._allocate_adjoint_state()
        return self._adjoint

    def _allocate_state(self):
        u = Field(
            data=cp.zeros((self.ny, self.nx+1),dtype=cp.float32),
            name='u',
            units='m a^{-1}',
            attrs={'long_name':'x component of depth-averaged velocity'})
        
        v = Field(
            data=cp.zeros((self.ny+1, self.nx),dtype=cp.float32),
            name='v',
            units='m a^{-1}',
            attrs={'long_name':'y component of depth-averaged velocity'})

        H = Field(
            data=cp.zeros((self.ny,self.nx),dtype=cp.float32),
            name='H',
            units='m',
            attrs={'long_name':'Ice thickness at t + dt (end of time step)'})
        
        H_prev = Field(
            data=cp.zeros((self.ny,self.nx),dtype=cp.float32),
            name='H_prev',
            units='m',
            attrs={'long_name':'Ice thickness at t (beginning of time step)'})

        phi = Field(
            data=cp.zeros((self.ny,self.nx),dtype=cp.float32),
            name='phi',
            units='m',
            attrs={'long_name':'Potential Head'})

        mask = Field(
            data=cp.zeros((self.ny,self.nx),dtype=cp.float32),
            name='mask',
            units='',
            attrs={'long_name':'''Active set mask - if unity, thickness is 
                         set to thklim in Dirichlet BC fashion'''})

        return State(u=u,v=v,H=H,H_prev=H_prev,phi=phi,mask=mask)

    def _allocate_adjoint_state(self):
        lambda_u = Field(
            data=cp.zeros((self.ny, self.nx+1),dtype=cp.float32),
            name='lambda_u',
            units='varies with objective fn',
            attrs={'long_name':'Adjoint variable for u'})

        lambda_v = Field(
            data=cp.zeros((self.ny+1, self.nx),dtype=cp.float32),
            name='lambda_v',
            units='varies with objective fn',
            attrs={'long_name':'Adjoint variable for v'})

        lambda_H = Field(cp.zeros((self.ny,self.nx),dtype=cp.float32),
            name='lambda_H',
            units='varies with objective fn',
            attrs={'long_name':'Adjoint variable for H'})

        return AdjointState(lambda_u=lambda_u,lambda_v=lambda_v,lambda_H=lambda_H)

    def _allocate_geometry(self):
        bed = Field(
            data=cp.zeros((self.ny,self.nx),dtype=cp.float32),
            name='bed',
            units='m',
            attrs={'long_name':'bed elevation (not necessarily the ice base)'})
        return Geometry(bed=bed)

    def _allocate_rheology(self):
        B = Field(
            data=cp.zeros((self.ny,self.nx),dtype=cp.float32),
            name='B',
            units='m',
            attrs={'long_name':'Rheologic prefactor.  B=A^{-1/n}'})

        return Rheology(B=B)

    def _allocate_sliding(self):
        beta = Field(
            data=cp.zeros((self.ny,self.nx),dtype=cp.float32),
            name='beta',
            units='?',
            attrs={'long_name':'Basal sliding coefficient'})

        return Sliding(beta=beta)

    def _allocate_calving(self):
        return Calving()
    
    def _allocate_forcing(self):
        smb = Field(
            data=cp.zeros((self.ny,self.nx),dtype=cp.float32),
            name='smb',
            units='m a^{-1}',
            attrs={'long_name':'Surface mass balance'})

        return Forcing(smb=smb)

    def spawn_child(self):
        child = Grid(
            self.ny // 2, self.nx // 2,
            self.dx * 2, parent=self
        )
        self.child = child
        return child

"""
    def _allocate_arrays(self):
        ny, nx = self.ny, self.nx

        # Previous thickness (for time stepping)
        ## self.H_prev = cp.zeros((ny, nx), dtype=cp.float32)

        # Primary state vector [u, v, H]
        ## self.U = cp.zeros(self.n_total, dtype=cp.float32)
        ## self.u, self.v, self.H = self._vec_to_fields(self.U)

        # Perturbation vector (for JVP)
        self.d_U = cp.zeros(self.n_total, dtype=cp.float32)
        self.d_u, self.d_v, self.d_H = self._vec_to_fields(self.d_U)

        # Update/correction vector
        self.delta_U = cp.zeros(self.n_total, dtype=cp.float32)
        self.delta_u, self.delta_v, self.delta_H = self._vec_to_fields(self.delta_U)

        # Adjoint state vector
        ## self.Lambda = cp.zeros(self.n_total, dtype=cp.float32)
        ## self.lambda_u, self.lambda_v, self.lambda_H = self._vec_to_fields(self.Lambda)
        
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
        ## self.bed = cp.zeros((ny, nx), dtype=cp.float32)
        ## self.beta = cp.zeros((ny, nx), dtype=cp.float32)
        self.grad_beta = cp.zeros((ny,nx),dtype=cp.float32)
        ## self.B = cp.zeros((ny, nx), dtype=cp.float32)
        ## self.smb = cp.zeros((ny, nx), dtype=cp.float32)
        ## self.mask = cp.zeros((ny, nx), dtype=cp.float32)
        self.error_mask = cp.zeros((ny, nx), dtype=cp.float32)
        self.error_mask.fill(1)
        self.gamma = cp.zeros((ny, nx), dtype=cp.float32)
        ## self.grounded = cp.zeros((ny,nx),dtype=cp.float32)


    def _kernel_config(self):
        block_size = (16, 16)
        stride = 14
        halo = 1
        grid_size = (self.nx // stride + 1, self.ny // stride + 1)
        return grid_size, block_size, stride, halo

    def compute_residual(self, use_mask=True, enable_calving=True, recompute_grounded=True):
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

        for _ in range(n_iter):
            self.vanka_smooth(n_inner=n_inner,enable_calving=enable_calving,recompute_grounded=recompute_grounded)
            self.U[:] += omega * self.delta_U
        if verbose:
            self.compute_residual(use_mask=True,recompute_grounded=False)
            print(self.dx,cp.linalg.norm(self.r_u),
                      cp.linalg.norm(self.r_v),
                      cp.linalg.norm(self.r_H))

    def vanka_sweep_adjoint(self, n_iter, verbose=True,omega=cp.float32(0.5), use_mask=True, enable_calving=True):
        for _ in range(n_iter):
            self.compute_residual_adjoint(use_mask=use_mask,enable_calving=enable_calving)
            self.vanka_smooth_adjoint(use_mask=use_mask,enable_calving=enable_calving)
            self.Lambda[:] += omega * self.delta_Lambda
        if verbose:
            print(self.dx,cp.linalg.norm(self.r_adj),cp.linalg.norm(self.r_adj_u),cp.linalg.norm(self.r_adj_v),cp.linalg.norm(self.r_adj_H))

    def compute_grad_beta(self):
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
"""
