from dataclasses import dataclass, field, fields
import cupy as cp
import xarray as xr
from cupy.typing import NDArray
from .field import Field, SubgridField, Constant, GridEntity
from .operators import ForwardOperators, AdjointOperators

@dataclass
class State:
    u: Field | None = None
    v: Field | None = None
    ud: Field | None = None
    vd: Field | None = None
    H: Field | None = None
    H_prev: Field | None = None
    phi: Field | None = None
    xi: Field | None = None
    mask: Field | None = None
    u_b: Field | None = None          # DIVA basal sliding speed |u_b| (None/zero under SSA & MOLHO)
    u_s: Field | None = None          # DIVA surface speed |u_s| = u_b + tau_b*F1

    def __repr__(self):
        return f'{self.u.compact_string}\n{self.v.compact_string}\n{self.H.compact_string}\n{self.H_prev.compact_string}\n{self.phi.compact_string}\n{self.mask.compact_string}'

@dataclass
class AdjointState:
    lambda_u: Field | None = None
    lambda_v: Field | None = None
    lambda_ud: Field | None = None
    lambda_vd: Field | None = None
    lambda_H: Field | None = None

    def __repr__(self):
        return f'{self.lambda_u.compact_string}\n{self.lambda_v.compact_string}\n{self.lambda_H.compact_string}'

@dataclass
class Geometry:
    bed: SubgridField | None = None
    depth: Field | None = None
    thklim: Constant = field(
        default_factory=lambda: Constant(
            value=cp.float32(0.1),
            name='thklim',
            units='m',
            attrs={'long_name':'minimum thickness'})
        )
    sigmoid_c: Constant = field(
        default_factory=lambda: Constant(
            value=cp.float32(0.1),
            name='sigmoid_c',
            units='m^{-1}',
            attrs={'long_name':("smoothing factor for sigmoidal \
                                  grounding flag used in driving stress. \
                                  Lower values imply a smoother transition \
                                  from grounded to floating physics")})
        )
    sigmoid_k: Constant = field(
        default_factory=lambda: Constant(
            value=cp.float32(3.0),
            name='sigmoid_k',
            units='m^{-1}',
            attrs={'long_name':("offset factor for sigmoidal \
                                  grounding flag used in driving stress. \
                                  larger values imply a more asymmetric \
                                  transition from grounded to floating \
                                  physics")})
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
    H_reg: Constant = field(
        default_factory = lambda: Constant(
            value=cp.float32(0.0),
            name='H_reg',
            units='m',
            attrs={'long_name':("Thickness regularizer for the vertical \
                                 shear terms: 1/H^2 -> 1/(H^2 + H_reg^2) \
                                 in the shear invariant, and consistently \
                                 eta/H -> eta*H/(H^2 + H_reg^2) in the \
                                 shear residual (the variational pair - \
                                 see get_sigma_xz_jac), plus a thin-ice \
                                 reactive spring kappa(H)*ud that pins \
                                 the deformational velocity on columns \
                                 thinner than ~H_reg, where the \
                                 regularized shear resistance would \
                                 otherwise vanish and let bare steep \
                                 terrain flow. Zero recovers the \
                                 unregularized model.")})
        )

    # --- DIVA closure (all None / zero under SSA & MOLHO) ---------------------------
    eta_bar: Field | None = None      # depth-averaged effective viscosity (diagnostic)
    F1: Field | None = None           # first shear moment H*int(zeta/eta)dzeta -> surface speed
    F2: Field | None = None           # second shear moment H*int(zeta^2/eta)dzeta -> beta_eff
    # closure sensitivities carried for the exact 2-pass adjoint (velocity + H) and param gradients:
    deta_deps: Field | None = None    # d(eta_bar)/d(eps_mem^2)
    deta_dU: Field | None = None      # d(eta_bar)/d(Ubar)
    dbe_deps: Field | None = None     # d(beta_eff)/d(eps_mem^2)
    dbe_dU: Field | None = None       # d(beta_eff)/d(Ubar)
    deta_dH: Field | None = None      # d(eta_bar)/d(H)  -- the closure's own thickness dependence
    dbe_dH: Field | None = None       # d(beta_eff)/d(H)
    deta_dbeta: Field | None = None   # d(eta_bar)/d(beta)  -- parameter gradients
    dbe_dbeta: Field | None = None    # d(beta_eff)/d(beta)
    deta_duc: Field | None = None     # d(eta_bar)/d(u_c)   -- zero under Weertman
    dbe_duc: Field | None = None      # d(beta_eff)/d(u_c)
    deta_dm: Field | None = None      # d(eta_bar)/d(m)     -- zero under Coulomb
    dbe_dm: Field | None = None       # d(beta_eff)/d(m)
    dus_deps: Field | None = None     # d(u_s)/d(...)  -- surface-velocity objectives
    dus_dU: Field | None = None
    dus_dbeta: Field | None = None
    dus_duc: Field | None = None
    dus_dm: Field | None = None
    dus_dH: Field | None = None
    eps_reg_shear: Constant = field(
        default_factory = lambda: Constant(
            value=cp.float32(1e-12),
            name='eps_reg_shear',
            units='s^{-2}',
            attrs={'long_name':'''DIVA-only strain-invariant regularizer for the SHEAR MOMENTS
                         F1 and F2.  Much smaller than eps_reg because those integrals converge
                         as it goes to zero, while eta_bar (which keeps eps_reg, applied as a
                         cap) does not.'''})
        )
    n_sigma: Constant = field(
        default_factory = lambda: Constant(
            value=cp.float32(8.0),
            name='n_sigma',
            units='',
            attrs={'long_name':'DIVA number of vertical sigma levels (Gauss-Legendre quadrature)'})
        )

    def __repr__(self):
        return f'{self.B.compact_string}\n{self.n}\n{self.eps_reg}\n{self.H_reg}'
    
@dataclass
class Sliding:
    beta: Field | None = None
    u_c: Field | None = None          # DIVA regularized-Coulomb rate-transition speed (unused by Weertman)
    beta_eff: Field | None = None     # DIVA secant drag c/(1 + c*F2); grounding already folded in
    sliding_law: Constant = field(
        default_factory = lambda: Constant(
            value=cp.float32(0.0),
            name='sliding_law',
            units='',
            attrs={'long_name':'0 = Weertman power law, 1 = regularized Coulomb (DIVA only)'})
        )
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
            x0: cp.float32=cp.float32(0.0),
            y0: cp.float32=cp.float32(0.0),
            crs=None,
            parent = None,
            state: State = None,
            adjoint: AdjointState = None,
            geometry: Geometry = None,
            rheology: Rheology = None,
            sliding: Sliding = None,
            calving: Calving = None,
            forcing: Forcing = None,
            stress_scheme: str = 'molho',
            ):

        self.parent = parent
        self.child = None

        # 'molho' solves the two-field shear-resolving model; 'ssa' pins the
        # deformational components (ud, vd) to zero everywhere via identity
        # rows, which reduces the momentum balance exactly to the SSA. The
        # state always carries ud/vd (identically zero under SSA) so all
        # downstream code is scheme-agnostic.
        #
        # 'diva' (depth-integrated viscosity approximation) shares the SSA
        # 2-field (u,v,H) momentum system and 5-DOF Vanka patch -- so it sets
        # ssa=True to select the SSA-shaped build -- but swaps in two cell-local
        # closure coefficients: the depth-averaged viscosity eta_bar and the
        # effective basal drag beta_eff (see cuda/diva.cu). The GLIDE_DIVA
        # compile flag, distinct from ssa, selects those coefficient reads.
        if stress_scheme not in ('molho', 'ssa', 'diva'):
            raise ValueError("stress_scheme must be 'molho', 'ssa', or 'diva'")
        self.stress_scheme = stress_scheme
        self.ssa = stress_scheme in ('ssa', 'diva')
        self.diva = stress_scheme == 'diva'
        
        self.ny = ny
        self.nx = nx
        
        self.dx = cp.float32(dx)

        self.x0 = cp.float32(x0)
        self.y0 = cp.float32(y0)
        self.crs = crs

        self.x_cell = cp.arange(x0,x0 + dx*nx, dx)
        self.y_cell = cp.arange(y0,y0 - dx*ny,-dx)

        self.x_vfacet = cp.arange(x0 - dx/2, x0 - dx/2 + dx*(nx+1), dx) 
        self.y_hfacet = cp.arange(y0 + dx/2, y0 + dx/2 - dx*(ny+1),-dx) 

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
    def adjoint_operators(self):
        if self._adjoint_operators is None:
            self._adjoint_operators = AdjointOperators(self)
        return self._adjoint_operators

    @property
    def adjoint(self):
        if self._adjoint is None:
            self._adjoint = self._allocate_adjoint_state()
        return self._adjoint

    def _allocate_state(self):
        u = Field(
            data=cp.zeros((self.ny, self.nx+1),dtype=cp.float32),
            grid_entity=GridEntity.VERTICAL_FACET,
            dx=self.dx,
            grid=self,
            name='u',
            units='m a^{-1}',
            attrs={'long_name':'x component of depth-averaged velocity'})
        
        v = Field(
            data=cp.zeros((self.ny+1, self.nx),dtype=cp.float32),
            grid_entity=GridEntity.HORIZONTAL_FACET,
            dx=self.dx,
            grid=self,
            name='v',
            units='m a^{-1}',
            attrs={'long_name':'y component of depth-averaged velocity'})

        ud = Field(
            data=cp.zeros((self.ny, self.nx+1),dtype=cp.float32),
            grid_entity=GridEntity.VERTICAL_FACET,
            dx=self.dx,
            grid=self,
            name='ud',
            units='m a^{-1}',
            attrs={'long_name':'x component of deformation velocity'})
        
        vd = Field(
            data=cp.zeros((self.ny+1, self.nx),dtype=cp.float32),
            grid_entity=GridEntity.HORIZONTAL_FACET,
            dx=self.dx,
            grid=self,
            name='vd',
            units='m a^{-1}',
            attrs={'long_name':'y component of deformation velocity'})



        H = Field(
            data=cp.zeros((self.ny,self.nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=self.dx,
            grid=self,
            name='H',
            units='m',
            attrs={'long_name':'Ice thickness at t + dt (end of time step)'})
        
        H_prev = Field(
            data=cp.zeros((self.ny,self.nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=self.dx,
            grid=self,
            name='H_prev',
            units='m',
            attrs={'long_name':'Ice thickness at t (beginning of time step)'})

        phi = Field(
            data=cp.zeros((self.ny,self.nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=self.dx,
            grid=self,
            name='phi',
            units='m',
            attrs={'long_name':'Potential head'})

        xi = Field(
            data=cp.zeros((self.ny,self.nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=self.dx,
            grid=self,
            name='phi',
            units='m',
            attrs={'long_name':'Flotation fraction'})


        mask = Field(
            data=cp.zeros((self.ny,self.nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=self.dx,
            grid=self,
            name='mask',
            units='',
            attrs={'long_name':'''Active set mask - if unity, thickness is 
                         set to thklim in Dirichlet BC fashion'''})

        if not self.diva:
            return State(u=u,v=v,ud=ud,vd=vd,H=H,H_prev=H_prev,phi=phi,xi=xi,mask=mask)

        # DIVA carries the per-cell basal and surface speeds diagnosed by the closure.
        def _cell(name, long_name):
            return Field(data=cp.zeros((self.ny,self.nx),dtype=cp.float32),
                         grid_entity=GridEntity.CELL, dx=self.dx, grid=self,
                         name=name, units='m a^{-1}', attrs={'long_name':long_name})
        u_b = _cell('u_b','DIVA basal sliding speed |u_b|')
        u_s = _cell('u_s','DIVA surface speed |u_s| = u_b + tau_b*F1')
        return State(u=u,v=v,ud=ud,vd=vd,H=H,H_prev=H_prev,phi=phi,xi=xi,mask=mask,u_b=u_b,u_s=u_s)

    def _allocate_adjoint_state(self):
        lambda_u = Field(
            data=cp.zeros((self.ny, self.nx+1),dtype=cp.float32),
            grid_entity=GridEntity.VERTICAL_FACET,
            dx=self.dx,
            grid=self,
            name='lambda_u',
            units='varies with objective fn',
            attrs={'long_name':'Adjoint variable for u'})

        lambda_v = Field(
            data=cp.zeros((self.ny+1, self.nx),dtype=cp.float32),
            grid_entity=GridEntity.HORIZONTAL_FACET,
            dx=self.dx,
            grid=self,
            name='lambda_v',
            units='varies with objective fn',
            attrs={'long_name':'Adjoint variable for v'})

        lambda_ud = Field(
            data=cp.zeros((self.ny, self.nx+1),dtype=cp.float32),
            grid_entity=GridEntity.VERTICAL_FACET,
            dx=self.dx,
            grid=self,
            name='lambda_ud',
            units='varies with objective fn',
            attrs={'long_name':'Adjoint variable for ud'})

        lambda_vd = Field(
            data=cp.zeros((self.ny+1, self.nx),dtype=cp.float32),
            grid_entity=GridEntity.HORIZONTAL_FACET,
            dx=self.dx,
            grid=self,
            name='lambda_vd',
            units='varies with objective fn',
            attrs={'long_name':'Adjoint variable for vd'})


        lambda_H = Field(
            cp.zeros((self.ny,self.nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=self.dx,
            grid=self,
            name='lambda_H',
            units='varies with objective fn',
            attrs={'long_name':'Adjoint variable for H'})

        return AdjointState(lambda_u=lambda_u,lambda_v=lambda_v,lambda_ud=lambda_ud,lambda_vd=lambda_vd,lambda_H=lambda_H)

    def _allocate_geometry(self):
        bed = SubgridField(
            data=cp.zeros((self.ny,self.nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=self.dx,
            grid=self,
            name='bed',
            units='m',
            attrs={'long_name':'bed elevation (not necessarily the ice base)'})

        depth = SubgridField(
            data=cp.zeros((self.ny,self.nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=self.dx,
            grid=self,
            name='depth',
            units='m',
            attrs={'long_name':'water depth'})
        return Geometry(bed=bed,depth=depth)

    def _allocate_rheology(self):
        B = Field(
            data=cp.zeros((self.ny,self.nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=self.dx,
            grid=self,
            name='B',
            units='m',
            attrs={'long_name':'Rheologic prefactor.  B=A^{-1/n}'})

        if not self.diva:
            return Rheology(B=B)

        # DIVA closure diagnostics + adjoint sensitivities, all cell-centred.
        def _cell(name, units, long_name):
            return Field(data=cp.zeros((self.ny,self.nx),dtype=cp.float32),
                         grid_entity=GridEntity.CELL, dx=self.dx, grid=self,
                         name=name, units=units, attrs={'long_name':long_name})

        return Rheology(B=B,
            eta_bar=_cell('eta_bar','Pa a','DIVA depth-averaged effective viscosity'),
            F1=_cell('F1','a','DIVA first shear moment H*int(zeta/eta)dzeta'),
            F2=_cell('F2','a Pa^{-1}','DIVA second shear moment H*int(zeta^2/eta)dzeta'),
            deta_deps=_cell('deta_deps','Pa a^3','DIVA d(eta_bar)/d(eps_mem^2)'),
            deta_dU=_cell('deta_dU','Pa a^2 m^{-1}','DIVA d(eta_bar)/d(Ubar)'),
            dbe_deps=_cell('dbe_deps','?','DIVA d(beta_eff)/d(eps_mem^2)'),
            dbe_dU=_cell('dbe_dU','?','DIVA d(beta_eff)/d(Ubar)'),
            deta_dH=_cell('deta_dH','Pa a m^{-1}','DIVA d(eta_bar)/d(H)'),
            dbe_dH=_cell('dbe_dH','?','DIVA d(beta_eff)/d(H)'),
            deta_dbeta=_cell('deta_dbeta','?','DIVA d(eta_bar)/d(beta)'),
            dbe_dbeta=_cell('dbe_dbeta','?','DIVA d(beta_eff)/d(beta)'),
            deta_duc=_cell('deta_duc','?','DIVA d(eta_bar)/d(u_c)'),
            dbe_duc=_cell('dbe_duc','?','DIVA d(beta_eff)/d(u_c)'),
            deta_dm=_cell('deta_dm','?','DIVA d(eta_bar)/d(m)'),
            dbe_dm=_cell('dbe_dm','?','DIVA d(beta_eff)/d(m)'),
            dus_deps=_cell('dus_deps','?','DIVA d(u_s)/d(eps_mem^2)'),
            dus_dU=_cell('dus_dU','?','DIVA d(u_s)/d(Ubar)'),
            dus_dbeta=_cell('dus_dbeta','?','DIVA d(u_s)/d(beta)'),
            dus_duc=_cell('dus_duc','?','DIVA d(u_s)/d(u_c)'),
            dus_dm=_cell('dus_dm','?','DIVA d(u_s)/d(m)'),
            dus_dH=_cell('dus_dH','a^{-1}','DIVA d(u_s)/d(H)'))

    def _allocate_sliding(self):
        beta = Field(
            data=cp.zeros((self.ny,self.nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=self.dx,
            grid=self,
            name='beta',
            units='?',
            attrs={'long_name':'Basal sliding coefficient'})

        if not self.diva:
            return Sliding(beta=beta)

        def _cell(name, units, long_name):
            return Field(data=cp.zeros((self.ny,self.nx),dtype=cp.float32),
                         grid_entity=GridEntity.CELL, dx=self.dx, grid=self,
                         name=name, units=units, attrs={'long_name':long_name})

        return Sliding(beta=beta,
            u_c=_cell('u_c','m a^{-1}','DIVA regularized-Coulomb rate-transition speed'),
            beta_eff=_cell('beta_eff','?','DIVA effective secant drag c/(1+c*F2), grounding folded in'))

    def _allocate_calving(self):
        return Calving()
    
    def _allocate_forcing(self):
        smb = Field(
            data=cp.zeros((self.ny,self.nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=self.dx,
            grid=self,
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

    def to_dataset(self,
            fields: dict[str, Field] | None = None,
            *,
            attrs: dict | None = None):
    
        data_vars = {name: fld.to_dataarray() for name,fld in fields.items()}
        ds = xr.Dataset(data_vars=data_vars)
        ds.attrs['dx'] = self.dx
        ds.attrs['crs'] = str(self.crs)
        ds.attrs["spatial_ref"] = self.crs.to_wkt()
        ds.attrs["crs_wkt"] = self.crs.to_wkt()        

        if attrs:
            ds.attrs.update(attrs)

        return ds
        
  
