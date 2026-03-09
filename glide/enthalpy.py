"""
Fixed-geometry enthalpy diffusion solver.

This first implementation solves only the vertical diffusion term in
terrain-following coordinates, one independent ice column at a time.
The thermodynamics are cold-only for now: enthalpy is treated as a
temperature-like scalar with prescribed surface values and basal heat flux.
"""

import cupy as cp

from .kernels import get_kernels


SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0


class EnthalpyGrid:
    """State container for the fixed-geometry column solver."""

    def __init__(self, ny, nx, nz, dx, thklim=0.1):
        self.ny = ny
        self.nx = nx
        self.nz = nz
        self.n_columns = ny * nx
        self.dx = cp.float32(dx)
        self.dsigma = cp.float32(1.0 / nz)
        self.dt = cp.float32(0.0)
        self.thklim = cp.float32(thklim)

        self.bed = cp.zeros((ny, nx), dtype=cp.float32)
        self.H = cp.zeros((ny, nx), dtype=cp.float32)

        self.enthalpy = cp.zeros((ny, nx, nz), dtype=cp.float32)
        self.enthalpy_prev = cp.zeros_like(self.enthalpy)

        self.surface_enthalpy = cp.zeros((ny, nx), dtype=cp.float32)
        self.geothermal_flux = cp.zeros((ny, nx), dtype=cp.float32)

        self.diffusivity = cp.zeros((ny, nx), dtype=cp.float32)
        self.conductivity = cp.zeros((ny, nx), dtype=cp.float32)

        # Thomas scratch arrays, stored with the same layout as enthalpy.
        self.c_prime = cp.zeros_like(self.enthalpy)
        self.d_prime = cp.zeros_like(self.enthalpy)

    def sigma_centers(self):
        """Return terrain-following cell-center coordinates."""
        return (cp.arange(self.nz, dtype=cp.float32) + 0.5) * self.dsigma


class EnthalpyPhysics:
    """
    GPU-accelerated column-wise enthalpy diffusion solver.

    The model keeps geometry fixed and advances only vertical diffusion
    using backward Euler in terrain-following coordinates.
    """

    def __init__(self, ny, nx, nz, dx, thklim=0.1):
        self.ny = ny
        self.nx = nx
        self.nz = nz
        self.dx = dx
        self.thklim = cp.float32(thklim)
        self.kernels = get_kernels()
        self.grid = EnthalpyGrid(ny, nx, nz, dx, thklim=thklim)
        self.grid.diffusivity.fill(float(self.thermal_diffusivity_from_si(1.09e-6)))
        self.grid.conductivity.fill(2.1)

    def _assign_cell_field(self, target, value):
        arr = cp.asarray(value, dtype=cp.float32)
        if arr.ndim == 0:
            target.fill(float(arr))
        else:
            target[:] = arr

    def set_geometry(self, bed, thickness):
        """Set fixed bed and thickness geometry."""
        self.grid.bed[:] = cp.asarray(bed, dtype=cp.float32)
        self.grid.H[:] = cp.asarray(thickness, dtype=cp.float32)

    def set_boundary_conditions(self, surface_enthalpy=None, geothermal_flux=None):
        """Set upper Dirichlet and lower geothermal flux boundary conditions."""
        if surface_enthalpy is not None:
            self._assign_cell_field(self.grid.surface_enthalpy, surface_enthalpy)
        if geothermal_flux is not None:
            self._assign_cell_field(self.grid.geothermal_flux, geothermal_flux)

    def set_parameters(self, diffusivity=None, conductivity=None):
        """Set cold-ice transport properties."""
        if diffusivity is not None:
            self._assign_cell_field(self.grid.diffusivity, diffusivity)
        if conductivity is not None:
            self._assign_cell_field(self.grid.conductivity, conductivity)

    def set_initial_enthalpy(self, enthalpy):
        """Set the full 3D initial enthalpy state."""
        arr = cp.asarray(enthalpy, dtype=cp.float32)
        if arr.ndim == 0:
            self.grid.enthalpy.fill(float(arr))
        else:
            self.grid.enthalpy[:] = arr
        self.grid.enthalpy_prev[:] = self.grid.enthalpy

    def initialize_linear_profile(self):
        """
        Initialize with a column-wise conductive profile.

        This uses the current surface enthalpy, conductivity, geothermal
        flux, and ice thickness fields to build a linear profile in
        physical height above the bed. It is a useful cold-only initial
        state for diffusion tests.
        """
        sigma = self.grid.sigma_centers()[cp.newaxis, cp.newaxis, :]
        height = sigma * self.grid.H[:, :, cp.newaxis]
        basal_enthalpy = (
            self.grid.surface_enthalpy
            + self.grid.geothermal_flux * self.grid.H / self.grid.conductivity
        )
        profile = (
            basal_enthalpy[:, :, cp.newaxis]
            - (self.grid.geothermal_flux / self.grid.conductivity)[:, :, cp.newaxis] * height
        )
        thin_mask = self.grid.H <= self.grid.thklim
        self.grid.enthalpy[:] = cp.where(
            thin_mask[:, :, cp.newaxis],
            self.grid.surface_enthalpy[:, :, cp.newaxis],
            profile,
        )
        self.grid.enthalpy_prev[:] = self.grid.enthalpy

    def forward(self, dt):
        """Advance one implicit diffusion step."""
        self.grid.dt = cp.float32(dt)
        self.grid.enthalpy_prev[:] = self.grid.enthalpy

        kernel = self.kernels.enthalpy.get_function("enthalpy_diffusion_step")
        block_size = 256
        grid_size = (self.grid.n_columns + block_size - 1) // block_size
        kernel(
            (grid_size,),
            (block_size,),
            (
                self.grid.enthalpy,
                self.grid.enthalpy_prev,
                self.grid.H,
                self.grid.surface_enthalpy,
                self.grid.geothermal_flux,
                self.grid.conductivity,
                self.grid.diffusivity,
                self.grid.dt,
                self.grid.dsigma,
                self.grid.thklim,
                self.grid.n_columns,
                self.grid.nz,
                self.grid.c_prime,
                self.grid.d_prime,
            ),
        )

        return self.grid.enthalpy

    def get_basal_enthalpy(self):
        """Return basal enthalpy."""
        return self.grid.enthalpy[:, :, 0]

    def get_surface_layer_enthalpy(self):
        """Return the uppermost cell-center enthalpy."""
        return self.grid.enthalpy[:, :, -1]

    def get_mean_enthalpy(self):
        """Return the vertical mean enthalpy."""
        return self.grid.enthalpy.mean(axis=2)

    @staticmethod
    def thermal_diffusivity_from_si(value_m2_s):
        """Convert thermal diffusivity from m^2/s to m^2/yr."""
        return cp.float32(value_m2_s * SECONDS_PER_YEAR)
