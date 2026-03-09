"""
Fixed-geometry enthalpy advection-diffusion solver.

This first implementation solves only vertical advection-diffusion in
terrain-following coordinates, one independent ice column at a time.
The prognostic variable is temperature-equivalent enthalpy: values below
the pressure-melting threshold behave like cold ice temperature, while
positive values represent latent heat content in temperate ice.
"""

import cupy as cp

from .kernels import get_kernels


SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0
LATENT_HEAT = 3.34e5           # J / kg
HEAT_CAPACITY = 2009.0         # J / kg / K
LATENT_HEAT_EQUIVALENT = LATENT_HEAT / HEAT_CAPACITY


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
        self.sigma_velocity = cp.zeros((ny, nx), dtype=cp.float32)

        self.diffusivity = cp.zeros((ny, nx), dtype=cp.float32)
        self.conductivity = cp.zeros((ny, nx), dtype=cp.float32)
        self.diffusivity_eff = cp.zeros_like(self.enthalpy)

        # Thomas scratch arrays, stored with the same layout as enthalpy.
        self.c_prime = cp.zeros_like(self.enthalpy)
        self.d_prime = cp.zeros_like(self.enthalpy)

    def sigma_centers(self):
        """Return terrain-following cell-center coordinates."""
        return (cp.arange(self.nz, dtype=cp.float32) + 0.5) * self.dsigma


class EnthalpyPhysics:
    """
    GPU-accelerated column-wise enthalpy advection-diffusion solver.

    The model keeps geometry fixed and advances only vertical transport
    using backward Euler in terrain-following coordinates.
    """

    def __init__(self, ny, nx, nz, dx, thklim=0.1):
        self.ny = ny
        self.nx = nx
        self.nz = nz
        self.dx = dx
        self.thklim = cp.float32(thklim)
        self.enthalpy_pmp = cp.float32(0.0)
        self.transition_width = cp.float32(0.05)
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

    def set_parameters(self, diffusivity=None, conductivity=None, sigma_velocity=None):
        """Set cold-ice transport properties and prescribed sigma velocity."""
        if diffusivity is not None:
            self._assign_cell_field(self.grid.diffusivity, diffusivity)
        if conductivity is not None:
            self._assign_cell_field(self.grid.conductivity, conductivity)
        if sigma_velocity is not None:
            self._assign_cell_field(self.grid.sigma_velocity, sigma_velocity)

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

    def _cold_fraction(self, enthalpy):
        """Return a smoothed cold-ice indicator from 1 (cold) to 0 (temperate)."""
        if float(self.transition_width) <= 0.0:
            return (enthalpy < self.enthalpy_pmp).astype(cp.float32)
        return 0.5 * (1.0 - cp.tanh((enthalpy - self.enthalpy_pmp) / self.transition_width))

    def _update_effective_diffusivity(self, enthalpy):
        """Update phase-dependent diffusivity for the current Picard iterate."""
        cold_fraction = self._cold_fraction(enthalpy)
        thin_mask = self.grid.H <= self.grid.thklim
        self.grid.diffusivity_eff[:] = cp.where(
            thin_mask[:, :, cp.newaxis],
            0.0,
            self.grid.diffusivity[:, :, cp.newaxis] * cold_fraction,
        )

    def forward(self, dt, n_picard=25, rtol=1e-5):
        """Advance one implicit polythermal enthalpy step with Picard iteration."""
        self.grid.dt = cp.float32(dt)
        self.grid.enthalpy_prev[:] = self.grid.enthalpy
        enthalpy_old = cp.array(self.grid.enthalpy)

        kernel = self.kernels.enthalpy.get_function("enthalpy_advection_diffusion_step")
        block_size = 256
        grid_size = (self.grid.n_columns + block_size - 1) // block_size
        for _ in range(n_picard):
            self._update_effective_diffusivity(enthalpy_old)
            kernel(
                (grid_size,),
                (block_size,),
                (
                    self.grid.enthalpy,
                    self.grid.enthalpy_prev,
                    self.grid.H,
                    self.grid.surface_enthalpy,
                    self.grid.geothermal_flux,
                    self.grid.sigma_velocity,
                    self.grid.conductivity,
                    self.grid.diffusivity,
                    self.grid.diffusivity_eff,
                    self.grid.dt,
                    self.grid.dsigma,
                    self.grid.thklim,
                    self.grid.n_columns,
                    self.grid.nz,
                    self.grid.c_prime,
                    self.grid.d_prime,
                ),
            )
            denom = cp.maximum(cp.max(cp.abs(self.grid.enthalpy_prev)), 1.0)
            rel = cp.max(cp.abs(self.grid.enthalpy - enthalpy_old)) / denom
            if float(rel) < rtol:
                break
            enthalpy_old[:] = self.grid.enthalpy

        self._update_effective_diffusivity(self.grid.enthalpy)

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

    def get_temperature(self):
        """Return cold-ice temperature reconstructed from enthalpy."""
        return cp.minimum(self.grid.enthalpy, self.enthalpy_pmp)

    def get_liquid_fraction(self):
        """Return temperate liquid water fraction from temperature-equivalent enthalpy."""
        excess = cp.maximum(self.grid.enthalpy - self.enthalpy_pmp, 0.0)
        return excess / cp.float32(LATENT_HEAT_EQUIVALENT)

    def get_temperate_fraction(self):
        """Return the fraction of temperate layers in each column."""
        return (self.grid.enthalpy > self.enthalpy_pmp).mean(axis=2)

    @staticmethod
    def thermal_diffusivity_from_si(value_m2_s):
        """Convert thermal diffusivity from m^2/s to m^2/yr."""
        return cp.float32(value_m2_s * SECONDS_PER_YEAR)
