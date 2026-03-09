"""
Idealized hemispherical ice-sheet growth example.

Builds an ice sheet from a small uniform initial thickness on a
100 km x 100 km domain using a radially symmetric SMB pattern,
then saves the spun-up geometry for later transient experiments.
"""

import cupy as cp
import numpy as np
from pathlib import Path

from glide import IcePhysics
from glide.io import VTIWriter, write_vti


# =============================================================================
# Configuration
# =============================================================================

OUTPUT_DIR = "./ideal_init_output"
GEOMETRY_FILE = f"{OUTPUT_DIR}/ideal_init_geometry.npz"

DOMAIN_SIZE = 100_000.0   # m
DX = 2_500.0              # m
NX = int(DOMAIN_SIZE / DX)
NY = int(DOMAIN_SIZE / DX)

DT = 10.0                  # years
N_STEPS = 500
N_LEVELS = 4
N_VCYCLES = 12

INITIAL_THICKNESS = 10.0  # m
DOME_RADIUS = 40_000.0    # m
DOME_HEIGHT = 1_500.0     # m
SMB_EDGE = -4.0           # m / yr
SMB_CENTER = 4.0          # m / yr

BETA = 2.0
RHO_ICE = 917.0
G = 9.81
N_GLEN = 3.0


def build_ideal_fields(nx, ny, dx):
    """Create a flat bed plus radial hemispherical reference fields."""
    x = (cp.arange(nx, dtype=cp.float32) + 0.5) * dx - 0.5 * nx * dx
    y = (cp.arange(ny, dtype=cp.float32) + 0.5) * dx - 0.5 * ny * dx
    xx, yy = cp.meshgrid(x, y)

    radius = cp.sqrt(xx ** 2 + yy ** 2)
    radial_fraction = cp.clip(radius / DOME_RADIUS, 0.0, 1.0)
    hemisphere = cp.sqrt(cp.clip(1.0 - radial_fraction ** 2, 0.0, 1.0))

    bed = cp.zeros((ny, nx), dtype=cp.float32)
    reference_thickness = DOME_HEIGHT * hemisphere
    smb = SMB_EDGE + (SMB_CENTER - SMB_EDGE) * hemisphere
    thickness = cp.full((ny, nx), INITIAL_THICKNESS, dtype=cp.float32)

    return bed, thickness, smb.astype(cp.float32), reference_thickness.astype(cp.float32)


def save_geometry(filename, dx, bed, thickness, smb, reference_thickness, surface):
    """Persist the spun-up geometry for reuse in transient runs."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        dx=np.float32(dx),
        bed=cp.asnumpy(bed),
        thickness=cp.asnumpy(thickness),
        smb=cp.asnumpy(smb),
        reference_thickness=cp.asnumpy(reference_thickness),
        surface=cp.asnumpy(surface),
    )


def main():
    bed, thickness, smb, reference_thickness = build_ideal_fields(NX, NY, DX)

    B_scalar = cp.float32(1e-17 ** (-1.0 / N_GLEN) / (RHO_ICE * G))
    B = cp.full((NY, NX), B_scalar, dtype=cp.float32)
    beta = cp.full((NY, NX), BETA, dtype=cp.float32)

    print(f"Grid: {NY} x {NX}, dx = {DX / 1000.0:.1f} km")
    print(f"Domain: {DOMAIN_SIZE / 1000.0:.1f} km x {DOMAIN_SIZE / 1000.0:.1f} km")
    print(
        "Initial conditions: "
        f"H = {INITIAL_THICKNESS:.1f} m everywhere, "
        f"SMB = [{SMB_EDGE:.1f}, {SMB_CENTER:.1f}] m/yr"
    )

    physics = IcePhysics(
        NY,
        NX,
        DX,
        n_levels=N_LEVELS,
        thklim=0.1,
        n=3.0,
        eps_reg=1e-5,
        m=1.0 / 3.0,
        u_reg=1.0,
        water_drag=1e-5,
        calving_rate=0.0,
        sigmoid_c=0.1,
    )
    physics.set_geometry(bed, thickness)
    physics.set_parameters(B=B, beta=beta, smb=smb)

    grid = physics.grid
    writer = VTIWriter(OUTPUT_DIR, base="ideal_init", dx=DX)

    write_vti(
        f"{OUTPUT_DIR}/initial_fields.vti",
        {
            "bed": bed,
            "smb": smb,
            "initial_thk": thickness,
            "reference_thk": reference_thickness,
        },
        DX,
    )

    t = 0.0
    print(f"Running {N_STEPS} steps of {DT:.1f} years...")
    for step in range(N_STEPS):
        print(
            f"Step {step:03d}: t = {t:6.1f} yr, "
            f"H_mean = {float(grid.H.mean()):7.2f} m, "
            f"H_max = {float(grid.H.max()):7.2f} m"
        )

        physics.forward(
            dt=DT,
            n_vcycles=N_VCYCLES,
            verbose=True,
            update_geometry=True,
            rtol=1e-4,
        )
        t += DT

        u_c, v_c = physics.get_velocities_cell_centered()
        surface = physics.get_surface()
        writer.write_step(
            step,
            t,
            {
                "thk": grid.H,
                "srf": surface,
                "smb": grid.smb,
                "reference_thk": reference_thickness,
                "vel": [u_c, v_c],
            },
        )

    writer.write_pvd()

    final_surface = physics.get_surface()
    save_geometry(
        GEOMETRY_FILE,
        DX,
        grid.bed,
        grid.H,
        grid.smb,
        reference_thickness,
        final_surface,
    )
    print(f"Saved geometry to {GEOMETRY_FILE}")
    print("Done!")


if __name__ == "__main__":
    main()
