"""
Transient hemispherical ice-sheet example with cyclic SMB forcing.

Loads the spun-up geometry from ideal_init.py, then applies a
sinusoidal SMB anomaly so the dome grows and shrinks repeatedly.
"""

from pathlib import Path

import cupy as cp
import numpy as np

from glide import IcePhysics
from glide.io import VTIWriter, write_vti

from ideal_init import GEOMETRY_FILE, N_LEVELS, N_VCYCLES, RHO_ICE, G, N_GLEN


OUTPUT_DIR = "./ideal_transient_output"
TRANSIENT_GEOMETRY_FILE = f"{OUTPUT_DIR}/ideal_transient_geometry.npz"

DT = 10.0
TOTAL_YEARS = 1_000.0
N_STEPS = int(TOTAL_YEARS / DT)

CYCLE_PERIOD = 250.0
SMB_PERTURBATION_AMPLITUDE = 2.5  # m / yr

BETA = 2.0


def load_geometry(filename):
    """Load the spun-up initial state from disk."""
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing geometry file: {filename}. Run ideal_init.py first."
        )

    with np.load(path) as data:
        dx = float(data["dx"])
        bed = cp.asarray(data["bed"], dtype=cp.float32)
        thickness = cp.asarray(data["thickness"], dtype=cp.float32)
        base_smb = cp.asarray(data["smb"], dtype=cp.float32)
        reference_thickness = cp.asarray(data["reference_thickness"], dtype=cp.float32)

    return dx, bed, thickness, base_smb, reference_thickness


def save_geometry(filename, dx, bed, thickness, smb, reference_thickness, surface):
    """Save the transient end state."""
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


def build_parameters(ny, nx):
    """Construct constant flow and friction parameters."""
    B_scalar = cp.float32(1e-17 ** (-1.0 / N_GLEN) / (RHO_ICE * G))
    B = cp.full((ny, nx), B_scalar, dtype=cp.float32)
    beta = cp.full((ny, nx), BETA, dtype=cp.float32)
    return B, beta


def smb_anomaly(time_years):
    """Periodic SMB anomaly with zero mean."""
    phase = 2.0 * np.pi * time_years / CYCLE_PERIOD
    return SMB_PERTURBATION_AMPLITUDE * np.sin(phase)


def main():
    dx, bed, thickness, base_smb, reference_thickness = load_geometry(GEOMETRY_FILE)
    ny, nx = thickness.shape
    B, beta = build_parameters(ny, nx)

    print(f"Loaded spun-up geometry from {GEOMETRY_FILE}")
    print(f"Grid: {ny} x {nx}, dx = {dx / 1000.0:.1f} km")
    print(
        f"Transient run: {TOTAL_YEARS:.0f} years total, "
        f"{CYCLE_PERIOD:.0f}-year SMB cycle, "
        f"amplitude = {SMB_PERTURBATION_AMPLITUDE:.2f} m/yr"
    )

    physics = IcePhysics(
        ny,
        nx,
        dx,
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
    physics.set_parameters(B=B, beta=beta, smb=base_smb)

    grid = physics.grid
    writer = VTIWriter(OUTPUT_DIR, base="ideal_transient", dx=dx)
    write_vti(
        f"{OUTPUT_DIR}/initial_state.vti",
        {
            "bed": bed,
            "thk": thickness,
            "base_smb": base_smb,
            "reference_thk": reference_thickness,
        },
        dx,
    )

    t = 0.0
    for step in range(N_STEPS):
        midpoint_time = t + 0.5 * DT
        anomaly = smb_anomaly(midpoint_time)
        current_smb = base_smb + cp.float32(anomaly)
        physics.set_parameters(smb=current_smb)

        print(
            f"Step {step:03d}: t = {t:7.1f} yr, "
            f"SMB anomaly = {anomaly:+5.2f} m/yr, "
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
                "smb_anomaly": cp.full((ny, nx), anomaly, dtype=cp.float32),
                "reference_thk": reference_thickness,
                "vel": [u_c, v_c],
            },
        )

    writer.write_pvd()

    final_surface = physics.get_surface()
    save_geometry(
        TRANSIENT_GEOMETRY_FILE,
        dx,
        grid.bed,
        grid.H,
        grid.smb,
        reference_thickness,
        final_surface,
    )
    print(f"Saved transient geometry to {TRANSIENT_GEOMETRY_FILE}")
    print("Done!")


if __name__ == "__main__":
    main()
