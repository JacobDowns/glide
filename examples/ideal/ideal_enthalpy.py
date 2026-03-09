"""
Fixed-geometry enthalpy diffusion test on the spun-up ideal dome.

This example loads the steady dome geometry from ideal_init.py and
solves only vertical diffusion in terrain-following coordinates using
implicit time stepping with one Thomas solve per ice column.
"""

from pathlib import Path

import cupy as cp
import numpy as np

from glide import EnthalpyPhysics
from glide.io import VTIWriter, VTSWriter, write_vti
from ideal_init import GEOMETRY_FILE


OUTPUT_DIR = "./ideal_enthalpy_output"
STATE_FILE = f"{OUTPUT_DIR}/ideal_enthalpy_state.npz"
STATIC_OUTPUT_DIR = f"{OUTPUT_DIR}/static"
VOLUME_OUTPUT_DIR = f"{OUTPUT_DIR}/volume"
CROSS_SECTION_OUTPUT_DIR = f"{OUTPUT_DIR}/cross_sections"

NZ = 32
DT = 20.0
N_STEPS = 10000
OUTPUT_EVERY = 1000

THERMAL_CONDUCTIVITY = 2.1  # W m^-1 K^-1
THERMAL_DIFFUSIVITY_M2_S = 1.09e-6
GEOTHERMAL_FLUX = 0.0      # W m^-2

SEA_LEVEL_SURFACE_TEMPERATURE = -5.0  # degC
LAPSE_RATE = 0.0 #6.5e-3                   # K m^-1


def load_geometry(filename):
    """Load the ideal spun-up dome geometry."""
    with np.load(filename) as data:
        dx = float(data["dx"])
        bed = cp.asarray(data["bed"], dtype=cp.float32)
        thickness = cp.asarray(data["thickness"], dtype=cp.float32)
        surface = cp.asarray(data["surface"], dtype=cp.float32)
    return dx, bed, thickness, surface


def build_boundary_fields(surface, thickness):
    """Construct surface enthalpy and geothermal forcing fields."""
    surface_enthalpy = SEA_LEVEL_SURFACE_TEMPERATURE - LAPSE_RATE * cp.maximum(surface, 0.0)
    geothermal_flux = cp.full_like(thickness, GEOTHERMAL_FLUX, dtype=cp.float32)
    return surface_enthalpy.astype(cp.float32), geothermal_flux


def build_steady_profile(surface_enthalpy, geothermal_flux, thickness, conductivity, nz):
    """Analytic steady conductive profile for the cold-only column model."""
    sigma = (cp.arange(nz, dtype=cp.float32) + 0.5) / nz
    height = sigma[cp.newaxis, cp.newaxis, :] * thickness[:, :, cp.newaxis]
    basal_enthalpy = surface_enthalpy + geothermal_flux * thickness / conductivity
    gradient = geothermal_flux / conductivity
    return basal_enthalpy[:, :, cp.newaxis] - gradient[:, :, cp.newaxis] * height


def save_state(filename, physics, surface_enthalpy, geothermal_flux, steady_profile):
    """Persist the full 3D enthalpy state and forcing fields."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        dx=np.float32(float(physics.grid.dx)),
        bed=cp.asnumpy(physics.grid.bed),
        thickness=cp.asnumpy(physics.grid.H),
        enthalpy=cp.asnumpy(physics.grid.enthalpy),
        surface_enthalpy=cp.asnumpy(surface_enthalpy),
        geothermal_flux=cp.asnumpy(geothermal_flux),
        steady_profile=cp.asnumpy(steady_profile),
        sigma=cp.asnumpy(physics.grid.sigma_centers()),
    )


def write_static_field_collections(dx, fields):
    """Write one-field ParaView collections for static inputs and references."""
    for name, field in fields.items():
        writer = VTIWriter(STATIC_OUTPUT_DIR, base=name, dx=dx)
        writer.write_step(0, 0.0, {name: field})
        writer.write_pvd()


def cell_to_node(field):
    """Average a cell-centered 2D field to grid nodes."""
    ny, nx = field.shape
    node = cp.zeros((ny + 1, nx + 1), dtype=cp.float32)
    count = cp.zeros((ny + 1, nx + 1), dtype=cp.float32)

    node[:-1, :-1] += field
    node[1:, :-1] += field
    node[:-1, 1:] += field
    node[1:, 1:] += field

    count[:-1, :-1] += 1.0
    count[1:, :-1] += 1.0
    count[:-1, 1:] += 1.0
    count[1:, 1:] += 1.0

    return node / count


def build_volume_points(bed, thickness, dx, nz):
    """Construct physical-space structured-grid points for the sigma layers."""
    ny, nx = bed.shape
    x_nodes = cp.arange(nx + 1, dtype=cp.float32) * dx - 0.5 * nx * dx
    y_nodes = cp.arange(ny + 1, dtype=cp.float32) * dx - 0.5 * ny * dx
    sigma_nodes = cp.linspace(0.0, 1.0, nz + 1, dtype=cp.float32)

    xx, yy = cp.meshgrid(x_nodes, y_nodes)
    bed_nodes = cell_to_node(bed)
    thickness_nodes = cell_to_node(thickness)

    x_points = cp.broadcast_to(xx[:, :, cp.newaxis], (ny + 1, nx + 1, nz + 1))
    y_points = cp.broadcast_to(yy[:, :, cp.newaxis], (ny + 1, nx + 1, nz + 1))
    z_points = bed_nodes[:, :, cp.newaxis] + thickness_nodes[:, :, cp.newaxis] * sigma_nodes[cp.newaxis, cp.newaxis, :]

    return x_points, y_points, z_points


def centerline_average(field, axis):
    """
    Extract a centerline section from an even or odd cell-centered field.

    For even-sized grids, average the two central rows/columns so the section
    lies on the geometric centerline of the dome.
    """
    if axis == "x":
        n = field.shape[0]
        if n % 2 == 0:
            lower = n // 2 - 1
            upper = n // 2
            return 0.5 * (field[lower] + field[upper])
        return field[n // 2]

    if axis == "y":
        n = field.shape[1]
        if n % 2 == 0:
            lower = n // 2 - 1
            upper = n // 2
            return 0.5 * (field[:, lower] + field[:, upper])
        return field[:, n // 2]

    raise ValueError(f"Unsupported axis: {axis}")


def build_cross_section_points(bed, thickness, dx, nz, axis):
    """Construct a 2D physical-space structured grid for a centerline section."""
    ny, nx = bed.shape
    x_nodes = cp.arange(nx + 1, dtype=cp.float32) * dx - 0.5 * nx * dx
    y_nodes = cp.arange(ny + 1, dtype=cp.float32) * dx - 0.5 * ny * dx
    sigma_nodes = cp.linspace(0.0, 1.0, nz + 1, dtype=cp.float32)

    bed_nodes = cell_to_node(bed)
    thickness_nodes = cell_to_node(thickness)

    if axis == "x":
        mid = ny // 2
        x_line = x_nodes
        y_line = cp.full_like(x_line, y_nodes[mid])
        bed_line = bed_nodes[mid]
        thickness_line = thickness_nodes[mid]
        x_points = cp.broadcast_to(x_line[cp.newaxis, :, cp.newaxis], (1, nx + 1, nz + 1))
        y_points = cp.broadcast_to(y_line[cp.newaxis, :, cp.newaxis], (1, nx + 1, nz + 1))
        z_points = bed_line[cp.newaxis, :, cp.newaxis] + thickness_line[cp.newaxis, :, cp.newaxis] * sigma_nodes[cp.newaxis, cp.newaxis, :]
        return x_points, y_points, z_points

    if axis == "y":
        mid = nx // 2
        y_line = y_nodes
        x_line = cp.full_like(y_line, x_nodes[mid])
        bed_line = bed_nodes[:, mid]
        thickness_line = thickness_nodes[:, mid]
        x_points = cp.broadcast_to(x_line[:, cp.newaxis, cp.newaxis], (ny + 1, 1, nz + 1))
        y_points = cp.broadcast_to(y_line[:, cp.newaxis, cp.newaxis], (ny + 1, 1, nz + 1))
        z_points = bed_line[:, cp.newaxis, cp.newaxis] + thickness_line[:, cp.newaxis, cp.newaxis] * sigma_nodes[cp.newaxis, cp.newaxis, :]
        return x_points, y_points, z_points

    raise ValueError(f"Unsupported axis: {axis}")


def plot_cross_section(points, enthalpy, misfit, axis, output_path):
    """Save a matplotlib plot for one final-time centerline section."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required to plot cross sections") from exc

    if axis == "x":
        horizontal = cp.asnumpy(points[0][0])
        vertical = cp.asnumpy(points[2][0])
        xlabel = "x (m)"
        title_axis = "x-z"
    elif axis == "y":
        horizontal = cp.asnumpy(points[1][:, 0, :])
        vertical = cp.asnumpy(points[2][:, 0, :])
        xlabel = "y (m)"
        title_axis = "y-z"
    else:
        raise ValueError(f"Unsupported axis: {axis}")

    enthalpy_np = cp.asnumpy(enthalpy)
    misfit_np = cp.asnumpy(misfit)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    enthalpy_plot = axes[0].pcolormesh(horizontal, vertical, enthalpy_np, shading="auto", cmap="coolwarm")
    axes[0].set_title(f"Final {title_axis} Enthalpy")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("z (m)")
    fig.colorbar(enthalpy_plot, ax=axes[0], label="enthalpy")

    misfit_limit = max(np.max(np.abs(misfit_np)), 1.0e-6)
    misfit_plot = axes[1].pcolormesh(
        horizontal,
        vertical,
        misfit_np,
        shading="auto",
        cmap="RdBu_r",
        vmin=-misfit_limit,
        vmax=misfit_limit,
    )
    axes[1].set_title(f"Final {title_axis} Enthalpy Misfit")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("z (m)")
    fig.colorbar(misfit_plot, ax=axes[1], label="enthalpy - analytic")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_final_cross_sections(physics, steady_profile, dx, final_time):
    """Write final x-z and y-z centerline sections for ParaView and Matplotlib."""
    enthalpy = physics.grid.enthalpy
    misfit = enthalpy - steady_profile

    xz_points = build_cross_section_points(physics.grid.bed, physics.grid.H, dx, physics.nz, axis="x")
    yz_points = build_cross_section_points(physics.grid.bed, physics.grid.H, dx, physics.nz, axis="y")

    xz_enthalpy = centerline_average(enthalpy, axis="x")
    xz_steady = centerline_average(steady_profile, axis="x")
    xz_misfit = centerline_average(misfit, axis="x")

    xz_writer = VTSWriter(CROSS_SECTION_OUTPUT_DIR, base="enthalpy_xz_final")
    xz_writer.write_step(
        0,
        final_time,
        xz_points,
        cell_data={
            "enthalpy": xz_enthalpy[cp.newaxis, :, :],
            "steady_enthalpy": xz_steady[cp.newaxis, :, :],
            "enthalpy_misfit": xz_misfit[cp.newaxis, :, :],
        },
    )
    xz_writer.write_pvd()
    plot_cross_section(
        xz_points,
        xz_enthalpy,
        xz_misfit,
        axis="x",
        output_path=f"{CROSS_SECTION_OUTPUT_DIR}/enthalpy_xz_final.png",
    )

    yz_enthalpy = centerline_average(enthalpy, axis="y")
    yz_steady = centerline_average(steady_profile, axis="y")
    yz_misfit = centerline_average(misfit, axis="y")

    yz_writer = VTSWriter(CROSS_SECTION_OUTPUT_DIR, base="enthalpy_yz_final")
    yz_writer.write_step(
        0,
        final_time,
        yz_points,
        cell_data={
            "enthalpy": yz_enthalpy[:, cp.newaxis, :],
            "steady_enthalpy": yz_steady[:, cp.newaxis, :],
            "enthalpy_misfit": yz_misfit[:, cp.newaxis, :],
        },
    )
    yz_writer.write_pvd()
    plot_cross_section(
        yz_points,
        yz_enthalpy,
        yz_misfit,
        axis="y",
        output_path=f"{CROSS_SECTION_OUTPUT_DIR}/enthalpy_yz_final.png",
    )


def main():
    geometry_path = Path(GEOMETRY_FILE)
    if not geometry_path.exists():
        raise FileNotFoundError(
            f"Missing geometry file: {GEOMETRY_FILE}. Run ideal_init.py first."
        )

    dx, bed, thickness, surface = load_geometry(geometry_path)
    ny, nx = thickness.shape

    physics = EnthalpyPhysics(ny, nx, NZ, dx, thklim=0.1)
    physics.set_geometry(bed, thickness)

    surface_enthalpy, geothermal_flux = build_boundary_fields(surface, thickness)
    diffusivity = physics.thermal_diffusivity_from_si(THERMAL_DIFFUSIVITY_M2_S)
    conductivity = cp.full_like(thickness, THERMAL_CONDUCTIVITY, dtype=cp.float32)

    physics.set_boundary_conditions(
        surface_enthalpy=surface_enthalpy,
        geothermal_flux=geothermal_flux,
    )
    physics.set_parameters(diffusivity=diffusivity, conductivity=conductivity)

    initial_enthalpy = cp.broadcast_to(
        surface_enthalpy[:, :, cp.newaxis],
        (ny, nx, NZ),
    ).copy()
    physics.set_initial_enthalpy(initial_enthalpy)

    steady_profile = build_steady_profile(
        surface_enthalpy,
        geothermal_flux,
        thickness,
        conductivity,
        NZ,
    )

    static_fields = {
        "bed": bed,
        "thk": thickness,
        "surface_enthalpy": surface_enthalpy,
        "geothermal_flux": geothermal_flux,
        "steady_basal_enthalpy": steady_profile[:, :, 0],
    }

    writer = VTIWriter(OUTPUT_DIR, base="ideal_enthalpy", dx=dx)
    volume_writer = VTSWriter(VOLUME_OUTPUT_DIR, base="ideal_enthalpy_volume")
    write_vti(
        f"{OUTPUT_DIR}/initial_fields.vti",
        static_fields,
        dx,
    )
    write_static_field_collections(dx, static_fields)
    volume_points = build_volume_points(bed, thickness, dx, NZ)
    volume_writer.write_step(
        0,
        0.0,
        volume_points,
        cell_data={
            "enthalpy": physics.grid.enthalpy,
            "steady_enthalpy": steady_profile,
            "enthalpy_misfit": physics.grid.enthalpy - steady_profile,
        },
    )

    print(f"Loaded geometry from {GEOMETRY_FILE}")
    print(f"Grid: {ny} x {nx} x {NZ}, dx = {dx / 1000.0:.1f} km")
    print(
        f"Running {N_STEPS} implicit steps of {DT:.1f} years "
        f"with diffusivity = {float(diffusivity):.2f} m^2/yr"
    )

    output_index = 1
    t = 0.0
    for step in range(N_STEPS):
        physics.forward(DT)
        t += DT

        if step % OUTPUT_EVERY == 0 or step == N_STEPS - 1:
            mean_enthalpy = physics.get_mean_enthalpy()
            basal_enthalpy = physics.get_basal_enthalpy()
            surface_layer = physics.get_surface_layer_enthalpy()
            steady_misfit = cp.sqrt(cp.mean((physics.grid.enthalpy - steady_profile) ** 2, axis=2))

            print(
                f"Step {step:03d}: t = {t:7.1f} yr, "
                f"E_mean = {float(mean_enthalpy.mean()):7.2f} C, "
                f"E_basal_max = {float(basal_enthalpy.max()):7.2f} C, "
                f"rmse = {float(cp.sqrt(cp.mean((physics.grid.enthalpy - steady_profile) ** 2))):7.3f} C"
            )

            writer.write_step(
                output_index,
                t,
                {
                    "mean_enthalpy": mean_enthalpy,
                    "basal_enthalpy": basal_enthalpy,
                    "surface_layer_enthalpy": surface_layer,
                    "surface_enthalpy": surface_enthalpy,
                    "steady_basal_enthalpy": steady_profile[:, :, 0],
                    "steady_rmse": steady_misfit,
                },
            )
            volume_writer.write_step(
                output_index,
                t,
                volume_points,
                cell_data={
                    "enthalpy": physics.grid.enthalpy,
                    "steady_enthalpy": steady_profile,
                    "enthalpy_misfit": physics.grid.enthalpy - steady_profile,
                },
            )
            output_index += 1

    writer.write_pvd()
    volume_writer.write_pvd()
    write_final_cross_sections(physics, steady_profile, dx, t)
    save_state(STATE_FILE, physics, surface_enthalpy, geothermal_flux, steady_profile)
    print(f"Saved state to {STATE_FILE}")
    print("Done!")


if __name__ == "__main__":
    main()
