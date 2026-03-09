"""
Input/output utilities for GLIDE.

Provides VTI (ParaView) and HDF5 output writers for visualization and analysis.
"""

import numpy as np
import cupy as cp
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
import h5py


def _pretty_xml(element):
    """Format XML element with indentation."""
    return minidom.parseString(ET.tostring(element)).toprettyxml(indent="  ")


def write_vti(filename, data, dx, dy=None, origin=(0.0, 0.0), time_value=None, flip_y=True):
    """
    Write fields to VTI (VTK ImageData) binary format.

    Parameters
    ----------
    filename : str or Path
        Output filename
    data : dict
        Dictionary mapping field names to data. Values can be:
        - CuPy/NumPy array: scalar field
        - List of arrays: vector field components
    dx : float
        Grid spacing in x
    dy : float, optional
        Grid spacing in y (defaults to dx)
    origin : tuple
        Grid origin (x, y)
    time_value : float, optional
        Time value for this snapshot
    flip_y : bool
        Flip arrays along y-axis (convert from image to VTK convention)
    """
    if dy is None:
        dy = dx

    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    # Separate scalars and vectors, transfer to CPU
    scalars = {}
    vectors = {}

    for name, value in data.items():
        if isinstance(value, list):
            components = [cp.asnumpy(c).astype(np.float32) for c in value]
            if flip_y:
                components = [np.flip(c, axis=0) for c in components]
            vectors[name] = components
        else:
            arr = cp.asnumpy(value).astype(np.float32)
            if flip_y:
                arr = np.flip(arr, axis=0)
            scalars[name] = arr

    # Get grid dimensions
    if scalars:
        first_field = next(iter(scalars.values()))
    else:
        first_field = next(iter(vectors.values()))[0]
    ny, nx = first_field.shape

    # Build XML
    root = ET.Element("VTKFile", type="ImageData", version="1.0", byte_order="LittleEndian")
    img = ET.SubElement(root, "ImageData",
                        WholeExtent=f"0 {nx-1} 0 {ny-1} 0 0",
                        Origin=f"{origin[0]} {origin[1]} 0",
                        Spacing=f"{dx} {dy} 1.0")
    piece = ET.SubElement(img, "Piece", Extent=f"0 {nx-1} 0 {ny-1} 0 0")

    if time_value is not None:
        fd = ET.SubElement(piece, "FieldData")
        da = ET.SubElement(fd, "DataArray",
                           type="Float32", Name="TimeValue",
                           NumberOfComponents="1", format="ascii")
        da.text = f"\n{float(time_value)}\n"

    pd_attrs = {}
    if scalars:
        pd_attrs["Scalars"] = next(iter(scalars.keys()))
    for name, components in vectors.items():
        if len(components) == 3:
            pd_attrs["Vectors"] = name
            break
    pd = ET.SubElement(piece, "PointData", **pd_attrs)

    binary_arrays = []
    offset = 0

    for name, arr in scalars.items():
        arr_bytes = arr.ravel(order='C').tobytes()
        ET.SubElement(pd, "DataArray",
                      type="Float32", Name=name,
                      NumberOfComponents="1",
                      format="appended",
                      offset=str(offset))
        binary_arrays.append(arr_bytes)
        offset += len(arr_bytes) + 4

    for name, components in vectors.items():
        ncomp = len(components)
        stacked = np.stack(components, axis=-1).astype(np.float32)
        vec_bytes = stacked.ravel(order='C').tobytes()
        ET.SubElement(pd, "DataArray",
                      type="Float32", Name=name,
                      NumberOfComponents=str(ncomp),
                      format="appended",
                      offset=str(offset))
        binary_arrays.append(vec_bytes)
        offset += len(vec_bytes) + 4

    xml_bytes = ET.tostring(root, encoding='utf-8', xml_declaration=True)

    with open(filename, 'wb') as f:
        xml_str = xml_bytes.decode('utf-8')
        if xml_str.endswith('</VTKFile>'):
            xml_str = xml_str[:-len('</VTKFile>')]
        elif xml_str.endswith('</VTKFile>\n'):
            xml_str = xml_str[:-len('</VTKFile>\n')]

        f.write(xml_str.encode('utf-8'))
        f.write(b'  <AppendedData encoding="raw">\n   _')

        for array_bytes in binary_arrays:
            size = np.uint32(len(array_bytes))
            f.write(size.tobytes())
            f.write(array_bytes)

        f.write(b'\n  </AppendedData>\n</VTKFile>\n')


def write_vts(filename, points, cell_data=None, point_data=None, time_value=None, flip_y=True):
    """
    Write a 3D structured grid (.vts) for ParaView.

    Parameters
    ----------
    filename : str or Path
        Output filename
    points : list[array_like]
        Grid point coordinates [x, y, z], each shaped (ny+1, nx+1, nz+1)
    cell_data : dict, optional
        Cell-centered fields shaped (ny, nx, nz)
    point_data : dict, optional
        Point-centered scalar fields shaped (ny+1, nx+1, nz+1)
    time_value : float, optional
        Time value for this snapshot
    flip_y : bool
        Flip arrays along y-axis to match VTK convention used elsewhere
    """
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    if cell_data is None:
        cell_data = {}
    if point_data is None:
        point_data = {}

    x, y, z = [cp.asnumpy(arr).astype(np.float32) for arr in points]
    if flip_y:
        x = np.flip(x, axis=0)
        y = np.flip(y, axis=0)
        z = np.flip(z, axis=0)

    ny_p, nx_p, nz_p = x.shape
    ny = ny_p - 1
    nx = nx_p - 1
    nz = nz_p - 1

    cell_scalars = {}
    for name, value in cell_data.items():
        arr = cp.asnumpy(value).astype(np.float32)
        if flip_y:
            arr = np.flip(arr, axis=0)
        cell_scalars[name] = arr

    point_scalars = {}
    for name, value in point_data.items():
        arr = cp.asnumpy(value).astype(np.float32)
        if flip_y:
            arr = np.flip(arr, axis=0)
        point_scalars[name] = arr

    root = ET.Element("VTKFile", type="StructuredGrid", version="1.0", byte_order="LittleEndian")
    grid = ET.SubElement(root, "StructuredGrid", WholeExtent=f"0 {nx} 0 {ny} 0 {nz}")
    piece = ET.SubElement(grid, "Piece", Extent=f"0 {nx} 0 {ny} 0 {nz}")

    if time_value is not None:
        fd = ET.SubElement(piece, "FieldData")
        da = ET.SubElement(
            fd,
            "DataArray",
            type="Float32",
            Name="TimeValue",
            NumberOfComponents="1",
            format="ascii",
        )
        da.text = f"\n{float(time_value)}\n"

    cd_attrs = {}
    if cell_scalars:
        cd_attrs["Scalars"] = next(iter(cell_scalars.keys()))
    cell_node = ET.SubElement(piece, "CellData", **cd_attrs)

    pd_attrs = {}
    if point_scalars:
        pd_attrs["Scalars"] = next(iter(point_scalars.keys()))
    point_node = ET.SubElement(piece, "PointData", **pd_attrs)
    points_node = ET.SubElement(piece, "Points")

    binary_arrays = []
    offset = 0

    for name, arr in cell_scalars.items():
        arr_bytes = np.transpose(arr, (2, 0, 1)).ravel(order="C").tobytes()
        ET.SubElement(
            cell_node,
            "DataArray",
            type="Float32",
            Name=name,
            NumberOfComponents="1",
            format="appended",
            offset=str(offset),
        )
        binary_arrays.append(arr_bytes)
        offset += len(arr_bytes) + 4

    for name, arr in point_scalars.items():
        arr_bytes = np.transpose(arr, (2, 0, 1)).ravel(order="C").tobytes()
        ET.SubElement(
            point_node,
            "DataArray",
            type="Float32",
            Name=name,
            NumberOfComponents="1",
            format="appended",
            offset=str(offset),
        )
        binary_arrays.append(arr_bytes)
        offset += len(arr_bytes) + 4

    stacked_points = np.stack(
        (
            np.transpose(x, (2, 0, 1)),
            np.transpose(y, (2, 0, 1)),
            np.transpose(z, (2, 0, 1)),
        ),
        axis=-1,
    ).astype(np.float32)
    point_bytes = stacked_points.ravel(order="C").tobytes()
    ET.SubElement(
        points_node,
        "DataArray",
        type="Float32",
        Name="Points",
        NumberOfComponents="3",
        format="appended",
        offset=str(offset),
    )
    binary_arrays.append(point_bytes)
    offset += len(point_bytes) + 4

    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)

    with open(filename, "wb") as f:
        xml_str = xml_bytes.decode("utf-8")
        if xml_str.endswith("</VTKFile>"):
            xml_str = xml_str[:-len("</VTKFile>")]
        elif xml_str.endswith("</VTKFile>\n"):
            xml_str = xml_str[:-len("</VTKFile>\n")]

        f.write(xml_str.encode("utf-8"))
        f.write(b'  <AppendedData encoding="raw">\n   _')

        for array_bytes in binary_arrays:
            size = np.uint32(len(array_bytes))
            f.write(size.tobytes())
            f.write(array_bytes)

        f.write(b'\n  </AppendedData>\n</VTKFile>\n')


class VTIWriter:
    """
    Write a time series of VTI files with a PVD manifest.

    ParaView can open the .pvd file to load the entire time series.

    Parameters
    ----------
    out_dir : str or Path
        Output directory
    base : str
        Base name for files (default "output")
    dx : float
        Grid spacing
    origin : tuple
        Grid origin (x, y)
    flip_y : bool
        Flip arrays along y-axis

    Examples
    --------
    >>> writer = VTIWriter("results", dx=1500.0)
    >>> for t in range(100):
    ...     writer.write_step(t, t * dt, {"H": thickness, "vel": [u, v]})
    >>> writer.write_pvd()
    """

    def __init__(self, out_dir, base="output", dx=1.0, dy=None, origin=(0.0, 0.0), flip_y=True):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.base = base
        self.dx = dx
        self.dy = dy if dy is not None else dx
        self.origin = origin
        self.flip_y = flip_y
        self.records = []

    def write_step(self, step_idx, time_value, data):
        """Write a timestep to a numbered VTI file."""
        fname = f"{self.base}_{step_idx:04d}.vti"
        fpath = self.out_dir / fname
        write_vti(fpath, data, self.dx, self.dy, self.origin,
                  time_value=time_value, flip_y=self.flip_y)
        self.records.append((float(time_value), fname))
        return fpath

    def write_pvd(self, pvd_name=None):
        """Write the PVD manifest file."""
        if pvd_name is None:
            pvd_name = f"{self.base}.pvd"

        root = ET.Element("VTKFile", type="Collection", version="0.1", byte_order="LittleEndian")
        coll = ET.SubElement(root, "Collection")
        for t, fname in self.records:
            ET.SubElement(coll, "DataSet",
                          timestep=str(t),
                          group="",
                          part="0",
                          file=str(fname))

        with open(self.out_dir / pvd_name, "w") as f:
            f.write(_pretty_xml(root))
        return self.out_dir / pvd_name


class VTSWriter:
    """
    Write a time series of VTS files with a PVD manifest.

    Parameters
    ----------
    out_dir : str or Path
        Output directory
    base : str
        Base name for files
    flip_y : bool
        Flip arrays along y-axis
    """

    def __init__(self, out_dir, base="output", flip_y=True):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.base = base
        self.flip_y = flip_y
        self.records = []

    def write_step(self, step_idx, time_value, points, cell_data=None, point_data=None):
        """Write a timestep to a numbered VTS file."""
        fname = f"{self.base}_{step_idx:04d}.vts"
        fpath = self.out_dir / fname
        write_vts(
            fpath,
            points,
            cell_data=cell_data,
            point_data=point_data,
            time_value=time_value,
            flip_y=self.flip_y,
        )
        self.records.append((float(time_value), fname))
        return fpath

    def write_pvd(self, pvd_name=None):
        """Write the PVD manifest file."""
        if pvd_name is None:
            pvd_name = f"{self.base}.pvd"

        root = ET.Element("VTKFile", type="Collection", version="0.1", byte_order="LittleEndian")
        coll = ET.SubElement(root, "Collection")
        for t, fname in self.records:
            ET.SubElement(coll, "DataSet", timestep=str(t), group="", part="0", file=str(fname))

        with open(self.out_dir / pvd_name, "w") as f:
            f.write(_pretty_xml(root))
        return self.out_dir / pvd_name


class HDF5Writer:
    """
    Write a time series to a single HDF5 file.

    Parameters
    ----------
    filename : str or Path
        Output HDF5 filename
    dx : float
        Grid spacing
    origin : tuple
        Grid origin (x, y)
    compression : str
        Compression filter ('lzf', 'gzip', or None)

    Examples
    --------
    >>> with HDF5Writer("results.h5", dx=1500.0) as writer:
    ...     for t in range(100):
    ...         writer.write_step(t * dt, {"H": thickness, "u": u, "v": v})
    """

    def __init__(self, filename, dx=1.0, dy=None, origin=(0.0, 0.0),
                 compression='lzf', compression_opts=None):
        self.filename = Path(filename)
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        self.dx = dx
        self.dy = dy if dy is not None else dx
        self.origin = origin
        self.compression = compression
        self.compression_opts = compression_opts
        self.step_count = 0

        with h5py.File(self.filename, 'w') as f:
            f.attrs['dx'] = dx
            f.attrs['dy'] = self.dy
            f.attrs['origin_x'] = origin[0]
            f.attrs['origin_y'] = origin[1]

    def write_step(self, time_value, data):
        """Write a timestep to a new group."""
        group_name = f"t{self.step_count:04d}"

        with h5py.File(self.filename, 'a') as f:
            grp = f.create_group(group_name)
            grp.attrs['time'] = float(time_value)

            for name, value in data.items():
                if isinstance(value, list):
                    suffixes = ['_x', '_y', '_z'][:len(value)]
                    for component, suffix in zip(value, suffixes):
                        arr = cp.asnumpy(component).astype(np.float32)
                        grp.create_dataset(
                            name + suffix, data=arr,
                            compression=self.compression,
                            compression_opts=self.compression_opts
                        )
                else:
                    arr = cp.asnumpy(value).astype(np.float32)
                    grp.create_dataset(
                        name, data=arr,
                        compression=self.compression,
                        compression_opts=self.compression_opts
                    )

        self.step_count += 1
        return group_name

    def close(self):
        """No-op for compatibility."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
