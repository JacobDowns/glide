"""
GLIDE: GPU-accelerated Lightweight Ice Dynamics Engine

A CUDA-accelerated ice sheet model implementing the shallow shelf approximation (SSA)
with support for forward simulation and adjoint-based inverse modeling.
"""

from .physics import IcePhysics
from .enthalpy import EnthalpyPhysics
from .grid import Grid
from .io import VTIWriter, VTSWriter, HDF5Writer

__version__ = "0.1.0"
__all__ = ["IcePhysics", "EnthalpyPhysics", "Grid", "VTIWriter", "VTSWriter", "HDF5Writer"]
