"""BMI 2.0 numerical interface machinery shared by the public adapters.

The 1-D arrays are *index-space vectors*, not physical rasters.  A separate
parcel/outlet ID mapping is required to infer spatial locations.  No BMI
variable here claims to use a reviewed CSDMS Standard Name.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from basin_bmp_sim.interfaces.exchange_variables import ExchangeVariable, numeric_vector


class BmiArrayComponent:
    """Implement BMI's array/metadata/grid methods without scientific logic.

    All non-scalar arrays use distinct 1-D uniform index-space grids.  In
    particular, these virtual grids do not pretend to describe actual parcel
    polygons or outlet point coordinates.  Repeated updates may replace the
    arrays; callers must not retain get_value_ptr() beyond an update.
    """

    def __init__(self, variables: tuple[ExchangeVariable, ...], *, name: str):
        self._component_name = name
        self._metadata = {v.name: v for v in variables}
        if len(self._metadata) != len(variables):
            raise ValueError("Duplicate BMI variable names")
        self._grids = {v.name: idx + 1 for idx, v in enumerate(variables)}
        self._grid_variables = {i: name for name, i in self._grids.items()}
        self._values: dict[str, np.ndarray] = {}
        self._initialized = False
        self._time = 0.0
        self._end_time = 0.0

    def get_component_name(self) -> str:
        return self._component_name

    def get_bmi_version(self) -> str:
        return "2.0"

    def get_input_var_names(self) -> tuple[str, ...]:
        return tuple(k for k, v in self._metadata.items() if v.intent == "input")

    def get_output_var_names(self) -> tuple[str, ...]:
        return tuple(k for k, v in self._metadata.items() if v.intent == "output")

    def get_input_item_count(self) -> int:
        """Return the number of advertised BMI input exchange variables."""
        return len(self.get_input_var_names())

    def get_output_item_count(self) -> int:
        """Return the number of advertised BMI output exchange variables."""
        return len(self.get_output_var_names())

    def _meta(self, name: str) -> ExchangeVariable:
        try:
            return self._metadata[name]
        except KeyError as exc:
            raise KeyError(f"Unknown BMI variable: {name}") from exc

    def _read(self, name: str) -> np.ndarray:
        self._meta(name)
        if not self._initialized:
            raise RuntimeError("Call initialize() before using BMI variables")
        return self._values[name]

    def _publish(self, values: Mapping[str, object]) -> None:
        for name, value in values.items():
            spec = self._meta(name)
            array = numeric_vector(value, spec.dtype)
            if spec.rank == 0 and array.size != 1:
                raise ValueError(f"BMI scalar variable {name} must have one element")
            # Keep a stable pointer for fixed-size arrays where possible.
            old = self._values.get(name)
            if old is not None and old.shape == array.shape:
                old[:] = array
            else:
                self._values[name] = array

    def get_var_units(self, name: str) -> str:
        return self._meta(name).units

    def get_var_type(self, name: str) -> str:
        return str(np.dtype(self._meta(name).dtype))

    def get_var_itemsize(self, name: str) -> int:
        return int(np.dtype(self._meta(name).dtype).itemsize)

    def get_var_nbytes(self, name: str) -> int:
        return self._read(name).nbytes

    def get_var_location(self, name: str) -> str:
        self._meta(name)
        return "node"

    def get_var_grid(self, name: str) -> int:
        self._meta(name)
        return 0 if self._meta(name).rank == 0 else self._grids[name]

    def get_value_ptr(self, name: str) -> np.ndarray:
        return self._read(name)

    @staticmethod
    def _destination(dest: np.ndarray, expected: np.ndarray) -> np.ndarray:
        """Require an actual, writable, contiguous BMI destination buffer.

        ``reshape(-1)`` on a noncontiguous destination can allocate a copy;
        writing to that temporary previously appeared to succeed while
        leaving the caller's original buffer unchanged.
        """
        if (not isinstance(dest, np.ndarray) or dest.ndim != 1
                or dest.dtype != expected.dtype or dest.size != expected.size
                or not dest.flags.c_contiguous or not dest.flags.writeable):
            raise ValueError("BMI destination must be a writable contiguous 1-D array "
                             "with the advertised size and dtype")
        return dest

    @staticmethod
    def _source(src: np.ndarray, dtype: str) -> np.ndarray:
        """Reject silent integer truncation, wrapping, and dtype coercion."""
        arr = np.asarray(src)
        if arr.ndim != 1 or arr.dtype != np.dtype(dtype):
            raise ValueError(f"BMI source must be a 1-D array of dtype {np.dtype(dtype)}")
        if arr.dtype.kind == "f" and not np.all(np.isfinite(arr)):
            raise ValueError("BMI source values must be finite")
        return np.ascontiguousarray(arr)

    @staticmethod
    def _indices(inds: np.ndarray, upper: int, *, unique: bool = False) -> np.ndarray:
        index = np.asarray(inds)
        if (index.ndim != 1 or index.dtype.kind not in "iu"
                or np.any(index < 0) or np.any(index >= upper)):
            raise ValueError("BMI indices must be a 1-D integer array within variable bounds")
        if unique and len(np.unique(index)) != len(index):
            raise ValueError("BMI indices must not contain duplicates")
        return index

    def get_value(self, name: str, dest: np.ndarray) -> None:
        value = self._read(name)
        np.copyto(self._destination(dest, value), value)

    def get_value_at_indices(self, name: str, dest: np.ndarray, inds: np.ndarray) -> None:
        value = self._read(name)
        selected = value[self._indices(inds, len(value))]
        np.copyto(self._destination(dest, selected), selected)

    def set_value(self, name: str, src: np.ndarray) -> None:
        spec = self._meta(name)
        if spec.intent != "input":
            raise ValueError(f"BMI variable is read-only: {name}")
        old = self._read(name)
        new = self._source(src, spec.dtype)
        if old.shape != new.shape:
            if not spec.resizable:
                raise ValueError("Cannot change BMI input array extent with set_value")
            # Query the new BMI grid size; any earlier pointer is invalid.
            self._values[name] = new.copy()
        else:
            old[:] = new

    def set_value_at_indices(self, name: str, inds: np.ndarray, src: np.ndarray) -> None:
        spec = self._meta(name)
        if spec.intent != "input":
            raise ValueError(f"BMI variable is read-only: {name}")
        arr = self._read(name)
        index = self._indices(inds, len(arr), unique=True)
        vals = self._source(src, spec.dtype)
        if len(index) != len(vals):
            raise ValueError("BMI indexed setter length mismatch")
        proposed = arr.copy()
        proposed[index] = vals
        # Dispatch through the public setter: component-specific scientific
        # and lifecycle validation must not be bypassed by indexed writes.
        self.set_value(name, proposed)

    def get_start_time(self) -> float:
        return 0.0

    def get_current_time(self) -> float:
        return float(self._time)

    def get_end_time(self) -> float:
        return float(self._end_time)

    def get_time_step(self) -> float:
        return 1.0

    def get_time_units(self) -> str:
        # Steps count realizations/placements, not physical time.
        return "none"

    def update_until(self, time: float) -> None:
        requested = float(time)
        if not np.isfinite(requested) or requested != int(requested):
            raise ValueError("BMI logical execution position must be an integer")
        if requested < self._time or requested > self._end_time:
            raise ValueError("BMI requested execution position is outside remaining range")
        while self._time < requested:
            self.update()

    def get_grid_type(self, grid: int) -> str:
        return "scalar" if grid == 0 else ("uniform_rectilinear" if grid in self._grid_variables else self._unknown_grid(grid))

    @staticmethod
    def _unknown_grid(grid: int):
        raise ValueError(f"Unknown BMI grid: {grid}")

    def get_grid_rank(self, grid: int) -> int:
        self.get_grid_type(grid)
        return 0 if grid == 0 else 1

    def get_grid_size(self, grid: int) -> int:
        self.get_grid_type(grid)
        return 1 if grid == 0 else self._read(self._grid_variables[grid]).size

    def get_grid_shape(self, grid: int, shape: np.ndarray) -> None:
        self.get_grid_type(grid)
        required = 0 if grid == 0 else 1
        if shape.size != required:
            raise ValueError("BMI grid shape destination rank mismatch")
        if required:
            shape[0] = self.get_grid_size(grid)

    def get_grid_spacing(self, grid: int, spacing: np.ndarray) -> None:
        self._index_grid_vector(grid, spacing, 1.0)

    def get_grid_origin(self, grid: int, origin: np.ndarray) -> None:
        self._index_grid_vector(grid, origin, 0.0)

    def _index_grid_vector(self, grid: int, dest: np.ndarray, value: float) -> None:
        if self.get_grid_rank(grid) != 1 or dest.size != 1:
            raise ValueError("Index-space grid requires one coordinate component")
        dest[0] = value

    def get_grid_x(self, grid: int, x: np.ndarray) -> None:
        if self.get_grid_rank(grid) != 1 or x.size != self.get_grid_size(grid):
            raise ValueError("BMI x-coordinate destination mismatch")
        np.copyto(x, np.arange(x.size, dtype=x.dtype))

    def get_grid_y(self, grid: int, y: np.ndarray) -> None:
        raise NotImplementedError("One-dimensional index-space grids have no y coordinates")

    def get_grid_z(self, grid: int, z: np.ndarray) -> None:
        raise NotImplementedError("One-dimensional index-space grids have no z coordinates")

    def get_grid_node_count(self, grid: int) -> int:
        return self.get_grid_size(grid)

    def get_grid_edge_count(self, grid: int) -> int:
        raise NotImplementedError("Index-space arrays do not encode grid edges")

    def get_grid_face_count(self, grid: int) -> int:
        raise NotImplementedError("Index-space arrays do not encode grid faces")

    def get_grid_edge_nodes(self, grid: int, edge_nodes: np.ndarray) -> None:
        raise NotImplementedError("Index-space arrays do not encode grid edges")

    def get_grid_face_nodes(self, grid: int, face_nodes: np.ndarray) -> None:
        raise NotImplementedError("Index-space arrays do not encode grid faces")

    def get_grid_face_edges(self, grid: int, face_edges: np.ndarray) -> None:
        raise NotImplementedError("Index-space arrays do not encode grid face edges")

    def get_grid_nodes_per_face(self, grid: int, nodes_per_face: np.ndarray) -> None:
        raise NotImplementedError("Index-space arrays do not encode grid faces")

    def finalize(self) -> None:
        self._initialized = False
        self._values.clear()
