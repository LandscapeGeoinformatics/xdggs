from collections.abc import Hashable, Mapping
from typing import Any, Union

import numpy as np
import xarray as xr
from xarray.indexes import Index, PandasIndex
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from xarray.core.types import ErrorOptions, JoinOptions, Self

from xdggs.grid import DGGSInfo
from xdggs.utils import GRID_REGISTRY, _extract_cell_id_variable


def decode(ds):
    variable_name = "cell_ids"
    if (ds[variable_name].attrs.get('grid_name', None) == 'isea'):
        # ISEA Grid Handling. The cell_ids index is created by stacking x,y coordinate.
        data = ds[variable_name]
        if (type(data) is PandasIndex):
            # Either it is already converted (in PandasIndexI)
            return ds.drop_indexes(variable_name, errors="ignore").set_xindex(
                variable_name, DGGSIndex)
        if (ds[variable_name].attrs.get('coordinate') is None):
            raise ValueError("ISEA DGGSInfo must consist of coordinate attribute")
        coords = ds[variable_name].attrs['coordinate']
        attrs = ds[variable_name].attrs
        if (len(coords) != 2):
            raise ValueError("ISEA DGGSInfo must consist of coordinate of size 2 [x, y]")
        ds[coords[0]].attrs = attrs
        ds[coords[1]].attrs = attrs
        ds.stack(variable_name=[coords[0], coords[1]], index_cls=DGGSIndex)
    else:
        return ds.drop_indexes(variable_name, errors="ignore").set_xindex(
            variable_name, DGGSIndex
        )


class DGGSIndex(Index):
    _dim: str
    _pd_index: PandasIndex

    def __init__(self, cell_ids: Any | PandasIndex, dim: str, grid_info: DGGSInfo):
        self._dim = dim

        if isinstance(cell_ids, PandasIndex):
            self._pd_index = cell_ids
        else:
            self._pd_index = PandasIndex(cell_ids, dim)

        self._grid = grid_info

    @classmethod
    def from_variables(
        cls: type["DGGSIndex"],
        variables: Mapping[Any, xr.Variable],
        *,
        options: Mapping[str, Any],
    ) -> "DGGSIndex":
        _, var, _ = _extract_cell_id_variable(variables)
        grid_name = var.attrs["grid_name"]
        cls = GRID_REGISTRY.get(grid_name)
        if cls is None:
            raise ValueError(f"unknown DGGS grid name: {grid_name}")

        return cls.from_variables(variables, options=options)

    @classmethod
    def stack(cls, variables: Mapping[Any, xr.Variable], dim: Hashable):
        return cls.from_variables(variables, options={})

    def concat(self, indexes: Sequence[Self], dim: Hashable, positions: Iterable[Iterable[int]] | None = None, ) -> Self:
        return self.concat(cls, indexes, dim, positions)


    def create_variables(
        self, variables: Mapping[Any, xr.Variable] | None = None
    ) -> dict[Hashable, xr.Variable]:
        return self._pd_index.create_variables(variables)

    def isel(
        self: "DGGSIndex", indexers: Mapping[Any, int | np.ndarray | xr.Variable]
    ) -> Union["DGGSIndex", None]:
        new_pd_index = self._pd_index.isel(indexers)
        if new_pd_index is not None:
            return self._replace(new_pd_index)
        else:
            return None

    def sel(self, labels, method=None, tolerance=None):
        if method == "nearest":
            raise ValueError("finding nearest grid cell has no meaning")
        return self._pd_index.sel(labels, method=method, tolerance=tolerance)

    def _replace(self, new_pd_index: PandasIndex):
        raise NotImplementedError()

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        return self._grid.cell_ids2geographic(self._pd_index.index.values)

    def _geometry(self, extent=None):
        raise NotImplementedError()

    @property
    def grid_info(self) -> DGGSInfo:
        return self._grid
