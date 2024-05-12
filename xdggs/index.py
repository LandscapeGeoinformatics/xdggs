from collections.abc import Hashable, Mapping
from typing import Any, Union

import numpy as np
import xarray as xr
from xarray.indexes import Index, PandasIndex
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from xarray.core.types import ErrorOptions, JoinOptions, Self
from xdggs.utils import GRID_REGISTRY, _extract_cell_id_variable


def decode(ds):
    variable_name = "cell_ids"

    return ds.drop_indexes(variable_name, errors="ignore").set_xindex(
        variable_name, DGGSIndex
    )


class DGGSIndex(Index):
    _dim: str
    _pd_index: PandasIndex

    def __init__(self, cell_ids: Any | PandasIndex, dim: str):
        self._dim = dim

        if isinstance(cell_ids, PandasIndex):
            self._pd_index = cell_ids
        else:
            self._pd_index = PandasIndex(cell_ids, dim)

    @classmethod
    def from_variables(
        cls: type["DGGSIndex"],
        variables: Mapping[Any, xr.Variable],
        *,
        options: Mapping[str, Any],
    ) -> "DGGSIndex":
        _, var, _ = _extract_cell_id_variable(variables)

        grid_name = var.attrs["grid_name"]
        cls = GRID_REGISTRY[grid_name]

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

    def _latlon2cellid(self, lat: Any, lon: Any) -> np.ndarray:
        """convert latitude / longitude points to cell ids."""
        raise NotImplementedError()

    def _cellid2latlon(self, cell_ids: Any) -> tuple[np.ndarray, np.ndarray]:
        """convert cell ids to latitude / longitude (cell centers)."""
        raise NotImplementedError()

    def _geometry(self, extent=None):
        raise NotImplementedError()

    @property
    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        return self._cellid2latlon(self._pd_index.index.values)
