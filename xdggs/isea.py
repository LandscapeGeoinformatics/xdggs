from collections.abc import Mapping
from typing import Any

import numpy as np
import xarray as xr
from xarray.indexes import PandasIndex

from xdggs.index import DGGSIndex
from xdggs.utils import _extract_cell_id_variable, register_dggs

from dggrid4py import DGGRIDv7
import geopandas as gpd
from itertools import product
import tempfile
import os

isea_type = ['isea4t', 'isea4d', 'isea4h', 'isea3h', 'isea7h', 'isea43h']


@register_dggs("isea")
class ISEAIndex(DGGSIndex):
    _resolution: int

    def __init__(
        self,
        cell_ids: Any | PandasIndex,
        dim: str,
        resolution: int,
        grid: str,
        aperture: int,
        topology: str
    ):
        super().__init__(cell_ids, dim)
        self._resolution = int(resolution) if (int(resolution) < 9 and int(resolution) > 0) else 9
        self._dggs_type = grid
        self._aperture = aperture
        self._topology = topology
        self._dggrid_instance = self._get_dggrid_instance()

    @classmethod
    def from_variables(
        cls: type["ISEAIndex"],
        variables: Mapping[Any, xr.Variable],
        *,
        options: Mapping[str, Any],
    ) -> "ISEAIndex":
        name, var, dim = _extract_cell_id_variable(variables)
        if (var.data.shape[-1] != 2):
            raise ValueError
        resolution = var.attrs.get("resolution", options.get("resolution", 9))
        aperture = var.attrs.get("aperture", options.get("aperture", 7))
        topology = var.attrs.get("topology", options.get("topology", 'h')).lower()
        lat = var.data[:, 0]
        lon = var.data[:, 1]
        grid = 'isea' + str(aperture) + topology
        if (grid not in isea_type):
            print('ISEA type error')
            raise ValueError()
        executable = os.environ['DGGRID_PATH']
        working_dir = tempfile.mkdtemp()
        dggs = DGGRIDv7(executable=executable, working_dir=working_dir, capture_logs=True, silent=True)
        df = gpd.GeoDataFrame(var.data, geometry=gpd.points_from_xy(lon, lat))
        cellids = dggs.cells_for_geo_points(df, True, grid.upper(), resolution)
        cellids = cellids['seqnums'].values.astype('int64')
        print(f'{grid} unique idx count : {len(np.unique(cellids,axis=-1))}')
        return cls(cellids, name, resolution, grid, aperture, topology)

    def _latlon2cellid(self, lat: Any, lon: Any) -> np.ndarray:
        lat = np.array(lat)
        lon = np.array(lon)
        lat = lat.reshape(-1) if (lat.shape == ()) else lat
        lon = lon.reshape(-1) if (lon.shape == ()) else lon
        pairs = np.array(list(product(lon, lat)))
        df = gpd.GeoDataFrame(pairs, geometry=gpd.points_from_xy(pairs[:, 0], pairs[:, 1]))
        cellids = self._dggrid_instance.cells_for_geo_points(df, True, self._dggs_type.upper(), self._resolution)
        return cellids['seqnums'].to_numpy(dtype='int64')

    def _cellid2latlon(self, cell_ids: Any) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

    def _repr_inline_(self, max_width: int):
        return f"ISEAIndex(dgg_type={self._dggs_type}, resolution={self._resolution})"

    def sel(self, labels, method=None, tolerance=None):
        print(f'cells id: {labels} {type(labels)}')
        if method == "nearest":
            raise ValueError("finding nearest grid cell has no meaning")
        target = list(labels.values())[0]
        key = list(labels.keys())[0]
        bidx = np.zeros((len(target), self._pd_index.index.shape[0]))
        for i, t in enumerate(target):
            bidx[i, np.where(self._pd_index.index == t)[0]] = 1
        bidx = np.sum(bidx, axis=0)
        labels[key] = np.where(bidx > 0, True, False)
        return self._pd_index.sel(labels, method=method, tolerance=tolerance)

    def _replace(self, new_pd_index: PandasIndex):
        return type(self)(new_pd_index, self._dim, self._resolution, self._dggs_type, self._aperture, self._topology)

    def _get_dggrid_instance(self):
        executable = os.environ['DGGRID_PATH']
        working_dir = tempfile.mkdtemp()
        return DGGRIDv7(executable=executable, working_dir=working_dir, capture_logs=True, silent=True)

