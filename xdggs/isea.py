from collections.abc import Mapping
from typing import Any

import numpy as np
import xarray as xr
from xarray.indexes import PandasIndex

from xdggs.index import DGGSIndex
from xdggs.utils import _extract_cell_id_variable, register_dggs

from tqdm.auto import tqdm
from dggrid4py import DGGRIDv7
import geopandas as gpd
from itertools import product
from shapely.geometry import box
import tempfile
import pymp
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
        self._resolution = int(resolution) if (type(resolution) == int) else resolution
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
        name, var, _ = _extract_cell_id_variable(variables)
        resolution = var.attrs.get("resolution", options.get("resolution", 9))
        aperture = var.attrs.get("aperture", options.get("aperture", 7))
        topology = var.attrs.get("topology", options.get("topology", 'h')).lower()
        src_epsg = var.attrs.get("epsg",options.get("epsg","4326"))
        mp = var.attrs.get("mp",options.get('mp',1))
        step = var.attrs.get("trunk",options.get('trunk',250000))
        grid = 'isea' + str(aperture) + topology
        dims = list(var.dims)
        lat, lon = None, None
        # we assume that the data is Nx2
        data =  var.data
        if (type(data) == np.ndarray and data.dtype == 'int64'):
            # Assume to be already in dggs ( ndarray and dtype is int64 )
            return cls(data, name, resolution, grid.upper(), aperture, topology)
        if (grid not in isea_type):
            print('ISEA type error')
            raise ValueError
        try:
            lat = dims.index('lat')
            lon = dims.index('lon')
        except ValueError:
            print('lat or lon not found in dim')
            raise ValueError
        print('Create index from lat,lon')
        executable = os.environ['DGGRID_PATH']
        working_dir = tempfile.mkdtemp()
        if (type(data[0]) == tuple and len(data[0]) == 2):
            data = np.array([[i[lat], i[lon]] for i in data])
        elif (len(data.shape) == 2):
            if(data.dtype == [('x', '<f8'), ('y', '<f8')]):
                data = data.reshape(-1)
                data = data.view(dtype=np.dtype([('x', 'float64')]))
                data = data.reshape(-1,2)
                print(data.shape)
        elif (data.shape[-1] != 2):
            print('Dim Error')
            raise ValueError
        dggs = DGGRIDv7(executable=executable, working_dir=working_dir, capture_logs=True, silent=True)
        cellids = np.full((data.shape[0], ),-1,dtype='int64')
        with pymp.Parallel(mp) as p:
            for i in tqdm(p.range(0,data.shape[0],step)):
                trunk = data[i:i+step]
                df=gpd.GeoDataFrame({'lon':trunk[:,lon],'lat':trunk[:,lat]},geometry=gpd.points_from_xy(trunk[:, lon], trunk[:, lat]), crs=f'EPSG:{src_epsg}')
                if (type(resolution) ==str and resolution.lower() == 'auto'):
                    print(f'Total Bounds: {df.total_bounds}')
                    df2 = df.to_crs(3857)
                    area = box(*df2.total_bounds).area
                    print(f'Total Bounds Area (m^2): {area}')
                    avg_area_per_data = (area / (1000 * 1000)) / len(df)
                    print(f'Area per center point (km^2): {avg_area_per_data}')
                    dggrid_resolution = dggs.grid_stats_table(grid.upper(), 30)
                    filter_ = dggrid_resolution[dggrid_resolution['Area (km^2)'] < avg_area_per_data]
                    resolution = 9
                    if (len(filter_) > 0):
                        resolution = filter_.iloc[0, 0]
                        print(f'Auto resolution : {resolution}')
                result = dggs.cells_for_geo_points(df, True, grid.upper(), resolution)
                cellids[i:i+result['seqnums'].values.shape[0]] = result['seqnums'].values.astype('int64')
        i = np.where(cellids == -1)[0][0]
        print(f'Remain {i}')
        trunk = data[i:]
        df=gpd.GeoDataFrame({'lon':trunk[:,lon],'lat':trunk[:,lat]},geometry=gpd.points_from_xy(trunk[:, lon], trunk[:, lat]), crs=f'EPSG:{src_epsg}')
        result = dggs.cells_for_geo_points(df, True, grid.upper(), resolution)
        cellids[i:i+result['seqnums'].values.shape[0]] = result['seqnums'].values.astype('int64')
        print(f'{grid} unique idx count : {len(np.unique(cellids,axis=-1))}')
        return cls(cellids, name, resolution, grid.upper(), aperture, topology)

    def create_variables(self, variables):
        key = list(variables.keys())[0]
        var = list(variables.values())[0]
        var = xr.Variable(self._dim, self._pd_index.index, var.attrs)
        idx_variables = {key: var}
        return idx_variables

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

    def to_pandas_index(self):
        return type(self)(self._pd_index, self._dim, self._resolution, self._dggs_type, self._aperture, self._topology)

    def _get_dggrid_instance(self):
        executable = os.environ['DGGRID_PATH']
        working_dir = tempfile.mkdtemp()
        return DGGRIDv7(executable=executable, working_dir=working_dir, capture_logs=True, silent=True)

