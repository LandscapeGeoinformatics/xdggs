from collections.abc import Mapping
from typing import Any

import numpy as np
import xarray as xr
from xarray.indexes import PandasIndex

from xdggs.index import DGGSIndex
from xdggs.utils import _extract_cell_id_variable, register_dggs
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from xarray.core.types import ErrorOptions, JoinOptions, Self

from tqdm.auto import tqdm
from dggrid4py import DGGRIDv7
import geopandas as gpd
from shapely.ops import transform
from shapely.geometry import shape, box
from numpy import cos, sin, power, deg2rad, arcsin, sqrt, pi
import tempfile
import importlib
import os
import time
import pandas as pd

from pyproj import Transformer

isea_type = ['isea4t', 'isea4d', 'isea4h', 'isea3h', 'isea7h', 'isea43h']


@register_dggs("isea")
@register_dggs("ISEA")
class ISEAIndex(DGGSIndex):
    _resolution: int

    def __init__(
        self,
        cell_ids: Any | PandasIndex,
        dim: str,
        resolution: int,
        grid: str,
        aperture: int,
        topology: str,
        mp: int,
        step: int,
        epsg: str
    ):
        super().__init__(cell_ids, dim)
        self._resolution = int(resolution) if (type(resolution) == int) else resolution
        self._dggs_type = grid
        self._aperture = aperture
        self._topology = topology
        self._dggrid_instance = self._get_dggrid_instance()
        self._mp = mp
        self._step = step
        self._epsg = epsg
        #print(f'{selef._dggs_type} unique idx count : {len(.unique(cellids,axis=-1))}')

    @classmethod
    def from_variables(
        cls: type["ISEAIndex"],
        variables: Mapping[Any, xr.Variable],
        *,
        options: Mapping[str, Any],
    ) -> "ISEAIndex":
        #name, var, _ = _extract_cell_id_variable(variables)

        # Data Preprocessing
        keys = list(variables.keys())
        # For using set_xindex
        if (len(keys) == 1):
            data = variables[keys[0]].data
            # if the index data is already in int64 , assume it is already a dggs cell id
            if (type(data) == np.ndarray and data.dtype == 'int64'):
                # Assume to be already in dggs ( ndarray and dtype is int64 )
                attrs = variables[keys[0]].attrs
                grid = 'isea' + str(attrs['aperture']) + attrs['topology']
                return cls(data, keys[0], attrs['resolution'], grid.upper(), attrs['aperture'], attrs['topology'], attrs['mp'], attrs['trunk'], attrs['epsg'])
            else:
                name, var, _ = _extract_cell_id_variable(variables)
        # For using stack
        elif ( len(keys) == 2):
            if('lat' not in keys or 'lon' not in keys ):
                print('Either variable lat or lon is not found')
                raise ValueError
            attrs = variables[keys[0]].attrs
            start = time.time()
            lat , lon = xr.broadcast(xr.DataArray(variables['lat'].data, dims='lat'),xr.DataArray(variables['lon'].data, dims='lon'))
            lon = np.stack([lat,lon],axis=-1)
            lon = lon.reshape(-1,2)
            del(lat)
            del(variables)
            end = time.time()
            print(f'Broadcast Array Compeleted : {end-start}')
            #lon = lon.view(dtype=np.dtype([('x', 'f8'), ('y', 'f8')]))
            #lon = lon.reshape(lon.shape[:-1])
            var = xr.Variable(['lat','lon'],lon,attrs=attrs)
            name = "cell_ids"
        else:
            raise ValueError

        resolution = var.attrs.get("resolution", options.get("resolution", 9))
        aperture = var.attrs.get("aperture", options.get("aperture", 7))
        topology = var.attrs.get("topology", options.get("topology", 'h')).lower()
        src_epsg = var.attrs.get("epsg",options.get("epsg","4326"))
        mp = var.attrs.get("mp",options.get('mp',1))
        grid = 'isea' + str(aperture) + topology
        dims = list(var.dims)
        lat, lon = None, None
        # we assume that the data is Nx2
        data = var.data
        step = var.attrs.get("trunk",options.get('trunk',250000)) if (mp>1) else data.shape[0]
        print(f'Data type : {data.dtype} , shape : {data.shape}')
        if (grid not in isea_type):
            print('ISEA type error')
            raise ValueError
        try:
            lat = dims.index('lat')
            lon = dims.index('lon')
        except ValueError:
            print('lat or lon not found in dim')
            raise ValueError
        print(f'Create index from lat,lon with dim index : lat:{lat} lon:{lon}')
        executable = os.environ['DGGRID_PATH']
        working_dir = tempfile.mkdtemp()
        if (data.shape[-1] != 2):
            print('Dim Error')
            raise ValueError
        dggs = DGGRIDv7(executable=executable, working_dir=working_dir, capture_logs=True, silent=True)
        cellids = None
        batch = int(np.ceil(data.shape[0]/step))
        #batch = batch + 1 if (mp>1) else batch
        if (resolution == -1):
            # auto resolution
            maxlat = np.max(data[:,lat])
            minlat = np.min(data[:,lat])
            maxlon = np.max(data[:,lon])
            minlon = np.min(data[:,lon])
            df = gpd.GeoDataFrame([0], geometry=[box(minlon, minlat, maxlon, maxlon)],crs=f'EPSG:{src_epsg}')
            print(f'Total Bounds ({src_epsg}): {df.total_bounds}')
            if (src_epsg != 4326):
                df = df.to_crs('wgs84')# if (src_epsg != 4326) else df
                print(f'Total Bounds (wgs84): {df.total_bounds}')
            R=6378
            lon1, lat1, lon2, lat2 = df.total_bounds
            lon1=deg2rad(lon1)
            lon2=deg2rad(lon2)
            lat1=deg2rad(lat1)
            lat2=deg2rad(lat2)
            a = ( sin((lon2-lon1) / 2) ** 2 + cos(lon1) * cos(lon2) * sin(0) ** 2 ) #
            d = 2 * arcsin(sqrt(a))
            normalize = 4*pi if (src_epsg == 3301) else 1
            area = abs(d*( (power(R,2)*sin(lat2)) - (power(R,2)*sin(lat1)) )) / normalize # symmetric of polar coords
            print(f'Total Bounds Area (km^2): {area}')
            avg_area_per_data = (area / data.shape[0])
            print(f'Area per center point (km^2): {avg_area_per_data}')
            dggrid_resolution = dggs.grid_stats_table(grid.upper(), 30)
            filter_ = dggrid_resolution[dggrid_resolution['Area (km^2)'] < avg_area_per_data]
            resolution = 9
            if (len(filter_) > 0):
                resolution = filter_.iloc[0, 0]
                print(f'Auto resolution : {resolution}')
        if (importlib.util.find_spec('pymp') is None):
            print(f"pymp not found, running on single core")
            df=gpd.GeoDataFrame([0]*data.shape[0],geometry=gpd.points_from_xy(data[:, lon], data[:, lat]), crs=f'EPSG:{src_epsg}')
            df = df.to_crs('EPSG:4326') if (src_epsg != 4326) else df
            result = dggs.cells_for_geo_points(df, True, grid.upper(), resolution)
            cellids = result['seqnums'].values.astype('int64')
        else:
            import pymp
            cellids = pymp.shared.array((data.shape[0]), dtype='int64')
            print(f"Batch Size: {batch}")
            with pymp.Parallel(mp) as p:
                for i in tqdm(p.range(batch)):
                    end = (i*step)+step if (((i*step)+step) < data.shape[0]) else data.shape[0]
                    trunk = data[(i*step):end]
                    df=gpd.GeoDataFrame([0]*trunk.shape[0],geometry=gpd.points_from_xy(trunk[:, lon], trunk[:, lat]), crs=f'EPSG:{src_epsg}')
                    df = df.to_crs('EPSG:4326') if (src_epsg != 4326) else df
                    result = dggs.cells_for_geo_points(df, True, grid.upper(), resolution)
                    cellids[(i*step):(i*step)+result['seqnums'].values.shape[0]] = result['seqnums'].values.astype('int64')
        print('Cell ID calcultion completed')
        return cls(cellids, name, resolution, grid.upper(), aperture, topology, mp, step, f'EPSG:{src_epsg}')

    def concat(self, indexes: Sequence[Self], dim: Hashable, positions: Iterable[Iterable[int]] | None = None) -> Self:
        idx = []

        attrs = { 'resolution': indexes[0]._resolution ,
                  'aperture':  indexes[0]._aperture ,
                  'topology': indexes[0]._topology  ,
                  'epsg': indexes[0]._epsg,
                  'mp': indexes[0]._mp,
                  'trunk': indexes[0]._step
                }
        pd_indexes = [idx._pd_index.index for idx in indexes]
        pd_indexes = pd_indexes[0].append(pd_indexes[1:])

        return ISEAIndex.from_variables({dim:xr.Variable(dim,pd_indexes.values,attrs)},options={})

    def create_variables(self, variables):
        var = list(variables.values())[0]
        var = xr.Variable(self._dim, self._pd_index.index, var.attrs)
        idx_variables = {self._dim: var}
        return idx_variables

    def _latlon2cellid(self, lat: Any, lon: Any) -> np.ndarray:
        lat = xr.DataArray(lat, dims='lat')
        lon = xr.DataArray(lon, dims='lon')
        lat, lon = xr.broadcast(lat, lon)
        lat = np.stack([lon, lat], axis=-1).reshape(-1, 2)
        print(lat)
        del(lon)
        df = gpd.GeoDataFrame({'lon': lat[:, 0], 'lat': lat[:, 1]}, geometry=gpd.points_from_xy(lat[:, 0], lat[:, 1]))
        cellids = self._dggrid_instance.cells_for_geo_points(df, True, self._dggs_type.upper(), self._resolution)
        return cellids['seqnums'].to_numpy(dtype='int64')

    def _cellid2latlon(self, cell_ids: Any) -> tuple[np.ndarray, np.ndarray]:
        mp = False if (importlib.util.find_spec('pymp') is None) else True
        data = cell_ids
        if (mp):
            import pymp
            cellids = pymp.shared.array((data.shape[0]), dtype='int64')
            mp = self._mp
            step = self._step
            batch = int(np.ceil(data.shape[0]/step))
            points = pymp.shared.array([data.shape[0],2])
            with pymp.Parallel(mp) as p:
                for i in tqdm(p.range(batch)):
                    end = (i*step)+step if (((i*step)+step) < data.shape[0]) else data.shape[0]
                    trunk = data[(i*step):end]
                    df = self._dggrid_instance.grid_centerpoint_from_cellids(trunk, self._dggs_type, self._resolution)
                    ps = df['geometry'].values
                    ps = np.array([[p.y, p.x] for p in ps])
                    points[(i*step):end] = ps
        else:
            df = self._dggrid_instance.grid_centerpoint_from_cellids(self._pd_index.index, self._dggs_type, self._resolution)
            points = df['geometry'].values
            points = np.array([[p.y, p.x] for p in points])

        return points[:, 1], points[:, 0]

    def _repr_inline_(self, max_width: int):
        return f"ISEAIndex(dgg_type={self._dggs_type}, resolution={self._resolution})"

    def sel(self, labels, method=None, tolerance=None):
        if method == "nearest":
            raise ValueError("finding nearest grid cell has no meaning")
        target = np.unique(list(labels.values())[0])
        key = list(labels.keys())[0]
        labels[key] = np.isin(self._pd_index.index.values, target)
        return self._pd_index.sel(labels, method=method, tolerance=tolerance)

    def _geometry(self, cell_ids=None):
        mp = False if (importlib.util.find_spec('pymp') is None) else True
        data = cell_ids if (cell_ids is not None) else self._pd_index.index
        if (mp):
            import pymp
            cellids = pymp.shared.array((data.shape[0]), dtype='int64')
            mp = self._mp
            step = self._step
            batch = int(np.ceil(data.shape[0]/step))
            geometryDF = pymp.shared.list([])
            with pymp.Parallel(mp) as p:
                for i in tqdm(p.range(batch)):
                    end = (i*step)+step if (((i*step)+step) < data.shape[0]) else data.shape[0]
                    trunk = data[(i*step):end]
                    df = self._dggrid_instance.grid_cell_polygons_from_cellids(trunk, self._dggs_type, self._resolution)
                    geometryDF.append(df)
            geometryDF = gpd.GeoDataFrame(pd.concat( geometryDF, ignore_index=True))
        else:
            geometryDF = self._dggrid_instance.grid_cell_polygons_from_cellids(data, self._dggs_type, self._resolution)
        return geometryDF

    def polygon_for_extent(self, geoobj, src_epsg):
        transformer = Transformer.from_crs(f"EPSG:{src_epsg}", "EPSG:4326").transform
        try:
            geoobj = shape(geoobj)
        except Exception as e:
            print(f'Invalid Extend : {e}')
        geoobj = transform(transformer, geoobj)
        df = self._dggrid_instance.grid_cellids_for_extent(self._dggs_type, self._resolution, clip_geom=geoobj)
        return df

    def _replace(self, new_pd_index: PandasIndex):
        return type(self)(new_pd_index, self._dim, self._resolution, self._dggs_type, self._aperture, self._topology, self._mp, self._step, self._epsg)

    def to_pandas_index(self):
        return self._pd_index.index

    def _get_dggrid_instance(self):
        executable = os.environ['DGGRID_PATH']
        working_dir = tempfile.mkdtemp()
        return DGGRIDv7(executable=executable, working_dir=working_dir, capture_logs=True, silent=True)

