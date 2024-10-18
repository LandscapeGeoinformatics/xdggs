from collections.abc import Mapping

import sys
import numpy as np
import xarray as xr
from xarray.indexes import PandasIndex

from xdggs.index import DGGSIndex
from xdggs.grid import DGGSInfo
from xdggs.utils import _extract_cell_id_variable, register_dggs
from typing import Any, ClassVar, Sequence, Hashable, Iterable
from dataclasses import dataclass
try:
    from typing import Self
except ImportError:  # pragma: no cover
    from typing_extensions import Self

from tqdm.auto import tqdm
from dggrid4py import DGGRIDv7
from dggrid4py import dggs_types
import geopandas as gpd
from shapely.ops import transform
from shapely.geometry import shape, box, Point
from pys2index import S2PointIndex
from numpy import cos, sin, power, deg2rad, arcsin, sqrt, pi
import tempfile
import importlib
import os
import time
import pandas as pd
import tempfile
from pyproj import Transformer


@dataclass(frozen=True)
class ISEAInfo(DGGSInfo):
    resolution: int
    aperture: int
    topology: str
    src_epsg: str
    coordinate: list
    method: str
    mp: int
    step: int
    dggs_type: str

    valid_parameters: ClassVar[dict[str, Any]] = {"resolution": range(-1, 15), "aperture": [3, 4, 7],
                                                  "topology": ['H'], "method": ["centerpoint", "nearestpoint"]}

    def __post_init__(self):
        if (self.resolution not in self.valid_parameters['resolution']):
            raise ValueError("resolution must be an integer between 0 and 15")
        if self.aperture not in self.valid_parameters["aperture"]:
            raise ValueError("aperture must be an integer 3,4 or 7")
        if self.topology.upper() not in self.valid_parameters["topology"]:
            raise ValueError("topology must be H")
        if self.method.lower() not in self.valid_parameters["method"]:
            raise ValueError("method {self.method.lower()} is not defined.")

    @classmethod
    def from_dict(cls: type[Self], mapping: dict[str, Any]) -> Self:
        params = {k: v for k, v in mapping.items() if k != 'grid_name'}
        return cls(**params)

    def to_dict(self: Self) -> dict[str, Any]:
        return {"grid_name": "isea", "resolution": self.resolution, "aperture": self.aperture, "topology": self.topology.upper(),
                "src_epsg": self.src_epsg, "coordinate": self.coordinate, 'dggs_type': self.dggs_type, "method": self.method,
                "mp": self.mp, "step": self.step}

    def cell_ids2geographic(
        self, cell_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # lat, lon = cells_to_coordinates(cell_ids, radians=False)
        pass

    def geographic2cell_ids(self, lon, lat):
        pass
        # return coordinates_to_cells(lat, lon, self.resolution, radians=False)


@register_dggs("isea")
class ISEAIndex(DGGSIndex):
    _grid: DGGSInfo

    def __init__(
        self,
        cell_ids: Any | PandasIndex,
        dim: str,
        grid_info: DGGSInfo,
    ):
        if not isinstance(grid_info, ISEAInfo):
            raise ValueError(f"grid info object has an invalid type: {type(grid_info)}")
        super().__init__(cell_ids, dim, grid_info)

    @classmethod
    def from_variables(
        cls: type["ISEAIndex"],
        variables: Mapping[Any, xr.Variable],
        *,
        options: Mapping[str, Any],
    ) -> "ISEAIndex":
        # name, var, _ = _extract_cell_id_variable(variables)
        isea = [k for k in variables.keys() if (variables[k].attrs.get('grid_name') == 'isea')]
        var = variables[isea[0]]
        # For using stack assume the coordinate is in x, y ordering
        # prepare to generate hexagon grid
        resolution = var.attrs.get("resolution", options.get("resolution", -1))
        coords = var.attrs.get('coordinate', options.get('coordinate'))
        aperture = var.attrs.get("aperture", options.get("aperture", 7))
        topology = var.attrs.get("topology", options.get("topology", 'h')).lower()
        src_epsg = var.attrs.get("src_epsg", options.get("src_epsg", "wgs84"))
        method = var.attrs.get("method", options.get("method", "nearestpoint"))
        mp = var.attrs.get("mp", options.get('mp', 1))
        step = var.attrs.get("trunk", options.get('trunk', 250000)) if (mp > 1) else var.data.shape[0]
        grid = 'isea' + str(aperture) + topology
        x = variables[coords[0]].data
        y = variables[coords[1]].data
        executable = os.environ['DGGRID_PATH']
        reproject = Transformer.from_crs(src_epsg, 'wgs84', always_xy=True)
        print(f'x shape: ({x.shape}), y shape: ({y.shape})')
        if (grid.upper() not in dggs_types):
            raise ValueError(f"{grid} is not defined in DGGRID")
        cellids = None
        batch = int(np.ceil(x.shape[0] / step))
        y_batch = int(np.ceil(y.shape[0] / step))
        y_offset = step * step
        x_mp = int(np.ceil(mp / 2))
        y_mp = (mp - x_mp) if (mp - x_mp > 0) else 1
        # Auto Resolution
        if (resolution == -1):
            maxlat, minlat, maxlng, minlng = np.max(y), np.min(y), np.max(x), np.min(x)
            working_dir = tempfile.mkdtemp()
            dggs = DGGRIDv7(executable=executable, working_dir=working_dir, capture_logs=True, silent=True)
            resolution = cls._autoResolution(minlng, minlat, maxlng, maxlat, src_epsg, (x.shape[0] * y.shape[0]), dggs, grid)
        # Generate Cells ID
        with tempfile.NamedTemporaryFile() as ntf:
            cellids = np.memmap(ntf, mode='w+', shape=(x.shape[0] * y.shape[0]), dtype=np.int64)
        if (method.lower() == 'nearestpoint'):
            print(f"---Generate Cell ID with resolution -1 by nearestpoint method, x batch:{batch}, y batch:{y_batch}---")
            import pymp
            start = time.time()
            pymp.config.nested = True
            print(f"--- Multiprocessing x:{x_mp}, y:{y_mp} ---")
            with pymp.Parallel(x_mp) as p:
                for i in tqdm(p.range(batch)):
                    end = (i * step) + step if (((i * step) + step) < x.shape[0]) else x.shape[0]
                    offset = i * (y.shape[0] * step)
                    with pymp.Parallel(y_mp) as q:
                        for j in q.range(y_batch):
                            working_dir = tempfile.mkdtemp()
                            dggs = DGGRIDv7(executable=executable, working_dir=working_dir, capture_logs=True, silent=True, tmp_geo_out_legacy=False)
                            y_end = (j * step) + step if (((j * step) + step) < y.shape[0]) else y.shape[0]
                            x_trunk, y_trunk = np.broadcast_arrays(x[(i * step):end],  y[(j * step):y_end, None])
                            x_trunk = np.stack([x_trunk, y_trunk], axis=-1)
                            del y_trunk
                            x_trunk = x_trunk.reshape(-1, 2)
                            x_trunk[:, 0], x_trunk[:, 1] = reproject.transform(x_trunk[:, 0], x_trunk[:, 1])
                            maxlat, minlat, maxlng, minlng = np.max(x_trunk[:, 1]), np.min(x_trunk[:, 1]), np.max(x_trunk[:, 0]), np.min(x_trunk[:, 0])
                            truck_df = gpd.GeoDataFrame([0], geometry=[box(minlng, minlat, maxlng, maxlat)], crs='wgs84')
                            result = dggs.grid_centerpoint_for_extent(grid.upper(), resolution, clip_geom=truck_df.geometry.values[0])
                            centroids = result.geometry.get_coordinates()
                            centroids = np.stack([centroids['x'].values, centroids['y'].values], axis=-1)
                            centroids_idx = S2PointIndex(centroids)
                            distance, idx = centroids_idx.query(x_trunk)
                            cells = result.iloc[idx]['name'].astype('int64')
                            # handling for x-axis boundary case
                            yoff = y_offset if (i != (batch - 1)) else x_trunk.shape[0]
                            # handling for x and y axis boundary case (top right corner)
                            yoff = (x[(i * step):end].shape[0] * step) if ((i == (batch - 1)) and (j == (y_batch - 1))) else yoff
                            cellids[offset + (j * yoff):((offset + (j * yoff)) + x_trunk.shape[0])] = cells
                            y_offset = x_trunk.shape[0]
                            cellids.flush()
            print(f'cell generation time: ({time.time()-start})')
        elif (method.lower() == 'centerpoint'):
            print(f"---Generate Cell ID with resolution -1 by centerpoint method x batch:{batch}, y batch:{y_batch}---")
            if (importlib.util.find_spec('pymp') is None):
                print("pymp not found, running on single core")
                # Center Point Method
                working_dir = tempfile.mkdtemp()
                dggs = DGGRIDv7(executable=executable, working_dir=working_dir, capture_logs=True, silent=True)
                data, y_trunk = np.broadcast_arrays(x, y[:, None])
                data = np.stack([data, y_trunk], axis=-1)
                data[:, 0], data[:, 1] = reproject.transform(data[:, 0], data[:, 1])
                del (y_trunk)
                df = gpd.GeoDataFrame([0] * data.shape[0], geometry=gpd.points_from_xy(data[:, 0], data[:, 1]), crs=src_epsg)
                result = dggs.cells_for_geo_points(df, True, grid.upper(), resolution)
                cellids = result['seqnums'].values
                cellids = cellids.astype('int64')
            else:
                import pymp
                pymp.config.nested = True
                print(f"--- Multiprocessing x:{x_mp}, y:{y_mp} ---")
                with pymp.Parallel(x_mp) as p:
                    for i in tqdm(p.range(batch)):
                        end = (i * step) + step if (((i * step) + step) < x.shape[0]) else x.shape[0]
                        offset = i * (y.shape[0] * step)
                        with pymp.Parallel(y_mp) as q:
                            for j in q.range(y_batch):
                                working_dir = tempfile.mkdtemp()
                                dggs = DGGRIDv7(executable=executable, working_dir=working_dir, capture_logs=True, silent=True)
                                y_end = (j * step) + step if (((j * step) + step) < y.shape[0]) else y.shape[0]
                                x_trunk, y_trunk = np.broadcast_arrays(x[(i * step):end], y[(j * step):y_end, None])
                                x_trunk = np.stack([x_trunk, y_trunk], axis=-1)
                                del y_trunk
                                x_trunk = x_trunk.reshape(-1, 2)
                                x_trunk[:, 0], x_trunk[:, 1] = reproject.transform(x_trunk[:, 0], x_trunk[:, 1])
                                df = gpd.GeoDataFrame([0] * x_trunk.shape[0], geometry=gpd.points_from_xy(x_trunk[:, 0], x_trunk[:, 1]), crs=src_epsg)
                                result = dggs.cells_for_geo_points(df, True, grid.upper(), resolution)
                                # handling for x-axis boundary case
                                yoff = y_offset if (i != (batch - 1)) else x_trunk.shape[0]
                                # handling for x and y axis boundary case (top right corner)
                                yoff = (x[(i * step):end].shape[0] * step) if ((i == (batch - 1)) and (j == (y_batch - 1))) else yoff
                                cellids[offset + (j * yoff): (offset + (j * yoff)) + x_trunk.shape[0]] = result['seqnums'].values.astype('int64')
        print(f'Cell ID calcultion completed, unique cell id :{np.unique(cellids).shape[0]}')
        arrts = {'resolution': resolution, 'aperture': aperture, 'topology': topology,
                 'src_epsg': src_epsg, 'coordinate': coords, 'method': method, 'mp': mp, 'step': step,
                 'dggs_type': f'ISEA{aperture}{topology.upper()}'}
        grid_info = ISEAInfo.from_dict(arrts | options)
        return cls(cellids, 'cell_ids', grid_info)

    def concat(self, indexes: Sequence[Self], dim: Hashable, positions: Iterable[Iterable[int]] | None = None) -> Self:
        attrs = indexes[0]._grid.to_dict()
        pd_indexes = [idx._pd_index.index for idx in indexes]
        pd_indexes = pd_indexes[0].append(pd_indexes[1:])
        return ISEAIndex.from_variables({dim:xr.Variable(dim,pd_indexes.values,attrs)},options={})

    def create_variables(self, variables):
        var = list(variables.values())[0]
        var = xr.Variable(self._dim, self._pd_index.index, var.attrs)
        idx_variables = {self._dim: var}
        return idx_variables

    def _latlng2cellid(self, lat: Any, lng: Any) -> np.ndarray:
        lat = xr.DataArray(lat, dims='lat')
        lng = xr.DataArray(lng, dims='lon')
        lat, lng = xr.broadcast(lat, lng)
        lat = np.stack([lng, lat], axis=-1).reshape(-1, 2)
        del lng
        df = gpd.GeoDataFrame({'lng': lat[:, 0], 'lat': lat[:, 1]}, geometry=gpd.points_from_xy(lat[:, 0], lat[:, 1]))
        cellids = self._dggrid_instance.cells_for_geo_points(df, True, self._grid.dggs_type.upper(), self._grid.resolution)
        return cellids['seqnums'].to_numpy(dtype='int64')

    def _cellid2latlng(self, cell_ids: Any) -> tuple[np.ndarray, np.ndarray]:
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
                    df = self._dggrid_instance.grid_centerpoint_from_cellids(trunk, self._grid.dggs_type, self._grid.resolution)
                    ps = df['geometry'].values
                    ps = np.array([[p.y, p.x] for p in ps])
                    points[(i*step):end] = ps
        else:
            df = self._dggrid_instance.grid_centerpoint_from_cellids(self._pd_index.index, self._grid.dggs_type, self._grid.resolution)
            points = df['geometry'].values
            points = np.array([[p.y, p.x] for p in points])

        return points[:, 1], points[:, 0]

    def _repr_inline_(self, max_width: int):
        return f"ISEAIndex(dgg_type={self._grid.dggs_type}, resolution={self._grid.resolution})"

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
        data = data[np.where(data>0)]
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
                    df = self._dggrid_instance.grid_cell_polygons_from_cellids(trunk, self._grid.dggs_type, self._grid.resolution)
                    geometryDF.append(df)
            geometryDF = gpd.GeoDataFrame(pd.concat( geometryDF, ignore_index=True))
        else:
            geometryDF = self._dggrid_instance.grid_cell_polygons_from_cellids(data, self._grid.dggs_type, self._grid.resolution)
        return geometryDF

    def polygon_for_extent(self, geoobj, src_epsg):
        transformer = Transformer.from_crs(f"EPSG:{src_epsg}", "EPSG:4326").transform
        try:
            geoobj = shape(geoobj)
        except Exception as e:
            print(f'Invalid Extend : {e}')
        geoobj = transform(transformer, geoobj)
        df = self._dggrid_instance.grid_cellids_for_extent(self._grid.dggs_type, self._grid.resolution, clip_geom=geoobj)
        return df

    def _replace(self, new_pd_index: PandasIndex):
        return type(self)(new_pd_index, self._dim, self._grid)

    def to_pandas_index(self):
        return self._pd_index.index

    def _get_dggrid_instance(self):
        executable = os.environ['DGGRID_PATH']
        working_dir = tempfile.mkdtemp()
        return DGGRIDv7(executable=executable, working_dir=working_dir, capture_logs=True, silent=True)

    @classmethod
    def _autoResolution(cls: type["ISEAIndex"], minlng, minlat, maxlng, maxlat, src_epsg, num_data, dggs, grid):
        print('Calculate Auto resolution')
        print(f'{minlat},{minlng},{maxlat},{maxlng}')
        df = gpd.GeoDataFrame([0], geometry=[box(minlng, minlat, maxlng, maxlat)], crs=src_epsg)
        print(f'Total Bounds ({src_epsg}): {df.total_bounds}')
        df = df.to_crs('wgs84')
        print(f'Total Bounds (wgs84): {df.total_bounds}')
        R = 6371
        lon1, lat1, lon2, lat2 = df.total_bounds
        lon1, lon2, lat1, lat2 = deg2rad(lon1), deg2rad(lon2), deg2rad(lat1), deg2rad(lat2)
        a = (sin((lon2 - lon1) / 2) ** 2 + cos(lon1) * cos(lon2) * sin(0) ** 2)
        d = 2 * arcsin(sqrt(a))
        # normalize = 4 * pi if (src_epsg.lower() == 'epsg:3301') else 1
        area = abs(d * ((power(R, 2) * sin(lat2)) - (power(R, 2) * sin(lat1))))
        print(f'Total Bounds Area (km^2): {area}')
        avg_area_per_data = (area / num_data)
        print(f'Area per center point (km^2): {avg_area_per_data}')
        dggrid_resolution = dggs.grid_stats_table(grid.upper(), 30)
        filter_ = dggrid_resolution[dggrid_resolution['Area (km^2)'] < avg_area_per_data]
        resolution = 5
        if (len(filter_) > 0):
            resolution = filter_.iloc[0, 0]
            print(f'Auto resolution : {resolution}')
        else:
            print(f'Auto resolution failed, using {resolution}')

        return resolution

