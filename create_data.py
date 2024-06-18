# TODO main.py -> download_data.py oder so umbenennen

import logging
import os
import shutil
from typing import List, Dict

import cdsapi
import numpy as np
import xarray as xr
import zarr.hierarchy
from scipy.ndimage import zoom

from src.Dataset import cds_utils
from src.Dataset.dimensions import ORASVariable, Dimension, VariableInfo, ERAATMOSVariable, \
    ERASURFACEVariable, VARIABLES
from src.Dataset.zarr_utils import create_zarr_file, get_zarr_root
from src.utils import log_exec_time, get_nc_files

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@log_exec_time
def download_data(data_dir: os.PathLike | str, start_year: int, end_year: int, force_download: bool = False) \
        -> Dict[str, List[os.PathLike | str]]:
    cds_client = cdsapi.Client()
    ds_folder_name = f'sy_{start_year}__ey_{end_year}'
    ds_download_path = os.path.join(data_dir, ds_folder_name)

    if force_download:
        shutil.rmtree(ds_download_path, ignore_errors=True)
    if len(get_nc_files(ds_download_path)) > 0:
        logger.info(f'Skipping Download, Mapping Exisiting NC Files')
        return cds_utils.map_nc_files(ds_download_path)
    os.makedirs(ds_download_path, exist_ok=True)
    oras_paths = cds_utils.download_oras_data(cds_client, ds_download_path, start_year, end_year)
    era_surface_paths = cds_utils.download_era5_surface_data(cds_client, ds_download_path, start_year, end_year)
    era_atmos_paths = cds_utils.download_era5_atmospheric_data(cds_client, ds_download_path, start_year, end_year)

    return {
        'ORAS': oras_paths,
        'ERA5SURFACE': era_surface_paths,
        'ERA5ATMOS': era_atmos_paths
    }


@log_exec_time
def resize_data(ds: xr.Dataset, variable: VariableInfo):
    var = ds[variable.cdf_name].values
    original_shape = var.shape
    nan_mask = np.isnan(var)
    array_filled = np.where(nan_mask, 0, var)
    zoom_factors = [n / o for n, o in zip((1, Dimension.LAT.value.size, Dimension.LON.value.size), original_shape)]
    resized_array = zoom(array_filled, zoom_factors)
    resized_nan_mask = zoom(nan_mask.astype(float), zoom_factors) > 0.5
    resized_array[resized_nan_mask] = np.nan

    return resized_array


@log_exec_time
def convert_data(root: zarr.hierarchy.Group, mean_root: zarr.hierarchy.Group, paths: Dict[str, List[os.PathLike | str]],
                 start_year: int, end_year: int):
    for key in paths.keys():
        for path in paths[key]:
            ds = xr.open_dataset(path)
            if key == 'ORAS':
                write_oras(root, mean_root, ds, path)
            if key == 'ERA5SURFACE':
                write_era_surface(root, mean_root, ds, start_year, end_year)
            if key == 'ERA5ATMOS':
                write_era_atmos(root, mean_root, ds)
    mean_oras(root, mean_root)


@log_exec_time
def get_metrics_array(values: np.ndarray):
    # ['min', 'max', 'mean', 'std']
    # array -> time x lat x lon | time x level x lat x lon
    if values.ndim == 3:
        return np.array([
            np.nanmin(values),
            np.nanmax(values),
            np.nanmean(values),
            np.nanstd(values)
        ])[None, :]
    values = np.transpose(values, [1, 0, 2, 3])
    min = np.nanmin(values, (1, 2, 3))
    max = np.nanmax(values, (1, 2, 3))
    mean = np.nanmean(values, (1, 2, 3))
    std = np.nanstd(values, (1, 2, 3))

    min = min.reshape(1, -1, 1)
    max = max.reshape(1, -1, 1)
    mean = mean.reshape(1, -1, 1)
    std = std.reshape(1, -1, 1)
    return np.concatenate((min, max, mean, std), axis=2)


@log_exec_time
def write_era_surface(root: zarr.hierarchy.Group, mean_root: zarr.hierarchy.Group,
                      ds: xr.Dataset, start_year: int, end_year: int):
    for variable in list(ERASURFACEVariable):
        variable_name = variable.value.name
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                vals = ds[variable.value.cdf_name].sel(time=f'{year}-{month}').values
                vals = np.mean(vals, axis=0, keepdims=True)
                root[variable_name].append(vals)

        vals = ds[variable.value.cdf_name].values
        metrics = get_metrics_array(vals)
        mean_root[variable_name].append(metrics)


@log_exec_time
def write_era_atmos(root: zarr.hierarchy.Group, mean_root: zarr.hierarchy.Group, ds: xr.Dataset):
    for variable in list(ERAATMOSVariable):
        root[variable.value.name].append(ds[variable.value.cdf_name].values)
        metrics = get_metrics_array(ds[variable.value.cdf_name].values)
        mean_root[variable.value.name].append(metrics)


@log_exec_time
def write_oras(root: zarr.hierarchy.Group, mean_root: zarr.hierarchy.Group,
               ds: xr.Dataset, path: os.PathLike | str):
    variable_name = str(path).split('/')[-1].split('_')[0]
    variable = list(filter(lambda x: x.value.cdf_name == variable_name, list(ORASVariable)))[0]
    variable_data = resize_data(ds, variable.value)
    root[variable.value.name].append(variable_data)


@log_exec_time
def mean_oras(root, mean_root):
    for variable in list(ORASVariable):
        metrics = get_metrics_array(root[variable.value.name])
        mean_root[variable.value.name].append(metrics)


if __name__ == '__main__':
    start_year = 1958
    end_year = 1958
    data_dir = "/Users/ksoll/git/FuXiClimatePrediction/data"
    os.makedirs(data_dir, exist_ok=True)
    root = get_zarr_root(data_dir, start_year, end_year)
    dims = [
        Dimension.TIME,
        Dimension.LAT,
        Dimension.LON,
        Dimension.LEVEL,
    ]
    create_zarr_file(root, dims, VARIABLES)
    paths = download_data(data_dir, start_year, end_year)

    mean_root = get_zarr_root(data_dir, start_year, end_year, is_mean_file=True)
    dims = [
        Dimension.METRICS_TIME,
        Dimension.LEVEL,
        Dimension.METRICS
    ]
    create_zarr_file(mean_root, dims, VARIABLES, is_mean_file=True)

    convert_data(root, mean_root, paths, start_year, end_year)
    zarr.consolidate_metadata(root.store)
    zarr.consolidate_metadata(mean_root.store)
