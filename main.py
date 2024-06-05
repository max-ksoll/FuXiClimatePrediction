import logging
import os
import shutil
from typing import List, Dict

import cdsapi
import numpy as np
import xarray as xr
import zarr.hierarchy
from tqdm import tqdm

from src.Dataset import cds_utils
from src.Dataset.dimensions import Variable, ORASVariable, Dimension, VariableInfo, ERAATMOSVariable, ERASURFACEVariable
from src.Dataset.zarr_utils import create_zarr_file, get_zarr_root
from src.utils import log_exec_time, get_nc_files

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@log_exec_time
def download_data(data_dir: os.PathLike | str, start_year: int, end_year: int, force_download: bool = False) \
        -> Dict[Variable, List[os.PathLike | str]]:
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
        Variable.ORAS: oras_paths,
        Variable.ERA5SURFACE: era_surface_paths,
        Variable.ERA5ATMOS: era_atmos_paths
    }


def interpolate_oras(ds: xr.Dataset, variable: VariableInfo):
    var = ds[variable.cdf_name]

    new_lat = np.linspace(-90, 90, Dimension.LAT.value.size)
    new_lon = np.linspace(0, 358.5, Dimension.LON.value.size)

    args = {
        Dimension.LAT.value.oras_name: new_lat,
        Dimension.LON.value.oras_name: new_lon,
        "method": "linear",
        "kwargs": {"fill_value": "extrapolate"}
    }

    data = var.interpolate_na(dim=Dimension.LAT.value.oras_name, method="nearest", fill_value="extrapolate")
    data = data.interpolate_na(dim=Dimension.LON.value.oras_name, method="nearest", fill_value="extrapolate")
    data = data.interp(**args)
    print(data.values)
    return data


def convert_data(root: zarr.hierarchy.Group, paths: Dict[Variable, List[os.PathLike | str]],
                 start_year: int, end_year: int):
    for key in paths.keys():
        for path in paths[key]:
            ds = xr.open_dataset(path)
            if key == Variable.ORAS:
                variable_name = str(path).split('/')[-1].split('_')[0]
                variable = list(filter(lambda x: x.value.cdf_name == variable_name, list(ORASVariable)))[0]
                variable_data = interpolate_oras(ds, variable.value)
                root[variable.value.name].append(variable_data)
            if key == Variable.ERA5ATMOS:
                for variable in list(ERAATMOSVariable):
                    root[variable.value.name].append(ds[variable.value.cdf_name])
            if key == Variable.ERA5SURFACE:
                for variable in list(ERASURFACEVariable):
                    for year in range(start_year, end_year + 1):
                        for month in range(1, 13):
                            vals = ds[variable.value.cdf_name].sel(time=f'{year}-{month}')
                            root[variable.value.name].append(vals.values)


if __name__ == '__main__':
    start_year = 1958
    end_year = 1958
    data_dir = "/Users/ksoll/git/FuXiClimatePrediction/data"
    root = get_zarr_root(data_dir, start_year, end_year)
    create_zarr_file(root)
    paths = download_data(data_dir, start_year, end_year)
    convert_data(root, paths, start_year, end_year)
    # training
