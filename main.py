import logging
import os
import shutil
import xarray as xr
import cdsapi
from typing import List, Dict

from src.Dataset import cds_utils
from src.Dataset.zarr_utils import create_zarr_file
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


def convert_data(paths: Dict[str, List[os.PathLike | str]]):
    for key in paths.keys():
        for path in paths[key]:
            if key == 'ORAS':
                ds = xr.open_dataset(path)
                print(ds)
                return


if __name__ == '__main__':
    start_year = 1958
    end_year = 1958
    data_dir = "/Users/ksoll/git/FuXiClimatePrediction/data"
    create_zarr_file(data_dir, start_year, end_year)
    paths = download_data(data_dir, start_year, end_year)
    convert_data(paths)
    # training
