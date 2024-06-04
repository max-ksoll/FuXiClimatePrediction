import logging
import os
import zipfile
from typing import List, Dict, Tuple
from tqdm import tqdm
import cdsapi

from src import utils
from src.utils import log_exec_time, get_nc_files, get_month_as_strings, get_years_as_strings

logger = logging.getLogger(__name__)


@log_exec_time
def download_era5_atmospheric_data(cds_client: cdsapi.api.Client, directory: os.PathLike | str,
                                   start_year: int, end_year: int) -> List[os.PathLike | str]:
    paths = []
    decade_strings, decades = utils.get_date_strings(start_year, end_year)
    for idx in tqdm(range(len(decade_strings))):
        file_path = os.path.join(directory, f"era5_atmospheric_{decades[idx]}.nc")
        cds_client.retrieve("reanalysis-era5-complete", {
            "class": "ea",
            "date": decade_strings[idx],
            "expver": "1",
            "levelist": "200/300/500/850/1000",
            "levtype": "pl",
            "param": "129/130/131/132/133",
            "stream": "moda",
            "type": "an",
            "format": "netcdf",
            'grid': [1.5, 1.5]
        }, file_path)
        paths.append(file_path)
    return paths


@log_exec_time
def download_era5_surface_data(cds_client: cdsapi.api.Client, directory: os.PathLike | str,
                               start_year: int, end_year: int) -> List[os.PathLike | str]:
    paths = []
    month = get_month_as_strings(start_year, end_year)
    for year in tqdm(get_years_as_strings(start_year, end_year)):
        file_path = os.path.join(directory, f'era5_surface_{start_year}.nc')
        cds_client.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': [
                    '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature',
                    'sea_surface_temperature', 'soil_temperature_level_1',
                    'surface_latent_heat_flux', 'total_precipitation', 'volumetric_soil_water_layer_2',
                ],
                'year': str(year),
                'month': month,
                'day': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                    '13', '14', '15',
                    '16', '17', '18',
                    '19', '20', '21',
                    '22', '23', '24',
                    '25', '26', '27',
                    '28', '29', '30',
                    '31',
                ],
                'time': ['00:00', '06:00', '12:00', '18:00'],
                'grid': [1.5, 1.5]
            },
            file_path)
        paths.append(file_path)
    return paths


@log_exec_time
def download_oras_data(cds_client: cdsapi.api.Client, directory: os.PathLike | str,
                       start_year: int, end_year: int) -> List[os.PathLike | str]:
    month = get_month_as_strings(start_year, end_year)
    years = get_years_as_strings(start_year, end_year)
    file_path = os.path.join(directory, f'oras_{start_year}_{end_year}.zip')
    cds_client.retrieve(
        'reanalysis-oras5',
        {
            'format': 'zip',
            'product_type': 'consolidated',
            'vertical_resolution': 'single_level',
            'variable': [
                'ocean_heat_content_for_the_total_water_column', 'ocean_heat_content_for_the_upper_300m',
            ],
            'year': years,
            'month': month,
            'grid': [1.5, 1.5]
        },
        file_path)
    before = get_nc_files(directory)
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(directory)
    os.remove(file_path)
    after = get_nc_files(directory)
    return list(after - before)


def map_nc_files(ds_download_path: os.PathLike | str) -> Dict[str, List[str]]:
    nc_files = set(map(lambda x: os.path.join(ds_download_path, x), get_nc_files(ds_download_path)))
    era_atmos_paths = set(filter(lambda f: 'atmos' in f, nc_files))
    era_surface_paths = set(filter(lambda f: 'surface' in f, nc_files))
    oras_paths = nc_files - era_atmos_paths - era_surface_paths
    return {
        'ORAS': list(oras_paths),
        'ERA5SURFACE': list(era_surface_paths),
        'ERA5ATMOS': list(era_atmos_paths)
    }


if __name__ == '__main__':
    cds_client = cdsapi.Client()

    download_era5_surface_data(
        cds_client,
        "/Users/ksoll/git/FuXiClimatePrediction/data/sy_1958__ey_1958",
        0, 0
    )
