# TODO main.py -> download_data.py oder so umbenennen

import logging
import os
import shutil
import zipfile
from typing import List

import cdsapi
import numpy as np
import xarray as xr
from tqdm import tqdm

from src import utils
from src.Dataset.dimensions import (
    ORAS_VARIABLES,
    ERA_SURFACE_VARIABLES,
    ERA_ATMOS_VARIABLES,
    LAT,
    LON,
)
from src.Dataset.regridding import transform_tripolar_to_lat_lon
from src.Dataset.zarr_handler import ZarrHandler
from src.global_vars import ORAS_LAT_LON_GRID_SIZE, LAT_LON_GRID_SIZE, ERA5_VARIABLES
from src.utils import log_exec_time, get_nc_files, resize_data

downloader_logger = logging.getLogger("Data Downloader")
builder_logger = logging.getLogger("Data Builder")
converter_logger = logging.getLogger("Data Converter")


class DataDownloader:
    def __init__(self, data_dir: os.PathLike | str, start_year: int, end_year: int):
        self.data_dir = data_dir
        self.start_year = start_year
        self.end_year = end_year
        self.download_path = os.path.join(data_dir, f"sy_{start_year}__ey_{end_year}")
        self.cds_client = cdsapi.Client()

        self.oras_paths: List[os.PathLike | str] = []
        self.era_surface_paths: List[os.PathLike | str] = []
        self.era_atmos_paths: List[os.PathLike | str] = []

    def download(self, force_download: bool = False):
        downloader_logger.info(
            f"Downloading Data for {self.start_year} to {self.end_year}"
        )
        if force_download:
            downloader_logger.info("Forcing download")
            downloader_logger.info(
                "Deleting existing files in {self.download_path} and Directory itself"
            )
            shutil.rmtree(self.download_path, ignore_errors=True)
        os.makedirs(self.data_dir, exist_ok=True)
        if len(get_nc_files(self.download_path)) > 0:
            downloader_logger.info("Files already downloaded. Continuing...")
            return self.map_nc_files()

        os.makedirs(self.download_path, exist_ok=True)
        self.download_oras_data()
        self.download_era5_surface_data()
        self.download_era5_atmospheric_data()

    def map_nc_files(self):
        nc_files = set(
            map(
                lambda x: os.path.join(self.download_path, x),
                get_nc_files(self.download_path),
            )
        )
        downloader_logger.debug(f"Found Files: {nc_files}")
        self.era_atmos_paths = set(filter(lambda f: "atmos" in f, nc_files))
        downloader_logger.debug(f"ERA5 Atmospheric Files: {self.era_atmos_paths}")
        self.era_surface_paths = set(filter(lambda f: "surface" in f, nc_files))
        downloader_logger.debug(f"ERA5 Surface Files: {self.era_surface_paths}")
        self.oras_paths = nc_files - self.era_atmos_paths - self.era_surface_paths
        downloader_logger.debug(f"ORAS Files: {self.oras_paths}")

    @log_exec_time
    def download_era5_atmospheric_data(self):
        downloader_logger.info("Downloading ERA5 Atmospheric Data")
        paths = []
        decade_strings, decades = utils.get_date_strings(start_year, end_year)
        for idx in tqdm(range(len(decade_strings))):
            downloader_logger.debug(
                f"Downloading ERA5 Atmospheric for decade: {decades[idx]}"
            )
            file_path = os.path.join(
                self.download_path, f"era5_atmospheric_{decades[idx]}.nc"
            )
            self.cds_client.retrieve(
                "reanalysis-era5-complete",
                {
                    "class": "ea",
                    "date": decade_strings[idx],
                    "expver": "1",
                    "levelist": "200/300/500/850/1000",
                    "levtype": "pl",
                    "param": "129/130/131/132/133",
                    "stream": "moda",
                    "type": "an",
                    "format": "netcdf",
                    "grid": [LAT_LON_GRID_SIZE[0], LAT_LON_GRID_SIZE[1]],
                },
                file_path,
            )
            paths.append(file_path)
        self.era_atmos_paths = paths

    @log_exec_time
    def download_era5_surface_data(self):
        downloader_logger.info("Downloading ERA5 Surface Variables")
        paths = []
        month = utils.get_month_as_strings(start_year, end_year)
        for year in utils.get_years_as_strings(start_year, end_year):
            downloader_logger.debug(
                f"Downloading ERA5 Surface Variables for year {year}"
            )
            file_path = os.path.join(
                self.download_path, f"era5_surface_{start_year}.nc"
            )
            self.cds_client.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "format": "netcdf",
                    "variable": [
                        ERA5_VARIABLES
                    ],
                    "year": str(year),
                    "month": month,
                    "day": [
                        "01",
                        "02",
                        "03",
                        "04",
                        "05",
                        "06",
                        "07",
                        "08",
                        "09",
                        "10",
                        "11",
                        "12",
                        "13",
                        "14",
                        "15",
                        "16",
                        "17",
                        "18",
                        "19",
                        "20",
                        "21",
                        "22",
                        "23",
                        "24",
                        "25",
                        "26",
                        "27",
                        "28",
                        "29",
                        "30",
                        "31",
                    ],
                    "time": ["00:00", "06:00", "12:00", "18:00"],
                    "grid": [1.5, 1.5],
                },
                file_path,
            )
            paths.append(file_path)
        self.era_surface_paths = paths

    @log_exec_time
    def download_oras_data(self):
        downloader_logger.info(f"Downloading ORAS data")
        month = utils.get_month_as_strings(start_year, end_year)
        years = utils.get_years_as_strings(start_year, end_year)
        file_path = os.path.join(
            self.download_path, f"oras_{start_year}_{end_year}.zip"
        )
        self.cds_client.retrieve(
            "reanalysis-oras5",
            {
                "format": "zip",
                "product_type": "consolidated",
                "vertical_resolution": "single_level",
                "variable": [
                    "ocean_heat_content_for_the_total_water_column",
                    "ocean_heat_content_for_the_upper_300m",
                ],
                "year": years,
                "month": month,
            },
            file_path,
        )
        before = get_nc_files(self.download_path)
        downloader_logger.info(f"Unzipping ORAS Data")
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(self.download_path)
        downloader_logger.info("Removing ORAS Zip File")
        os.remove(file_path)
        after = get_nc_files(self.download_path)
        self.oras_paths = list(
            map(lambda x: os.path.join(self.download_path, x), list(after - before))
        )


class DataConverter:
    def __init__(self, zarr_handler: ZarrHandler, data_downloader: DataDownloader):
        self.zarr_handler = zarr_handler
        self.data_downloader = data_downloader

    def write(self):
        converter_logger.info("Writing Data to Zarr File")
        self.write_oras()
        self.write_era_surface()
        self.write_era_atmos()

    @log_exec_time
    def write_oras(self):
        converter_logger.info("Writing ORAS Data")
        for path in self.data_downloader.oras_paths:
            converter_logger.debug(f"Writing ORAS Data for {path}")
            ds = xr.open_dataset(path)
            variable_name = str(path).split("/")[-1].split("_")[0]
            variable = list(
                filter(lambda x: x.cdf_name == variable_name, ORAS_VARIABLES)
            )[0]
            converter_logger.debug(f"Regridding ORAS Data for {path}")
            lat_lon_grid_size = ORAS_LAT_LON_GRID_SIZE
            variable_data = transform_tripolar_to_lat_lon(
                ds[variable.cdf_name].values,
                ds["nav_lat"].values,
                ds["nav_lon"].values,
                lat_lon_grid_size,
            )
            converter_logger.debug(f"Resizing ORAS Data for {path}")
            variable_data = resize_data(variable_data, (LAT.size, LON.size))
            self.zarr_handler.append_data(variable, variable_data[None, :, :])

    @log_exec_time
    def write_era_surface(self):
        converter_logger.info("Writing ERA5 Surface Data")
        for path in self.data_downloader.era_surface_paths:
            converter_logger.debug(f"Writing ERA5 Surface for {path}")
            ds = xr.open_dataset(path)
            for variable in ERA_SURFACE_VARIABLES:
                converter_logger.debug(f"Writing ERA5 Surface for {variable.name}")
                for year in range(
                    self.data_downloader.start_year, self.data_downloader.end_year + 1
                ):
                    for month in range(1, 13):
                        converter_logger.debug(
                            f"Writing ERA5 Surface for {variable} at time {year}/{month}"
                        )
                        vals = ds[variable.cdf_name].sel(time=f"{year}-{month}").values
                        vals = np.mean(vals, axis=0, keepdims=True)
                        vals = np.flip(vals, -2)
                        vals = np.roll(vals, vals.shape[-1] // 2, axis=-1)
                        self.zarr_handler.append_data(variable, vals)

    @log_exec_time
    def write_era_atmos(self):
        converter_logger.info("Writing ERA5 Atmospheric Data")
        for path in self.data_downloader.era_atmos_paths:
            converter_logger.debug(f"Writing ERA5 Atmospheric for {path}")
            ds = xr.open_dataset(path)
            for variable in ERA_ATMOS_VARIABLES:
                converter_logger.debug(f"Writing ERA5 Atmospheric for {variable}")
                vals = ds[variable.cdf_name].values
                vals = np.flip(vals, -2)
                vals = np.roll(vals, vals.shape[-1] // 2, axis=-1)
                self.zarr_handler.append_data(variable, vals)


class DataBuilder:
    def __init__(self, data_dir: os.PathLike | str, start_year: int, end_year: int):
        self.builder = ZarrHandler(data_dir, start_year, end_year)
        self.data_downloader = DataDownloader(data_dir, start_year, end_year)
        self.data_converter = DataConverter(self.builder, self.data_downloader)

    def generate_data(self, force_donwload: bool = False):
        builder_logger.info("Generating data...")
        self.builder.build()
        self.data_downloader.download(force_donwload)
        self.data_converter.write()
        self.builder.finish()
        builder_logger.info("Finished Generating data...")


if __name__ == "__main__":
    start_year = 1958
    end_year = 2010
    data_dir = "/Users/ksoll/git/FuXiClimatePrediction/data"

    builder = DataBuilder(data_dir, start_year, end_year)
    builder.generate_data()
