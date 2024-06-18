import xarray as xr

from src.Dataset.era5_dataset import ERA5Dataset

ds = ERA5Dataset(dataset_path="/Users/ksoll/git/FuXiClimatePrediction/data/1958_1958.zarr",
                 means_file="/Users/ksoll/git/FuXiClimatePrediction/data/mean_1958_1958.zarr")
