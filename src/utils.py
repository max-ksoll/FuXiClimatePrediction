import logging
import os
import time
from typing import Set, List, Tuple, Callable, Dict

import numpy as np
import xarray as xr
from scipy.ndimage import zoom

timing_logger = logging.getLogger("Timing Logger")
utils_logger = logging.getLogger(__name__)

def log_exec_time(func: Callable):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        timing_logger.info(
            f"Function '{func.__name__}' executed in {execution_time:.4f} seconds"
        )
        return result

    return wrapper


def get_nc_files(directory: os.PathLike | str) -> Set[str]:
    if not os.path.exists(directory):
        return {}
    return {f for f in os.listdir(directory) if f.endswith(".nc")}


def get_decades(start: int, end: int) -> List[int]:
    decades = (((end // 10) * 10) - ((start // 10) * 10)) // 10 + 1
    starting_decade = (start // 10) * 10
    return [starting_decade + 10 * i for i in range(decades)]


def get_date_strings(start_year: int, end_year: int) -> Tuple[List[str], List[int]]:
    date_strings = []
    decades = get_decades(start_year, end_year)
    for decade in decades:
        sy = decade
        ey = decade + 10
        if start_year > decade:
            sy = start_year
        if end_year < ey:
            ey = end_year + 1
        date = "/".join(
            [f"{year}{month:02d}01" for year in range(sy, ey) for month in range(1, 13)]
        )
        date_strings.append(date)
    return date_strings, decades


def get_years_as_strings(start_year: int, end_year: int) -> List[str]:
    return [str(year) for year in range(start_year, end_year + 1)]


def get_month_as_strings(start_year: int, end_year: int) -> List[str]:
    return [f"{m:02d}" for m in range(1, 13)]


def resize_data(data: np.ndarray, shape: Tuple[int, ...]):
    original_shape = data.shape
    nan_mask = np.isnan(data)
    array_filled = np.where(nan_mask, 0, data)
    zoom_factors = [n / o for n, o in zip(shape, original_shape)]
    resized_array = zoom(array_filled, zoom_factors)
    resized_nan_mask = zoom(nan_mask.astype(float), zoom_factors) > 0.5
    resized_array[resized_nan_mask] = np.nan

    return resized_array


def get_metrics_array(values: np.ndarray):
    # ['min', 'max', 'mean', 'std']
    # array -> time x lat x lon | time x level x lat x lon
    if values.ndim == 3:
        return np.array(
            [
                np.nanmin(values),
                np.nanmax(values),
                np.nanmean(values),
                np.nanstd(values),
            ]
        )
    values = np.transpose(values, [1, 0, 2, 3])
    min = np.nanmin(values, (1, 2, 3))
    max = np.nanmax(values, (1, 2, 3))
    mean = np.nanmean(values, (1, 2, 3))
    std = np.nanstd(values, (1, 2, 3))

    min = min.reshape(-1, 1)
    max = max.reshape(-1, 1)
    mean = mean.reshape(-1, 1)
    std = std.reshape(-1, 1)
    return np.concatenate((min, max, mean, std), axis=1).transpose()


def config_epoch_to_autoregression_steps(config: Dict[str, int], epoch: int) -> int:
    utils_logger.debug(f"Config: {config}, epoch: {epoch}")
    for key, value in config.items():
        if epoch < int(key):
            utils_logger.debug(f"Config key {key} is less than epoch {epoch}")
            return value
    return config.get("-1", 1)


def get_dataloader_params(batch_size: int):
    return {
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': os.cpu_count() // 2,
        'pin_memory': True
    }
