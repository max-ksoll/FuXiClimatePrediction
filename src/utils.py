import copy
import logging
import os
import time
from typing import Set, List, Tuple, Callable, Dict, Any

import numpy as np
import torch
import zarr
from scipy.ndimage import zoom

from src.Dataset.dimensions import (
    Dimension,
    SURFACE_VARIABLES,
    LEVEL_VARIABLES,
    METRICS_ARRAY,
)

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


def get_years_as_strings(
    start_year: int, end_year: int, wrap_n_elements_in_array=-1
) -> List[List[str]]:
    if wrap_n_elements_in_array > 0:
        res = []
        tmp = []
        for year_idx, year in enumerate(range(start_year, end_year + 1)):
            if not year_idx % wrap_n_elements_in_array and len(tmp) > 0:
                res.append(copy.copy(tmp))
                tmp.clear()
            tmp.append(str(year))
        if len(tmp) > 0:
            res.append(copy.copy(tmp))
        return res
    return [[str(year) for year in range(start_year, end_year + 1)]]


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


def get_dataloader_params(
    batch_size: int, is_train_dataloader: bool = False
) -> Dict[str, Any]:
    return {
        "batch_size": batch_size,
        "shuffle": is_train_dataloader,
        "num_workers": min(os.cpu_count() // 2, 8),
        "pin_memory": True,
    }


def get_latitude_weights(lat_dim: Dimension):
    weights = np.cos(
        np.deg2rad(
            np.array(np.linspace(lat_dim.min_val, lat_dim.max_val, lat_dim.size))
        )
    )
    return torch.Tensor(weights)[:, None]


def get_clima_mean(
    dataset_path: os.PathLike | str, means_file: os.PathLike | str
) -> torch.Tensor:
    store = zarr.DirectoryStore(dataset_path)
    sources = zarr.group(store=store)

    store = zarr.DirectoryStore(means_file)
    means = zarr.group(store=store)
    clim = torch.cat(
        [
            torch.mean(
                torch.stack(
                    [
                        torch.tensor(np.array(sources[var.name]))
                        for var in SURFACE_VARIABLES
                    ],
                    0,
                ),
                dim=1,
            ),
            torch.mean(
                torch.stack(
                    [
                        torch.tensor(np.array(sources[var.name]))
                        for var in LEVEL_VARIABLES
                    ],
                    0,
                ),
                dim=1,
            ).flatten(start_dim=0, end_dim=1),
        ],
        dim=0,
    )

    mode_idx = METRICS_ARRAY.index("min")
    min = torch.cat(
        [
            torch.stack(
                [
                    torch.tensor(np.array(means[var.name][mode_idx]))
                    for var in SURFACE_VARIABLES
                ],
                0,
            ),
            torch.stack(
                [
                    torch.tensor(np.array(means[var.name][mode_idx, :]))
                    for var in LEVEL_VARIABLES
                ],
                0,
            ).flatten(),
        ],
        dim=0,
    )

    mode_idx = METRICS_ARRAY.index("max")
    max = torch.cat(
        [
            torch.stack(
                [
                    torch.tensor(np.array(means[var.name][mode_idx]))
                    for var in SURFACE_VARIABLES
                ],
                0,
            ),
            torch.stack(
                [
                    torch.tensor(np.array(means[var.name][mode_idx, :]))
                    for var in LEVEL_VARIABLES
                ],
                0,
            ).flatten(),
        ],
        dim=0,
    )

    return (clim - min[:, None, None]) / (max - min)[:, None, None]


if __name__ == "__main__":
    auto = {
        "25": 1,
        "50": 2,
        "75": 4,
        "100": 6,
        "125": 8,
        "150": 10,
        "-1": 12,
    }
    for i in range(100):
        print(i, config_epoch_to_autoregression_steps(auto, i))
