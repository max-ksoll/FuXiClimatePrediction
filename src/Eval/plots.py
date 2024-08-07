import os
from typing import Tuple, List
import cartopy.crs as ccrs
import numpy as np
import torch
from matplotlib import pyplot as plt

from src.Dataset.dimensions import LAT, LON
from src.Dataset.fuxi_dataset import FuXiDataset
from src.utils import log_exec_time


def plot_average_difference_over_time(
    fig_base_path: os.PathLike | str,
    difference: torch.Tensor,
    variable_idx: int,
    autoregression_steps_plots: List[int],
) -> Tuple[List[np.ndarray], str]:
    # AUTOREGRESSION X VARIABLES X LATITUDE X LONGITUDE
    diff = difference[:, variable_idx, :, :]
    var_name, var_level = FuXiDataset.get_var_name_and_level_at_idx(variable_idx)

    if var_level >= 0:
        var_name += f"_{var_level}"

    images = []

    for auto_step_to_plot in autoregression_steps_plots:
        if diff.shape[0] <= auto_step_to_plot:
            break

        data = diff[auto_step_to_plot]
        images.append(
            _plot_average_difference_over_time(
                fig_base_path, data, var_name, auto_step_to_plot
            )
        )

    del data, diff

    return images, var_name


@log_exec_time
def _plot_average_difference_over_time(
    fig_base_path: os.PathLike | str,
    data: torch.Tensor,
    var_name: str,
    auto_step_to_plot: int,
):
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={"projection": ccrs.Robinson()})
    ax.coastlines()

    lats = np.linspace(LAT.min_val, LAT.max_val, LAT.size)
    lons = np.linspace(-180, 180, LON.size)
    im = ax.pcolormesh(lons, lats, data, transform=ccrs.PlateCarree(), shading="auto")
    plt.colorbar(im, ax=ax, orientation="vertical")

    ax.set_title(f"{var_name} {auto_step_to_plot+1}m into future")
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close("all")
    del lats, lons, im, fig, ax

    return data


def plot_model_minus_clim(
    fig_base_path: os.PathLike | str, model_minus_clim: torch.Tensor, variable_idx: int
) -> Tuple[str | os.PathLike, str]:
    # AUTOREGRESSION X VARIABLES X LATITUDE X LONGITUDE
    difference = model_minus_clim[variable_idx, :, :]
    var_name, var_level = FuXiDataset.get_var_name_and_level_at_idx(variable_idx)

    if var_level >= 0:
        var_name += f"_{var_level}"

    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={"projection": ccrs.Robinson()})
    ax.coastlines()

    lats = np.linspace(LAT.min_val, LAT.max_val, LAT.size)
    lons = np.linspace(-180, 180, LON.size)
    im = ax.pcolormesh(
        lons, lats, difference, transform=ccrs.PlateCarree(), shading="auto"
    )
    plt.colorbar(im, ax=ax, orientation="vertical")
    ax.set_title(f"{var_name} difference to clim")
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close("all")
    del difference, lats, lons, im, fig, ax

    return data, var_name
