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
) -> Tuple[List[str | os.PathLike], str]:
    # AUTOREGRESSION X VARIABLES X LATITUDE X LONGITUDE
    diff = difference[:, variable_idx, :, :]
    var_name, var_level = FuXiDataset.get_var_name_and_level_at_idx(variable_idx)

    if var_level >= 0:
        var_name += f"{var_level}"

    paths = []

    for auto_step_to_plot in autoregression_steps_plots:
        if diff.shape[0] <= auto_step_to_plot:
            return paths, var_name

        data = diff[auto_step_to_plot]
        paths.append(
            _plot_average_difference_over_time(
                fig_base_path, data, var_name, auto_step_to_plot
            )
        )


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
    save_path = os.path.join(
        fig_base_path,
        f"avg-diff-time_{var_name}_{auto_step_to_plot+1}m_into_future.png",
    )

    plt.savefig(save_path)
    plt.close("all")

    return save_path


def plot_model_minus_clim(
    fig_base_path: os.PathLike | str, model_minus_clim: torch.Tensor, variable_idx: int
) -> Tuple[str | os.PathLike, str]:
    # AUTOREGRESSION X VARIABLES X LATITUDE X LONGITUDE
    difference = model_minus_clim[variable_idx, :, :]
    var_name, var_level = FuXiDataset.get_var_name_and_level_at_idx(variable_idx)

    if var_level >= 0:
        var_name += f"{var_level}"

    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={"projection": ccrs.Robinson()})
    ax.coastlines()

    lats = np.linspace(LAT.min_val, LAT.max_val, LAT.size)
    lons = np.linspace(-180, 180, LON.size)
    im = ax.pcolormesh(
        lons, lats, difference, transform=ccrs.PlateCarree(), shading="auto"
    )
    plt.colorbar(im, ax=ax, orientation="vertical")
    ax.set_title(f"{var_name} difference to clim")
    save_path = os.path.join(
        fig_base_path,
        f"diff-clim_{var_name}.png",
    )

    plt.savefig(save_path)
    plt.close("all")

    return save_path, var_name
