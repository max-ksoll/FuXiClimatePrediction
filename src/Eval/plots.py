import os
from typing import Tuple, List

import cartopy.crs as ccrs
import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.Dataset.dimensions import LAT, LON
from src.Dataset.fuxi_dataset import FuXiDataset
from src.utils import log_exec_time, get_subdirectories


def plot_average_difference_over_time(
    path,
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
        _plot_average_difference_over_time(path, data, var_name, auto_step_to_plot)


@log_exec_time
def _plot_average_difference_over_time(
    path: os.PathLike | str,
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

    ax.set_title(f"{var_name} {auto_step_to_plot+1} month into future")
    plt.savefig(os.path.join(path, f"diff_{var_name}_{auto_step_to_plot+1}.jpg"))
    plt.close(fig)


def plot_model_minus_clim(
    path, model_minus_clim: torch.Tensor, variable_idx: int
) -> Tuple[np.ndarray, str]:
    # AUTOREGRESSION X VARIABLES X LATITUDE X LONGITUDE
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={"projection": ccrs.Robinson()})
    ax.coastlines()

    difference = model_minus_clim[variable_idx, :, :]
    var_name, var_level = FuXiDataset.get_var_name_and_level_at_idx(variable_idx)

    if var_level >= 0:
        var_name += f"_{var_level}"

    lats = np.linspace(LAT.min_val, LAT.max_val, LAT.size)
    lons = np.linspace(-180, 180, LON.size)
    im = ax.pcolormesh(
        lons, lats, difference, transform=ccrs.PlateCarree(), shading="auto"
    )
    plt.colorbar(im, ax=ax, orientation="vertical")
    ax.set_title(f"{var_name} difference to clim")

    plt.savefig(os.path.join(path, f"{var_name}_clim.jpg"))

    plt.close(fig)


if __name__ == "__main__":

    for run_id_path in get_subdirectories(
        "/Users/ksoll/git/FuXiClimatePrediction/tensors/tensors"
    ):
        for epoch_path in tqdm(
            sorted(
                get_subdirectories(run_id_path),
                key=lambda x: int(x.split("/")[-1]),
                reverse=True,
            )
        ):
            diff_pt = torch.load(os.path.join(epoch_path, "diff.pt"))
            clim_pt = torch.load(os.path.join(epoch_path, "clim.pt"))
            print(epoch_path)
            for var_idx in tqdm(range(diff_pt.shape[1])):
                plot_average_difference_over_time(epoch_path, diff_pt, var_idx, [5])
                plot_model_minus_clim(epoch_path, clim_pt, var_idx)
                plt.close("all")
