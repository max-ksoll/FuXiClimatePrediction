import os
from typing import Tuple, List

import numpy as np
import torch
from matplotlib import pyplot as plt

from src.Dataset.fuxi_dataset import FuXiDataset
from src.Eval.scores import weighted_rmse, weighted_acc, weighted_mae
from src.utils import log_exec_time

import cartopy.crs as ccrs


class ModelEvaluator:
    def __init__(
        self,
        clima_mean: torch.Tensor,
        lat_weights: torch.Tensor,
        fig_path: str | os.PathLike,
    ):
        self.clima_mean = clima_mean.detach().cpu()
        self.lat_weights = lat_weights.detach().cpu()
        self.fig_path = fig_path
        # In month 1, 3, 5, ...
        # Es werden nur Grafiken erstellt für die Vorhandenen Schritte
        # 1 Autoregression -> nur 1 Plot
        # 5 Autoregression -> 3 Plots
        self.autoregression_steps_plots = [
            0,
            2,
            4,
            6,
            9,
        ]
        self.model_outs = {}
        self.gt = {}

    def reset(self):
        self.model_outs.clear()

    def update(self, outs: torch.Tensor, gt: torch.Tensor, batch_idx: int):
        """

        Args:
            gt: Groundtruth
            outs: Tensor of Shape Batch Size x Autoregression x Variables x Latitude x Longitude
            batch_idx: Idx of the Batch, because of possible DDP

        Returns:

        """
        self.model_outs[batch_idx] = outs.detach().cpu()
        self.gt[batch_idx] = gt.detach().cpu()

    def evaluate(self):
        model_preds = torch.stack(
            [self.model_outs[key] for key in sorted(self.model_outs.keys())]
        )

        gt = torch.stack([self.gt[key] for key in sorted(self.model_outs.keys())])

        assert model_preds.shape == gt.shape

        diff_tensor_list = []
        model_out_minus_clim = []
        accs = []
        maes = []
        rmses = []

        for idx in range(model_preds.shape[0]):
            timeseries = gt[idx]
            model_outs = model_preds[idx]

            # accs.append(
            #     weighted_acc(model_outs, timeseries, self.lat_weights, self.clima_mean)
            # )
            # maes.append(weighted_mae(model_outs, timeseries, self.lat_weights))
            # rmses.append(weighted_rmse(model_outs, timeseries, self.lat_weights))
            #
            diff_tensor_list.append(model_outs - timeseries)
            model_out_minus_clim.append(model_outs - self.clima_mean)

        # return_dict = {
        #     "acc": np.mean(accs),
        #     "mae": np.mean(maes),
        #     "rmse": np.mean(rmses),
        # }
        return_dict = dict()

        diff_tensor = torch.cat(diff_tensor_list)
        diff_tensor = diff_tensor.nanmean(dim=0)
        diff_tensor_list.clear()

        model_minus_clim = torch.cat(model_out_minus_clim)
        model_minus_clim = model_minus_clim.nanmean(dim=[0, 1])
        model_out_minus_clim.clear()

        # TODO Level wollen wir einerseits gemeant und andererseits auch einzeln haben
        image_dict_avg_diff = {}
        image_dict_minus_clim = {}
        for var_idx in range(35):
            paths, var_name = self.plot_average_difference_over_time(
                diff_tensor, var_idx
            )
            image_dict_avg_diff[var_name] = paths
            path, var_name = self.plot_model_minus_clim(model_minus_clim, var_idx)
            image_dict_minus_clim[var_name] = path

        return_dict["img"] = {}
        return_dict["img"]["average_difference_over_time"] = image_dict_avg_diff
        return_dict["img"]["model_out_minus_clim"] = image_dict_minus_clim

        return return_dict
        # return {
        #     "acc": 0,
        #     "mae": 0,
        #     "rmse": 0,
        #     "img": {
        #         "average_difference_over_time": {
        #             "specific_humidity": ["fig_path"],
        #             "temperature": ["fig_path"],
        #             "u_component_of_wind": ["fig_path"],
        #             "v_component_of_wind": ["fig_path"],
        #         },
        #         "model_out_minus_clim": {
        #             "specific_humidity": ["fig_path"],
        #             "temperature": ["fig_path"],
        #             "u_component_of_wind": ["fig_path"],
        #             "v_component_of_wind": ["fig_path"],
        #         },
        #     },
        # }

    def plot_average_difference_over_time(
        self, difference, variable_idx
    ) -> Tuple[List[str | os.PathLike], str]:
        # AUTOREGRESSION X VARIABLES X LATITUDE X LONGITUDE
        difference[:, :, -2:2, :] = 0
        difference = difference[:, variable_idx, :, :]
        var_name, var_level = FuXiDataset.get_var_name_and_level_at_idx(variable_idx)

        if var_level >= 0:
            var_name += f" {var_level}"

        paths = []

        for auto_step_to_plot in self.autoregression_steps_plots:
            if difference.shape[0] <= auto_step_to_plot:
                return paths, var_name

            data = difference[auto_step_to_plot]
            paths.append(
                self._plot_average_difference_over_time(
                    data, var_name, auto_step_to_plot
                )
            )

    @log_exec_time
    def _plot_average_difference_over_time(self, data, var_name, auto_step_to_plot):
        fig, ax = plt.subplots(
            figsize=(12, 8), subplot_kw={"projection": ccrs.Robinson()}
        )
        ax.coastlines()

        lats = np.linspace(LAT.min_val, LAT.max_val, LAT.size)
        lons = np.linspace(-180, 180, LON.size)
        im = ax.pcolormesh(
            lons, lats, data, transform=ccrs.PlateCarree(), shading="auto"
        )
        plt.colorbar(im, ax=ax, orientation="vertical")

        # Daten plotten. ,vmin=-getExtrem(variable), vmax=getExtrem(variable)
        # plt.colorbar(im, ax=ax, orientation="horizontal", shrink=0.5).set_label(
        #     f"bias [{get_units(variable)}]"
        # )
        ax.set_title(f"{var_name} {auto_step_to_plot+1}m into future")
        save_path = os.path.join(
            self.fig_path,
            f"avg-diff-time_{var_name}_{auto_step_to_plot+1}m_into_future.png",
        )

        plt.savefig(save_path)
        plt.close()

        return save_path

    def plot_model_minus_clim(
        self, model_minus_clim, variable_idx
    ) -> Tuple[str | os.PathLike, str]:
        # AUTOREGRESSION X VARIABLES X LATITUDE X LONGITUDE
        difference = model_minus_clim[variable_idx, :, :]
        var_name, var_level = FuXiDataset.get_var_name_and_level_at_idx(variable_idx)

        if var_level >= 0:
            var_name += f" {var_level}"

        fig, ax = plt.subplots(
            figsize=(12, 8), subplot_kw={"projection": ccrs.Robinson()}
        )
        ax.coastlines()

        lats = np.linspace(LAT.min_val, LAT.max_val, LAT.size)
        lons = np.linspace(-180, 180, LON.size)
        im = ax.pcolormesh(
            lons, lats, difference, transform=ccrs.PlateCarree(), shading="auto"
        )
        plt.colorbar(im, ax=ax, orientation="vertical")

        # Daten plotten. ,vmin=-getExtrem(variable), vmax=getExtrem(variable)
        # plt.colorbar(im, ax=ax, orientation="horizontal", shrink=0.5).set_label(
        #     f"bias [{get_units(variable)}]"
        # )
        ax.set_title(f"{var_name} difference to clim")
        save_path = os.path.join(
            self.fig_path,
            f"diff-clim_{var_name}.png",
        )

        plt.savefig(save_path)
        plt.close()

        return save_path, var_name


if __name__ == "__main__":
    from src.utils import get_latitude_weights
    from src.Dataset.dimensions import LAT, LON
    import logging

    logging.basicConfig(level=logging.INFO)

    BS = 3
    AR = 4

    eval = ModelEvaluator(
        clima_mean=torch.rand((35, 121, 240)),
        lat_weights=get_latitude_weights(LAT),
        fig_path="/Users/ksoll/git/FuXiClimatePrediction/data-viz",
    )
    eval.reset()
    for idx, elem in enumerate(range(21)):
        gt = torch.rand((BS, AR, 35, 121, 240))
        model_out = gt + torch.rand_like(gt)
        eval.update(model_out, gt, idx)
    print(eval.evaluate())
