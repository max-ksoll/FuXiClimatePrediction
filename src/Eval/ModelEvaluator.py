import os

import numpy as np
import torch

from src.Dataset.fuxi_dataset import FuXiDataset
from src.Eval.scores import weighted_rmse, weighted_acc, weighted_mae


class ModelEvaluator:
    def __init__(
        self,
        clima_mean: torch.Tensor,
        lat_weights: torch.Tensor,
        dataloader: torch.utils.data.DataLoader,
        fig_path: str | os.PathLike,
    ):
        self.clima_mean = clima_mean.detach().cpu()
        self.lat_weights = lat_weights.detach().cpu()
        self.dataloader = dataloader
        self.fig_path = fig_path
        # In month 1, 3, 5, ...
        # Es werden nur Grafiken erstellt für die Vorhandenen Schritte
        # 1 Autoregression -> nur 1 Plot
        # 5 Autoregression -> 3 Plots
        self.autoregression_steps_for_average_difference_over_time_plots = [
            1,
            3,
            5,
            7,
            10,
        ]
        self.model_outs = {}

    def reset(self):
        self.model_outs.clear()

    def set_dataloader(self, dataloader):
        self.dataloader = dataloader

    def update(self, outs: torch.Tensor, batch_idx: int):
        """

        Args:
            outs: Tensor of Shape Batch Size x Autoregression x Variables x Latitude x Longitude
            batch_idx: Idx of the Batch, because of possible DDP

        Returns:

        """
        self.model_outs[batch_idx] = outs.detach().cpu()

    def evaluate(self):
        model_preds = torch.stack(
            [self.model_outs[key] for key in sorted(self.model_outs.keys())]
        )
        diff_tensor_list = []
        accs = []
        maes = []
        rmses = []
        for idx, batch in enumerate(self.dataloader):
            batch = batch[:, 2:]
            model_outs = model_preds[idx]
            accs.append(
                weighted_acc(model_outs, batch, self.lat_weights, self.clima_mean)
            )
            maes.append(weighted_mae(model_outs, batch, self.lat_weights))
            rmses.append(weighted_rmse(model_outs, batch, self.lat_weights))
            diff_tensor_list.append(batch - model_outs)
        diff_tensor = torch.cat(diff_tensor_list)
        diff_tensor_list.clear()
        diff_tensor.nanmean(dim=0)

        return_dict = {
            "acc": np.mean(accs),
            "mae": np.mean(maes),
            "rmse": np.mean(rmses),
        }

        # TODO Level wollen wir einerseits gemeant und andererseits auch einzeln haben
        for _ in []:
            ...

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


if __name__ == "__main__":
    from src.utils import get_dataloader_params, get_latitude_weights
    from torch.utils.data import DataLoader
    from src.Dataset.dimensions import LAT

    BS = 3
    AR = 4

    ds = FuXiDataset(
        dataset_path="/Users/ksoll/git/FuXiClimatePrediction/data/1958_1959.zarr",
        means_file="/Users/ksoll/git/FuXiClimatePrediction/data/mean_1958_1959.zarr",
        max_autoregression_steps=AR,
    )
    dl = DataLoader(ds, **get_dataloader_params(BS))
    eval = ModelEvaluator(
        clima_mean=torch.rand((35, 121, 240)),
        lat_weights=get_latitude_weights(LAT),
        dataloader=dl,
        fig_path="/Users/ksoll/git/FuXiClimatePrediction/data-viz",
    )
    eval.reset()
    for idx, elem in enumerate(dl):
        eval.update(torch.rand((BS, AR, 35, 121, 240)), idx)
    print(eval.evaluate())
