import csv
import logging
import os
import sys

sys.path.append(os.environ["MODULE_PATH"])

import torch
import numpy as np
from torch.utils.data import DataLoader

from src.Dataset.dimensions import LAT
from src.utils import get_dataloader_params, get_latitude_weights
from src.Dataset.fuxi_dataset import FuXiDataset
from src.PyModel.fuxi_ligthning import FuXi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    def __init__(
        self,
        model_path: os.PathLike | str,
        dataset: DataLoader,
        autoregression_steps: int,
        output_path,
    ):
        self.autoregression_steps = autoregression_steps
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)

        self.dataset = dataset
        self.model = None
        self.load_model(model_path)
        self.lat_weights = get_latitude_weights(LAT)
        self.results_file_path = os.path.join(
            output_path, f"results_{self.autoregression_steps}_steps.csv"
        )

    @torch.no_grad()
    def evaluate(self):
        mae = np.array([])
        mse = np.array([])

        lat_weighted_mae = np.array([])
        lat_weighted_mse = np.array([])

        mae_vars = None
        mse_vars = None
        lat_weighted_mae_vars = None
        lat_weighted_mse_vars = None

        for x in iter(self.dataset):
            last_timestep = x[:, -1]

            x = x.cuda()
            model_out = self.model(x, None).cpu()
            out_last_timestep = model_out[:, -1]

            # last_timestep = self.dataset.dataset.denormalize(last_timestep)
            # out_last_timestep = self.dataset.dataset.denormalize(out_last_timestep)

            abs_error = torch.abs(last_timestep - out_last_timestep)
            mask = ~torch.isnan(abs_error)

            mask_sum = mask.sum(dim=[1, 2, 3])
            mask_sum_vars = mask.sum(dim=[2, 3])

            abs_error = torch.nan_to_num(abs_error, nan=0.0)

            abs_error_sqr = abs_error**2
            abs_error_sqr_lat = abs_error_sqr * self.lat_weights

            mae = np.append(
                mae, (torch.sum(abs_error, dim=[1, 2, 3]) / mask_sum).numpy()
            )

            mse = np.append(
                mse, (torch.sum(abs_error_sqr, dim=[1, 2, 3]) / mask_sum).numpy()
            )

            mae_per_var = (torch.sum(abs_error, dim=[2, 3]) / mask_sum_vars).numpy().T
            if mae_vars is not None:
                mae_vars = np.append(mae_vars, mae_per_var, axis=1)
            else:
                mae_vars = mae_per_var

            mse_per_var = (
                (torch.sum(abs_error_sqr, dim=[2, 3]) / mask_sum_vars).numpy().T
            )
            if mse_vars is not None:
                mse_vars = np.append(mse_vars, mse_per_var, axis=1)
            else:
                mse_vars = mse_per_var

            lat_weighted_mae = np.append(
                lat_weighted_mae,
                (
                    torch.sum(abs_error * self.lat_weights, dim=[1, 2, 3]) / mask_sum
                ).numpy(),
            )
            lat_weighted_mse = np.append(
                lat_weighted_mse,
                (torch.sum(abs_error_sqr_lat, dim=[1, 2, 3]) / mask_sum).numpy(),
            )

            lat_weighted_mae_per_var = (
                (torch.sum(abs_error * self.lat_weights, dim=[2, 3]) / mask_sum_vars)
                .numpy()
                .T
            )
            if lat_weighted_mae_vars is not None:
                lat_weighted_mae_vars = np.append(
                    lat_weighted_mae_vars, lat_weighted_mae_per_var, axis=1
                )
            else:
                lat_weighted_mae_vars = lat_weighted_mae_per_var

            lat_weighted_mse_per_var = (
                (torch.sum(abs_error_sqr_lat, dim=[2, 3]) / mask_sum_vars).numpy().T
            )
            if lat_weighted_mse_vars is not None:
                lat_weighted_mse_vars = np.append(
                    lat_weighted_mse_vars, lat_weighted_mse_per_var, axis=1
                )
            else:
                lat_weighted_mse_vars = lat_weighted_mse_per_var

        rmse = mse**0.5
        lat_weighted_rmse = lat_weighted_mse**0.5

        results = {
            "mean.all.-1.mae": np.mean(mae),
            "std.all.-1.mae": np.std(mae),
            "mean.all.-1.mse": np.mean(mse),
            "std.all.-1.mse": np.std(mse),
            "mean.all.-1.rmse": np.mean(rmse),
            "std.all.-1.rmse": np.std(rmse),
            "mean.all.-1.mae.lat_weighted": np.mean(lat_weighted_mae),
            "std.all.-1.mae.lat_weighted": np.std(lat_weighted_mae),
            "mean.all.-1.mse.lat_weighted": np.mean(lat_weighted_mse),
            "std.all.-1.mse.lat_weighted": np.std(lat_weighted_mse),
            "mean.all.-1.rmse.lat_weighted": np.mean(lat_weighted_rmse),
            "std.all.-1.rmse.lat_weighted": np.std(lat_weighted_rmse),
        }

        rmse_vars = np.sqrt(mse_vars)
        lat_weighted_rmse_vars = np.sqrt(lat_weighted_mse_vars)

        for idx in range(mae_vars.shape[0]):
            var_name, level = FuXiDataset.get_var_name_and_level_at_idx(idx)
            results[f"mean.{var_name}.{level}.mae"] = np.mean(mae_vars[idx])
            results[f"std.{var_name}.{level}.mae"] = np.std(mae_vars[idx])
            results[f"mean.{var_name}.{level}.mse"] = np.mean(mse_vars[idx])
            results[f"std.{var_name}.{level}.mse"] = np.std(mse_vars[idx])
            results[f"mean.{var_name}.{level}.rmse"] = np.mean(rmse_vars[idx])
            results[f"std.{var_name}.{level}.rmse"] = np.std(rmse_vars[idx])
            results[f"mean.{var_name}.{level}.mae.lat_weighted"] = np.mean(
                lat_weighted_mae_vars[idx]
            )
            results[f"std.{var_name}.{level}.mae.lat_weighted"] = np.std(
                lat_weighted_mae_vars[idx]
            )
            results[f"mean.{var_name}.{level}.mse.lat_weighted"] = np.mean(
                lat_weighted_mse_vars[idx]
            )
            results[f"std.{var_name}.{level}.mse.lat_weighted"] = np.std(
                lat_weighted_mse_vars[idx]
            )
            results[f"mean.{var_name}.{level}.rmse.lat_weighted"] = np.mean(
                lat_weighted_rmse_vars[idx]
            )
            results[f"std.{var_name}.{level}.rmse.lat_weighted"] = np.std(
                lat_weighted_rmse_vars[idx]
            )
        self.write_result_to_file(results)

    def write_result_to_file(self, results):
        with open(self.results_file_path, "w+") as csvfile:
            csv_writer = csv.writer(
                csvfile, delimiter=";", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            csv_writer.writerow(
                [
                    "autoregression_steps",
                    "metric",
                    "mean/std",
                    "variable",
                    "level",
                    "lat_weighted",
                    "value",
                ]
            )
            for key, value in results.items():
                name_split = key.split(".")
                lat_weighted = len(name_split) == 5
                csv_writer.writerow(
                    [
                        self.autoregression_steps,
                        name_split[3],
                        name_split[0],
                        name_split[1],
                        name_split[2],
                        lat_weighted,
                        value,
                    ]
                )

    def load_model(self, path: os.PathLike | str):
        logger.info("Loading Model")
        self.model = FuXi.load_from_checkpoint(path)
        self.model.eval()
        self.model.autoregression_steps = self.autoregression_steps


if __name__ == "__main__":
    model_path = os.environ["MODEL_FILE"]
    data_path = os.environ["DATA_PATH"]
    mean_data_path = os.environ["MEAN_DATA_PATH"]
    autoregression_steps = int(os.environ["AUTOREGRESSION_STEPS"])
    output_path = os.environ["OUTPUT_PATH"]
    batch_size = int(os.environ["BATCH_SIZE"])

    dataset = FuXiDataset(
        data_path, mean_data_path, max_autoregression_steps=autoregression_steps
    )
    dataloader = DataLoader(
        dataset,
        **get_dataloader_params(batch_size),
    )

    model_evaluator = ModelEvaluator(
        model_path,
        dataloader,
        autoregression_steps,
        output_path,
    )
    model_evaluator.evaluate()
