import csv
import logging
import os
import sys

sys.path.append(os.environ["MODULE_PATH"])

import torch
import numpy as np
from torch.utils.data import DataLoader
import cartopy

from src.Dataset.dimensions import LAT
from src.utils import get_dataloader_params, get_latitude_weights
from src.Dataset.fuxi_dataset import FuXiDataset
from src.PyModel.fuxi_ligthning import FuXi


cartopy.config["pre_existing_data_dir"] = os.environ["CARTOPY_DIR"]
TASK_ID = int(os.environ.get("SLURM_ARRAY_TASK_ID", -1))
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
        mae = []
        mse = []

        lat_weighted_mae = []
        lat_weighted_mse = []
        for x in iter(self.dataset):
            last_timestep = x[:, -1]

            x = x.cuda()
            model_out = self.model(x, None).cpu()
            out_last_timestep = model_out[:, -1]

            error = last_timestep - out_last_timestep
            mask = ~torch.isnan(error)
            mask_sum = mask.sum(dim=[1, 2, 3])
            error = torch.nan_to_num(error, nan=0.0)

            mae.extend(
                list((torch.sum(torch.abs(error), dim=[1, 2, 3]) / mask_sum).numpy())
            )
            mse.extend(
                list(
                    (torch.sum(torch.abs(error**2), dim=[1, 2, 3]) / mask_sum).numpy()
                )
            )

            lat_weighted_mae.extend(
                list(
                    (
                        torch.sum(torch.abs(error) * self.lat_weights, dim=[1, 2, 3])
                        / mask_sum
                    ).numpy()
                )
            )
            lat_weighted_mse.extend(
                list(
                    (
                        torch.sum(
                            torch.abs(error**2) * self.lat_weights, dim=[1, 2, 3]
                        )
                        / mask_sum
                    ).numpy()
                )
            )

        rmse = list(map(lambda x: x**0.5, mse))
        lat_weighted_rmse = list(map(lambda x: x**0.5, lat_weighted_mse))

        results = {
            "mean.mae": np.mean(mae),
            "std.mae": np.std(mae),
            "mean.mse": np.mean(mse),
            "std.mse": np.std(mse),
            "mean.rmse": np.mean(rmse),
            "std.rmse": np.std(rmse),
            "mean.lat_weighted_mae": np.mean(lat_weighted_mae),
            "std.lat_weighted_mae": np.std(lat_weighted_mae),
            "mean.lat_weighted_mse": np.mean(lat_weighted_mse),
            "std.lat_weighted_mse": np.std(lat_weighted_mse),
            "mean.lat_weighted_rmse": np.mean(lat_weighted_rmse),
            "std.lat_weighted_rmse": np.std(lat_weighted_rmse),
        }
        self.write_result_to_file(results)

    def write_result_to_file(self, results):
        with open(self.results_file_path, "w+") as csvfile:
            csv_writer = csv.writer(
                csvfile, delimiter=";", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            csv_writer.writerow(["autoregression_steps", "metric", "mean/std", "value"])
            for key, value in results.items():
                name_split = key.split(".")
                csv_writer.writerow(
                    [self.autoregression_steps, name_split[-1], name_split[0], value]
                )

    def load_model(self, path: os.PathLike | str):
        logger.info("Loading Model")
        self.model = FuXi.load_from_checkpoint(path)
        self.model.eval()
        self.model.autoregression_steps = self.autoregression_steps + 2


if __name__ == "__main__":
    model_path = os.environ["MODEL_FILE"]
    data_path = os.environ["DATA_PATH"]
    mean_data_path = os.environ["MEAN_DATA_PATH"]
    autoregression_steps = int(os.environ["AUTOREGRESSION_STEPS"])
    output_path = os.environ["OUTPUT_PATH"]
    batch_size = int(os.environ["BATCH_SIZE"])

    dataset = FuXiDataset(data_path, mean_data_path)
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
