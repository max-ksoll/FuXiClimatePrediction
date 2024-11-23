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
            output_path, f"results_{self.autoregression_steps}_steps"
        )

    @torch.no_grad()
    def evaluate(self):
        mae = []
        mse = []
        rmse = []

        lat_weighted_mae = []
        lat_weighted_mse = []
        lat_weighted_rmse = []
        for x in iter(self.dataset):
            last_timestep = x[:, -1]
            mask = ~torch.isnan(last_timestep)

            out_last_timestep = self.model(x)[:, -1]
            error = last_timestep - out_last_timestep

            mae.append(float(torch.sum(torch.abs(error) * mask) / mask.sum()))
            mse.append(float(torch.sum(torch.abs(error**2) * mask) / mask.sum()))
            rmse.append(mse[-1] ** 0.5)

            lat_weighted_mae.append(
                float(
                    torch.sum(torch.abs(error) * self.lat_weights * mask) / mask.sum()
                )
            )
            lat_weighted_mse.append(
                float(
                    torch.sum(torch.abs(error**2) * self.lat_weights * mask)
                    / mask.sum()
                )
            )
            lat_weighted_rmse.append(lat_weighted_mse[-1] ** 0.5)

        results = {
            "mae": np.mean(mae),
            "mse": np.mean(mse),
            "rmse": np.mean(rmse),
            "lat_weighted_mae": np.mean(lat_weighted_mae),
            "lat_weighted_mse": np.mean(lat_weighted_mse),
            "lat_weighted_rmse": np.mean(lat_weighted_rmse),
        }
        self.write_result_to_file(results)

    def write_result_to_file(self, results):
        with open(self.results_file_path, "") as f:
            f.write(f"{self.autoregression_steps} Autoregression Steps\n")
            for key, value in results.items():
                f.write(f"{key}: {value}\n")

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
