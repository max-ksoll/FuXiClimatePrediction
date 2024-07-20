import os

import torch
from torch.utils.data import DataLoader

from src.Dataset.fuxi_dataset import FuXiDataset
from src.PyModel.fuxi_ligthning import FuXi
from src.wandb_utils import get_optimizer_config

import wandb


class FuXiEvaluator:
    def __init__(
        self,
        model_path: os.PathLike | str,
        optimizer_config,
        autoregression_config
        # dataset: FuXiDataset,
        # data_dir: os.PathLike | str,
        # start_year: int,
        # end_year: int,
        # dataset_path: os.PathLike | str,
        # dataset_mean_path: os.PathLike | str,
        # batch_size: int = 1,
    ):
        # if not model and not model_path:
        #     raise ValueError("Either a model or a model path is required.")
        # self.model = model
        self.model = torch.load(
            model_path,
            optimizer_config,
            autoregression_config,
            map_location=torch.device("cpu"),
        )
        # if dataset:
        #     self.dataset = dataset
        # elif data_dir and end_year and start_year:
        #     ds_path = os.path.join(data_dir, f"{start_year}_{end_year}.zarr")
        #     ds_mean_path = os.path.join(data_dir, f"mean_{start_year}_{end_year}.zarr")
        #     self.dataset = FuXiDataset()
        # self.batch_size = batch_size

    def evaluate(self):
        dataloader = DataLoader(
            dataset=self.dataset, batch_size=self.batch_size, shuffle=False
        )
        for batch in dataloader:
            metrics = self.model.validation_step(batch, ...)
            print(metrics)


if __name__ == "__main__":
    ckpt_path: str = "/Users/xgxtphg/Documents/git/FuXiClimatePrediction/models/epoch=169-step=65579.ckpt"
    eval = FuXiEvaluator(model_path=ckpt_path)
