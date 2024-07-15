import os

from torch.utils.data import DataLoader

from src.Dataset.fuxi_dataset import FuXiDataset
from src.PyModel.fuxi_ligthning import FuXi


class FuXiEvaluator:
    def __init__(
            self,
            model: FuXi,
            model_path: os.PathLike | str,
            dataset: FuXiDataset,
            data_dir: os.PathLike | str,
            start_year: int,
            end_year: int,
            dataset_path: os.PathLike | str,
            dataset_mean_path: os.PathLike | str,
            batch_size: int = 1
    ):
        if not model and not model_path:
            raise ValueError("Either a model or a model path is required.")
        self.model = model
        if not self.model:
            self.model = FuXi.load_from_checkpoint(model_path)
        if dataset:
            self.dataset = dataset
        elif data_dir and end_year and start_year:
            ds_path = os.path.join(data_dir, f"{start_year}_{end_year}.zarr")
            ds_mean_path = os.path.join(data_dir, f"mean_{start_year}_{end_year}.zarr")
            self.dataset = FuXiDataset(

            )
        self.batch_size = batch_size

    def evaluate(self):
        dataloader = DataLoader(
            dataset=self.dataset, batch_size=self.batch_size, shuffle=False
        )
        for batch in dataloader:
            metrics = self.model.validation_step(batch, ...)
            print(metrics)


if __name__ == "__main__":
    ckpt_path: str = "/Users/ksoll/git/FuXiClimatePrediction/models/epoch=0-step=9.ckpt"
    eval = FuXiEvaluator(model_path=ckpt_path, )
