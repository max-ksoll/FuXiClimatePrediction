import logging
import os

import lightning as L
from torch.utils.data import DataLoader
from wandb import Config

from src.Dataset.create_data import DataBuilder
from src.Dataset.fuxi_dataset import FuXiDataset
from src.utils import get_dataloader_params

logger = logging.getLogger(__name__)


class FuXiDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: os.PathLike | str,
        start_year: int,
        end_year: int,
        val_start_year: int,
        val_end_year: int,
        test_start_year: int,
        test_end_year: int,
        config: Config,
        skip_data_preparing: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.start_year = start_year
        self.end_year = end_year
        self.val_start_year = val_start_year
        self.val_end_year = val_end_year
        self.test_start_year = test_start_year
        self.test_end_year = test_end_year
        self.train_ds_path = os.path.join(data_dir, f"{start_year}_{end_year}.zarr")
        self.train_mean_path = os.path.join(
            data_dir, f"mean_{start_year}_{end_year}.zarr"
        )
        self.val_ds_path = os.path.join(
            data_dir, f"{val_start_year}_{val_end_year}.zarr"
        )
        self.val_mean_path = os.path.join(
            data_dir, f"mean_{val_start_year}_{val_end_year}.zarr"
        )
        self.test_ds_path = os.path.join(
            data_dir, f"{val_start_year}_{val_end_year}.zarr"
        )
        self.test_mean_path = os.path.join(
            data_dir, f"mean_{val_start_year}_{val_end_year}.zarr"
        )
        self.batch_size = config.get("batch_size", 1)
        self.autoregression_steps_epoch = config.get("autoregression_steps_epochs")
        self.skip_data_preparing = skip_data_preparing
        self.train_ds = None
        self.val_ds = None

    def prepare_data(self):
        if self.skip_data_preparing:
            return
        logger.info("Creating Training Data")
        builder = DataBuilder(self.data_dir, self.start_year, self.end_year)
        builder.generate_data()

        logger.info("Creating Validation Data")
        builder = DataBuilder(self.data_dir, self.val_start_year, self.val_end_year)
        builder.generate_data()

        logger.info("Creating Test Data")
        builder = DataBuilder(self.data_dir, self.test_start_year, self.test_end_year)
        builder.generate_data()

    def setup(self, stage: str):
        ...

    def train_dataloader(self):
        return DataLoader(
            FuXiDataset(
                self.train_ds_path,
                self.train_mean_path,
                self.trainer.model.autoregression_steps,
            ),
            **get_dataloader_params(self.batch_size, is_train_dataloader=True),
        )

    def val_dataloader(self):
        return DataLoader(
            FuXiDataset(
                self.val_ds_path,
                self.val_mean_path,
                self.trainer.model.autoregression_steps,
            ),
            **get_dataloader_params(self.batch_size),
        )

    def test_dataloader(self):
        return DataLoader(
            FuXiDataset(
                self.test_ds_path,
                self.test_mean_path,
                self.trainer.model.autoregression_steps,
            ),
            **get_dataloader_params(self.batch_size),
        )

    def predict_dataloader(self):
        raise NotImplementedError()

    def teardown(self, stage: str):
        ...
