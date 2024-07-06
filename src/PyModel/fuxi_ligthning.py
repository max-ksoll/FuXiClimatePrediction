import logging
import os
from typing import Any

import pytorch_lightning as L
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader

from src.Dataset.fuxi_dataset import FuXiDataset
from src.PyModel.fuxi import FuXi as FuXiBase
from src.PyModel.score_torch import *
import torch

from src.utils import config_epoch_to_autoregression_steps, get_dataloader_params

logger = logging.getLogger(__name__)


class FuXi(L.LightningModule):
    def __init__(
            self,
            input_vars: int,
            channels: int,
            transformer_blocks: int,
            transformer_heads: int,
            lr: float,
            config: Dict[str, int],
            train_ds_path: os.PathLike | str,
            train_mean_ds_path: os.PathLike | str,
            val_ds_path: os.PathLike | str,
            val_mean_ds_path: os.PathLike | str,
            batch_size: int = 1,
    ):
        super().__init__()
        self.model: FuXiBase = FuXiBase(
            input_vars,
            channels,
            transformer_blocks,
            121,
            240,
            heads=transformer_heads,
        )
        self.lr = lr
        self.autoregression_steps = config_epoch_to_autoregression_steps(config, 0)

        self.train_ds_path = train_ds_path
        self.train_mean_ds_path = train_mean_ds_path
        self.val_ds_path = val_ds_path
        self.val_mean_ds_path = val_mean_ds_path

        self.batch_size = batch_size
        self.config = config
        self.save_hyperparameters(
            'input_vars', 'channels', 'transformer_blocks', 'transformer_heads'
        )
        # Register climate mean as buffer to ensure it is moved to the correct device
        # self.register_buffer("CLIMA_MEAN", clima_mean)

    def on_train_epoch_end(self) -> None:
        old_auto_steps = self.autoregression_steps
        if config_epoch_to_autoregression_steps(self.config, self.current_epoch) != old_auto_steps:
            logger.debug("End of Train Epoch: Setting new Autoregression steps")
            self.autoregression_steps = config_epoch_to_autoregression_steps(self.config, self.current_epoch)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        logger.debug("Setting Train Dataloader")
        return DataLoader(
            FuXiDataset(self.train_ds_path, self.train_mean_ds_path, self.autoregression_steps),
            **get_dataloader_params(self.batch_size)
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        logger.debug("Setting Val Dataloader")
        return DataLoader(
            FuXiDataset(self.val_ds_path, self.val_mean_ds_path, self.autoregression_steps),
            **get_dataloader_params(self.batch_size)
        )

    def set_lr(self, lr):
        self.lr = lr

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        ts = args[0][0]
        lat_weights = args[0][1]
        loss, out = self.model.step(
            ts,
            lat_weights,
            autoregression_steps=self.autoregression_steps,
            return_out=True,
        )
        return out

    def training_step(self, batch, batch_idx):
        ts, lat_weights = batch
        loss = self.model.step(
            ts, lat_weights, autoregression_steps=self.autoregression_steps
        )
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        ts, lat_weights = batch
        label = torch.clone(ts[:, 2:, :, :, :])

        loss, outs = self.model.step(
            ts,
            lat_weights,
            autoregression_steps=self.autoregression_steps,
            return_out=True,
        )
        self.log("val_loss", loss)

        ret_dict = dict()
        ret_dict["loss"] = loss

        rmse = weighted_rmse(outs, label, lat_weights)
        self.log("val_rmse", rmse)
        ret_dict["rmse"] = rmse

        if self.trainer.train_dataloader is not None:
            acc = weighted_acc(outs, label, lat_weights,
                               self.trainer.train_dataloader.dataset.get_from_means_file('mean'))
            self.log("val_acc", acc)
            ret_dict["acc"] = acc
        else:
            logger.warning("No Train Dataloader, skipping ACC Metric")

        mae = weighted_mae(outs, label, lat_weights)
        self.log("val_mae", mae)
        ret_dict["mae"] = mae

        return ret_dict

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, betas=(0.9, 0.95), weight_decay=0.1
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=2,
            T_mult=2,
            eta_min=1e-7,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
