import logging
from typing import Any, Dict

import lightning as L
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT

from src.Dataset.dimensions import LAT, LON
from src.Eval.scores import weighted_rmse, weighted_acc, weighted_mae
from src.PyModel.fuxi import FuXi as FuXiBase
from src.global_vars import OPTIMIZER_REQUIRED_KEYS
from src.utils import config_epoch_to_autoregression_steps, get_latitude_weights

logger = logging.getLogger(__name__)


class FuXi(L.LightningModule):
    def __init__(
        self,
        input_vars: int,
        channels: int,
        transformer_blocks: int,
        transformer_heads: int,
        autoregression_config: Dict[str, int],
        optimizer_config: Dict[str, Any],
        raw_fc_layer=False,
    ):
        super().__init__()
        self.model: FuXiBase = FuXiBase(
            input_vars,
            channels,
            transformer_blocks,
            LAT.size,
            LON.size,
            heads=transformer_heads,
            raw_fc_layer=raw_fc_layer,
        )
        self.autoregression_steps = config_epoch_to_autoregression_steps(
            autoregression_config, 0
        )

        self.config = autoregression_config
        self.optimizer_config = optimizer_config
        self.lat_weights = get_latitude_weights(LAT)
        self.save_hyperparameters(
            "input_vars",
            "channels",
            "transformer_blocks",
            "transformer_heads",
            "raw_fc_layer",
        )

    def on_train_epoch_end(self) -> None:
        old_auto_steps = self.autoregression_steps
        if (
            config_epoch_to_autoregression_steps(self.config, self.current_epoch)
            != old_auto_steps
        ):
            logger.debug("End of Train Epoch: Setting new Autoregression steps")
            self.autoregression_steps = config_epoch_to_autoregression_steps(
                self.config, self.current_epoch
            )

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        ts = args[0][0]
        out = self.model.step(
            ts,
            self.lat_weights,
            autoregression_steps=self.autoregression_steps,
            return_out=True,
            return_loss=False,
        )["output"]
        return out

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss = self.model.step(
            batch,
            self.lat_weights,
            autoregression_steps=self.autoregression_steps,
            return_loss=True,
            return_out=False,
        )["loss"]
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, _) -> Dict[str, torch.Tensor]:
        label = torch.clone(batch[:, 2:, :, :, :])

        returns = self.model.step(
            batch,
            self.lat_weights,
            autoregression_steps=self.autoregression_steps,
        )
        loss = returns["loss"]
        outs = returns["output"]

        self.log("val_loss", loss)

        ret_dict = dict()
        ret_dict["loss"] = loss

        rmse = weighted_rmse(outs, label, self.lat_weights)
        self.log("val_rmse", rmse)
        ret_dict["rmse"] = rmse

        if self.trainer.train_dataloader is not None:
            acc = weighted_acc(
                outs,
                label,
                self.lat_weights,
                self.trainer.train_dataloader.dataset.get_clima_mean().to(outs),
            )
            self.log("val_acc", acc)
            ret_dict["acc"] = acc
        else:
            logger.warning("No Train Dataloader, skipping ACC Metric")

        mae = weighted_mae(outs, label, self.lat_weights)
        self.log("val_mae", mae)
        ret_dict["mae"] = mae

        return ret_dict

    def configure_optimizers(self):
        for key in OPTIMIZER_REQUIRED_KEYS:
            if key not in self.optimizer_config:
                logger.error(f"Optimizer config is missing '{key}'")
                raise KeyError(f"Optimizer config is missing '{key}'")

        logger.debug("Setting Optimizer Values")
        betas = self.optimizer_config["optimizer_config_betas"][0]
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.optimizer_config["optimizer_config_lr"],
            betas=(betas[0], betas[1]),
            weight_decay=self.optimizer_config["optimizer_config_weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.optimizer_config["optimizer_config_T_0"],
            T_mult=self.optimizer_config["optimizer_config_T_mult"],
            eta_min=self.optimizer_config["optimizer_config_eta_min"],
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        ts, lat_weights = args
