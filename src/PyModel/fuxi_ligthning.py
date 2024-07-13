import logging
from typing import Any, List

import lightning as L

from src.PyModel.fuxi import FuXi as FuXiBase
from src.PyModel.score_torch import *
from src.utils import config_epoch_to_autoregression_steps

from src.global_vars import OPTIMIZER_REQUIRED_KEYS, LAT_DIM, LONG_DIM
from src.utils import config_epoch_to_autoregression_steps, get_dataloader_params

logger = logging.getLogger(__name__)


class FuXi(L.LightningModule):
    def __init__(
            self,
            input_vars: int,
            channels: int,
            transformer_blocks: int,
            transformer_heads: int,
            config: Dict[str, int],
            optimizer_config: Dict[str, Any],
    ):
        super().__init__()
        self.model: FuXiBase = FuXiBase(
            input_vars,
            channels,
            transformer_blocks,
            LAT_DIM,
            LONG_DIM,
            heads=transformer_heads,
        )
        self.autoregression_steps = config_epoch_to_autoregression_steps(
            config, 0
        )

        self.config = config
        self.optimizer_config = optimizer_config
        self.save_hyperparameters(
            "input_vars", "channels", "transformer_blocks", "transformer_heads"
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
        lat_weights = args[0][1]
        _, out = self.model.step(
            ts,
            lat_weights,
            autoregression_steps=self.autoregression_steps,
            return_out=True,
        )
        return out

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        ts, lat_weights = batch
        loss = self.model.step(
            ts, lat_weights, autoregression_steps=self.autoregression_steps
        )
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, _) -> Dict[str, torch.Tensor]:
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
            acc = weighted_acc(
                outs,
                label,
                lat_weights,
                self.trainer.train_dataloader.dataset.get_from_means_file("mean"),
            )
            self.log("val_acc", acc)
            ret_dict["acc"] = acc
        else:
            logger.warning("No Train Dataloader, skipping ACC Metric")

        mae = weighted_mae(outs, label, lat_weights)
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
