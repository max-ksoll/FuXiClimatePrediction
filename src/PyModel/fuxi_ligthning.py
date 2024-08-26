import gc
import logging
import os
from typing import Any, Dict

import lightning as L
import torch
from typing_extensions import Self

from src.Dataset.dimensions import LAT, LON
from src.Eval.scores import weighted_acc, weighted_mae, weighted_rmse
from src.PyModel.fuxi import FuXi as FuXiBase
from src.global_vars import OPTIMIZER_REQUIRED_KEYS
from src.utils import config_epoch_to_autoregression_steps, log_exec_time
from src.utils import get_latitude_weights

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
        tensor_path: str,
        clima_mean: torch.Tensor,
        raw_fc_layer: bool = False,
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
        self.tensor_path = tensor_path
        os.makedirs(tensor_path, exist_ok=True)
        self.clima_mean = clima_mean

        self.val_diff_to_gt = []
        self.val_diff_to_clim = []
        self.autoregression_steps_plots = [
            0,
            2,
            4,
            6,
            9,
        ]

    def on_train_epoch_start(self) -> None:
        old_auto_steps = self.autoregression_steps
        if (
            config_epoch_to_autoregression_steps(self.config, self.current_epoch)
            != old_auto_steps
        ):
            logger.debug("End of Train Epoch: Setting new Autoregression steps")
            self.autoregression_steps = config_epoch_to_autoregression_steps(
                self.config, self.current_epoch
            )

    def to(self, *args: Any, **kwargs: Any) -> Self:
        super().to(*args, **kwargs)
        self.lat_weights = self.lat_weights.to(*args, **kwargs)
        self.clima_mean = self.clima_mean.to(*args, **kwargs)
        return self

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
        self.log("train_loss", loss, sync_dist=True)
        if self.trainer.is_global_zero:
            self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_index) -> None:
        returns = self.model.step(
            batch,
            self.lat_weights,
            autoregression_steps=self.autoregression_steps,
        )

        self.val_diff_to_gt.append((returns["output"] - batch[:, 2:]).cpu())
        self.val_diff_to_clim.append((returns["output"] - self.clima_mean).cpu())

        if self.clima_mean is not None:
            acc = weighted_acc(
                returns["output"], batch[:, 2:], self.lat_weights, self.clima_mean
            )
            self.log("val_acc", acc, sync_dist=True)

        mae = weighted_mae(returns["output"], batch[:, 2:], self.lat_weights)
        rmse = weighted_rmse(returns["output"], batch[:, 2:], self.lat_weights)

        self.log("val_loss", returns["loss"], sync_dist=True)
        self.log("val_mae", mae, sync_dist=True)
        self.log("val_rmse", rmse, sync_dist=True)

    @torch.no_grad()
    def on_validation_end(self) -> None:
        if not self.trainer.is_global_zero:
            return

        if len(self.val_diff_to_gt) == 0 or len(self.val_diff_to_clim) == 0:
            logger.warning("Skipping val_end because of an empty list - clearing...")
            self.val_diff_to_gt.clear()
            self.val_diff_to_clim.clear()
            return

        epoch_path = os.path.join(self.tensor_path, str(self.current_epoch))
        os.makedirs(epoch_path, exist_ok=True)

        diff_tensor = torch.cat(self.val_diff_to_gt, dim=0)
        diff_tensor = diff_tensor.nanmean(dim=0)
        torch.save(
            diff_tensor,
            os.path.join(epoch_path, "diff.pt"),
        )

        model_minus_clim = torch.cat(self.val_diff_to_clim, dim=0)
        model_minus_clim = model_minus_clim.nanmean(dim=[0, 1])
        torch.save(
            model_minus_clim,
            os.path.join(epoch_path, "clim.pt"),
        )

        self.val_diff_to_gt.clear()
        self.val_diff_to_clim.clear()
        gc.collect()

    @log_exec_time
    def test_step(self, batch, batch_index) -> None:
        returns = self.model.step(
            batch,
            self.lat_weights,
            autoregression_steps=self.autoregression_steps,
        )
        if self.clima_mean is not None:
            acc = weighted_acc(
                returns["output"], batch[:, 2:], self.lat_weights, self.clima_mean
            )
            self.log("test_acc", acc, sync_dist=True)

        mae = weighted_mae(returns["output"], batch[:, 2:], self.lat_weights)
        rmse = weighted_rmse(returns["output"], batch[:, 2:], self.lat_weights)

        self.log("test_loss", returns["loss"], sync_dist=True)
        self.log("test_mae", mae, sync_dist=True)
        self.log("test_rmse", rmse, sync_dist=True)

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
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer,
        #     T_0=self.optimizer_config["optimizer_config_T_0"],
        #     T_mult=self.optimizer_config["optimizer_config_T_mult"],
        #     eta_min=self.optimizer_config["optimizer_config_eta_min"],
        # )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.optimizer_config["optimizer_config_T_max"],
            eta_min=self.optimizer_config["optimizer_config_eta_min"],
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
