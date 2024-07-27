import logging
from typing import Any, Dict

import lightning as L
import torch

from src.Dataset.dimensions import LAT, LON
from src.Eval.ModelEvaluator import ModelEvaluator
from src.PyModel.fuxi import FuXi as FuXiBase
from src.global_vars import OPTIMIZER_REQUIRED_KEYS
from src.utils import config_epoch_to_autoregression_steps, log_exec_time
from src.utils import get_latitude_weights
from src.wandb_utils import log_eval_dict

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
        fig_path: str,
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
        self.valModelEvaluator = None
        self.testModelEvaluator = None
        self.autoregression_steps = config_epoch_to_autoregression_steps(
            autoregression_config, 0
        )

        self.config = autoregression_config
        self.optimizer_config = optimizer_config
        self.lat_weights = get_latitude_weights(LAT).to(self.device)
        self.save_hyperparameters(
            "input_vars",
            "channels",
            "transformer_blocks",
            "transformer_heads",
            "raw_fc_layer",
        )
        self.fig_path = fig_path

    def on_train_epoch_end(self) -> None:
        if self.current_epoch == 0:
            self.valModelEvaluator = ModelEvaluator(
                self.trainer.train_dataloader.dataset.get_clima_mean(),
                self.lat_weights,
                self.trainer.val_dataloaders,
                self.fig_path,
            )
            self.testModelEvaluator = ModelEvaluator(
                self.trainer.train_dataloader.dataset.get_clima_mean(),
                self.lat_weights,
                self.trainer.test_dataloaders,
                self.fig_path,
            )
        old_auto_steps = self.autoregression_steps
        if (
            config_epoch_to_autoregression_steps(self.config, self.current_epoch)
            != old_auto_steps
        ):
            logger.debug("End of Train Epoch: Setting new Autoregression steps")
            self.autoregression_steps = config_epoch_to_autoregression_steps(
                self.config, self.current_epoch
            )
        self.modelEvaluator.set_dl(self.trainer.val_dataloaders)

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
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        return loss

    @log_exec_time
    def validation_step(self, batch, batch_index) -> None:
        returns = self.model.step(
            batch,
            self.lat_weights,
            autoregression_steps=self.autoregression_steps,
        )
        self.valModelEvaluator.update(returns["output"], batch_index)
        self.log("val_loss", returns["loss"])

    @log_exec_time
    def on_validation_epoch_end(self) -> Dict[str, torch.Tensor]:
        if self.trainer.is_global_zero:
            model_eval = self.valModelEvaluator.evaluate()
            log_eval_dict(model_eval, "val")
            return model_eval

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

    @log_exec_time
    def test_step(self, batch, batch_index) -> None:
        returns = self.model.step(
            batch,
            self.lat_weights,
            autoregression_steps=self.autoregression_steps,
        )
        self.testModelEvaluator.update(returns["output"], batch_index)
        self.log("val_loss", returns["loss"])

    @log_exec_time
    def on_test_epoch_end(self) -> Dict[str, torch.Tensor]:
        if self.trainer.is_global_zero:
            model_eval = self.testModelEvaluator.evaluate()
            log_eval_dict(model_eval, "test")
            return model_eval
