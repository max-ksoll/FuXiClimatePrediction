from typing import Any

import pytorch_lightning as L

from src.fuxi import FuXi as FuXiBase
from src.score_torch import *
import torch

class FuXi(L.LightningModule):
    def __init__(
        self,
        input_vars,
        channels,
        transformer_blocks,
        transformer_heads,
        lr,
        T_0,
        T_mult,
        eta_min,
        clima_mean,
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
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.CLIMA_MEAN = clima_mean
        self.autoregression_steps = 1
        self.save_hyperparameters()
        # Register climate mean as buffer to ensure it is moved to the correct device
        self.register_buffer("CLIMA_MEAN", torch.tensor(clima_mean))

    def set_autoregression_steps(self, autoregression_steps):
        self.autoregression_steps = autoregression_steps

    def set_lr(self, lr, T_0,T_mult,eta_min):
        self.lr = lr
        self.T_0 = T_0,
        self.T_mult = T_mult,
        self.eta_min = eta_min,

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
        score = Score()
        rmse = score.compute_weighted_rmse(outs, label, lat_weights)
        self.log("val_rmse", rmse)
        ret_dict["rmse"] = rmse

        if self.CLIMA_MEAN is not None:
            acc = score.compute_weighted_acc(outs, label, self.CLIMA_MEAN, lat_weights)
            self.log("val_acc", acc)
            ret_dict["acc"] = acc

        mae = score.compute_weighted_mae(outs, label, lat_weights)
        self.log("val_mae", mae)
        ret_dict["mae"] = mae

        return ret_dict

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, betas=(0.9, 0.95), weight_decay=0.1
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.T_0,
            T_mult=self.T_mult,
            eta_min=self.eta_min,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
