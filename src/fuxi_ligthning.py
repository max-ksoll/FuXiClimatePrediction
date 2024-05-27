from typing import Any

import pytorch_lightning as L

from src.fuxi import FuXi as FuXiBase
from src.score_torch import *
import torch


class FuXi(L.LightningModule):
    def __init__(self, input_vars, channels, transformer_blocks, transformer_heads, lr, clima_mean):
        super().__init__()
        self.model: FuXiBase = FuXiBase(
            input_vars,
            channels,
            transformer_blocks,
            121, 240,
            heads=transformer_heads,
        )
        self.lr = lr
        self.CLIMA_MEAN = clima_mean
        self.autoregression_steps = 1
        self.save_hyperparameters()

    def set_autoregression_steps(self, autoregression_steps):
        self.autoregression_steps = autoregression_steps

    def set_lr(self, lr):
        self.lr = lr

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        ts = args[0][0]
        lat_weights = args[0][1]
        loss, out = self.model.step(ts, lat_weights, autoregression_steps=self.autoregression_steps, return_out=True)
        return out

    def training_step(self, batch, batch_idx):
        ts, lat_weights = batch
        loss = self.model.step(ts, lat_weights, autoregression_steps=self.autoregression_steps)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        ts, lat_weights = batch
        label = torch.clone(ts[:, 2:, :, :, :])

        loss, outs = self.model.step(ts, lat_weights, autoregression_steps=self.autoregression_steps, return_out=True)
        self.log('val_loss', loss)

        ret_dict = dict()
        ret_dict['loss'] = loss

        rmse = compute_weighted_rmse(outs.cpu(), label.cpu(), lat_weights.cpu())
        self.log('val_rmse', rmse)
        ret_dict['rmse'] = rmse

        if self.CLIMA_MEAN is not None:
            acc = compute_weighted_acc(outs.cpu(), label.cpu(), lat_weights.cpu(), self.CLIMA_MEAN)
            self.log('val_acc', acc)
            ret_dict['acc'] = acc

        mae = compute_weighted_mae(outs.cpu(), label.cpu(), lat_weights.cpu())
        self.log('val_mae', mae)
        ret_dict['mae'] = mae

        return ret_dict

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=2,
            T_mult=2,
            eta_min=1e-7,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
