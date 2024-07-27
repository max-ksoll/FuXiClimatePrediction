import logging
import os

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy

import wandb
from src.Dataset.FuXiDataModule import FuXiDataModule
from src.PyModel.fuxi_ligthning import FuXi
from src.sweep_config import getSweepID
from src.wandb_utils import get_optimizer_config, get_model_parameter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = "auto"
if torch.backends.mps.is_available():
    device = "cpu"


def get_autoregression_step_epochs():
    return {
        "10": 1,  # until epoch n -> m autoregression step
        "20": 2,
        "30": 4,
        "40": 6,
        "50": 8,
        "-1": 10,
    }


def init_model():
    logger.info("Creating Model")
    model_parameter = get_model_parameter()
    # TODO warum so kompliziert?
    channels = 256
    transformer_blocks = 8
    transformer_heads = 8
    optimizer_config = {
        "optimizer_config_lr": 1e-5,
        "optimizer_config_betas": [(0.9, 0.95)],
        "optimizer_config_weight_decay": 0.1,
        "optimizer_config_T_0": 2,
        "optimizer_config_eta_min": 1e-7,
        "optimizer_config_T_mult": 2,
    }
    autoregression_step_epochs = get_autoregression_step_epochs()
    raw_fc = os.environ.get("RAW_FC_LAYER", "false").lower() == "true"
    model = FuXi(
        35,
        channels,
        transformer_blocks,
        transformer_heads,
        autoregression_step_epochs,
        optimizer_config=optimizer_config,
        fig_path=os.environ.get("FIG_PATH"),
        raw_fc_layer=raw_fc,
    )
    return model


def train():
    wandb_dir = os.environ.get("WANDB_DIR", None)
    with wandb.init(dir=wandb_dir, mode="offline") as run:
        model = init_model()

        wandb_logger = WandbLogger(id=run.id, resume="allow")
        wandb_logger.watch(model, log_freq=100)
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.environ.get("MODEL_DIR", "./models"),
            save_on_train_epoch_end=True,
            save_top_k=-1,
        )

        if os.environ.get("MULTI_GPU", False):
            strategy = DDPStrategy(find_unused_parameters=False)
            devices = os.environ.get("DEVICES", 1)
            num_nodes = os.environ.get("NODES", 1)
        else:
            strategy = "auto"
            devices = "auto"
            num_nodes = 1

        trainer = L.Trainer(
            accelerator=device,
            strategy=strategy,
            devices=devices,
            num_nodes=num_nodes,
            logger=wandb_logger,
            callbacks=[checkpoint_callback, StochasticWeightAveraging(swa_lrs=1e-2)],
            gradient_clip_val=0.5,
            reload_dataloaders_every_n_epochs=1,
            max_epochs=200,
        )

        data_dir = os.environ.get("DATA_PATH", False)
        if not data_dir:
            raise ValueError("DATA_PATH muss in dem .env File gesetzt sein!")

        skip_data_prep = (
            os.environ.get("SKIP_DATA_PREPARATION", "false").lower() == "true"
        )
        dm = FuXiDataModule(
            data_dir=data_dir,
            start_year=1958,
            end_year=2005,
            val_start_year=2006,
            val_end_year=2014,
            test_start_year=2006,
            test_end_year=2014,
            autoregression_step_epochs=get_autoregression_step_epochs(),
            skip_data_preparing=skip_data_prep,
            batch_size=1,
        )
        trainer.fit(model, datamodule=dm)
        wandb_logger.experiment.unwatch(model)


if __name__ == "__main__":
    train()
