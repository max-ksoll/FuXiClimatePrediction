import logging
import os

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy

import wandb
from src.Dataset.FuXiDataModule import FuXiDataModule
from src.PyModel.fuxi_ligthning import FuXi
from src.utils import get_clima_mean

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("medium")


def get_config():
    config = dict()
    config["autoregression_step_epochs"] = get_autoregression_step_epochs()
    config["model_parameter"] = get_model_parameter()
    config["opt_config"] = get_opt_config()


def get_autoregression_step_epochs():
    return {
        "25": 2,
        "50": 4,
        "75": 6,
        "100": 8,
        "125": 10,
        "150": 12,
        "-1": 24,
    }


def get_model_parameter():
    return {"channels": 2048, "transformer_blocks": 40, "transformer_heads": 16}


def get_opt_config():
    return {
        "optimizer_config_lr": 1e-5,
        "optimizer_config_betas": [(0.9, 0.95)],
        "optimizer_config_weight_decay": 0.1,
        "optimizer_config_T_max": 200,
        "optimizer_config_eta_min": 1e-8,
    }


def init_model(data_dir: os.PathLike | str, start_year: int, end_year: int):
    logger.info("Creating Model")
    # TODO warum so kompliziert?
    optimizer_config = get_opt_config()
    autoregression_step_epochs = get_autoregression_step_epochs()
    raw_fc = os.environ.get("RAW_FC_LAYER", "false").lower() == "true"
    clim = get_clima_mean(
        dataset_path=os.path.join(data_dir, f"{start_year}_{end_year}.zarr"),
        means_file=os.path.join(data_dir, f"mean_{start_year}_{end_year}.zarr"),
    )
    model_params = get_model_parameter()
    model = FuXi(
        35,
        model_params["channels"],
        model_params["transformer_blocks"],
        model_params["transformer_heads"],
        autoregression_step_epochs,
        optimizer_config=optimizer_config,
        fig_path=os.environ.get("FIG_PATH"),
        raw_fc_layer=raw_fc,
        clima_mean=clim,
    )
    return model


def train():
    wandb_dir = os.environ.get("WANDB_DIR", None)
    with wandb.init(
        project="FuXiClimatePrediction",
        dir=wandb_dir,
        mode="offline",
        config=get_config(),
    ) as run:
        data_dir = os.environ.get("DATA_PATH", False)
        if not data_dir:
            raise ValueError("DATA_PATH muss in dem .env File gesetzt sein!")

        model = init_model(data_dir, 1958, 2005)

        wandb_logger = WandbLogger(id=run.id, resume="allow")
        wandb_logger.watch(model, log_freq=100)
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.environ.get("MODEL_DIR", "./models"),
            save_on_train_epoch_end=True,
            save_top_k=-1,
        )

        multi_gpu = os.environ.get("MULTI_GPU", "false") == "true"
        # TODO hier könnte man mal über fdsp nachdenken
        # vor allem, wenn wir auf den voll skalierten Daten trainieren könnte das sinvoll werden
        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.strategies.FSDPStrategy.html
        strategy = DDPStrategy(find_unused_parameters=False) if multi_gpu else "auto"
        num_nodes = os.environ.get("NODES", 1)

        trainer = L.Trainer(
            accelerator="auto",
            precision="16-mixed",
            strategy=strategy,
            devices=-1,
            num_nodes=num_nodes,
            logger=wandb_logger,
            callbacks=[checkpoint_callback, StochasticWeightAveraging(swa_lrs=1e-2)],
            gradient_clip_val=0.5,
            reload_dataloaders_every_n_epochs=1,
            max_epochs=1000,
        )

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
