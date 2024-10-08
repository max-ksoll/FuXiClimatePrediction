import logging
import os

import dotenv
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy

import wandb
from src.Dataset.FuXiDataModule import FuXiDataModule
from src.PyModel.fuxi_ligthning import FuXi
from src.sweep_config import getSweepID
from src.utils import get_clima_mean
from src.wandb_utils import get_optimizer_config, get_model_parameter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


dotenv.load_dotenv()


device = "auto"
if torch.backends.mps.is_available():
    device = "cpu"


def init_model(run, data_dir, start_year, end_year):
    config = run.config

    logger.info("Creating Model")
    model_parameter = get_model_parameter()
    # TODO warum so kompliziert?
    channels = model_parameter["model_parameter_channel"]
    transformer_blocks = model_parameter["model_parameter_transformer_blocks"]
    transformer_heads = model_parameter["model_parameter_heads"]
    optimizer_config = get_optimizer_config()
    raw_fc = os.environ.get("RAW_FC_LAYER", "false").lower() == "true"
    clim = get_clima_mean(
        dataset_path=os.path.join(data_dir, f"{start_year}_{end_year}.zarr"),
        means_file=os.path.join(data_dir, f"mean_{start_year}_{end_year}.zarr"),
    )
    model = FuXi(
        35,
        channels,
        transformer_blocks,
        transformer_heads,
        autoregression_config=config.get("autoregression_steps_epochs"),
        optimizer_config=optimizer_config,
        fig_path=os.environ.get("FIG_PATH"),
        clima_mean=clim,
        raw_fc_layer=raw_fc,
    )
    return model


def train():
    with wandb.init() as run:
        config = run.config
        data_dir = os.environ.get("DATA_PATH", False)
        if not data_dir:
            raise ValueError("DATA_PATH muss in dem .env File gesetzt sein!")
        model = init_model(run, data_dir, 1958, 1960)

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
            max_epochs=config.get("max_epochs", None),
        )

        skip_data_prep = (
            os.environ.get("SKIP_DATA_PREPARATION", "false").lower() == "true"
        )
        dm = FuXiDataModule(
            data_dir=data_dir,
            start_year=1958,
            end_year=1959,
            val_start_year=1960,
            val_end_year=1960,
            test_start_year=1960,
            test_end_year=1960,
            config=config,
            skip_data_preparing=skip_data_prep,
        )
        trainer.fit(model, datamodule=dm)
        wandb_logger.experiment.unwatch(model)


if __name__ == "__main__":
    wandb.agent(getSweepID(), train)
