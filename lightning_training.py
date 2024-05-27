import logging
import os
from typing import Tuple

import dotenv
import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import wandb
from src.era5_dataset import ERA5Dataset, TimeMode
from src.fuxi_ligthning import FuXi
from src.sweep_config import getSweepID

dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = 'auto'
if torch.backends.mps.is_available():
    device = 'cpu'


def create_train_test_datasets(batch_size, max_autoregression_steps) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    logger.info('Creating Dataset')
    col_names = os.environ.get('COL_NAMES', 'lessig')
    train_ds = ERA5Dataset(
        os.environ.get('DATAFOLDER'),
        TimeMode.BETWEEN,
        start_time="1979-01-01T00:00:00",
        end_time="2019-12-31T18:00:00",
        max_autoregression_steps=max_autoregression_steps,
        zarr_col_names=col_names
    )
    test_ds = ERA5Dataset(
        os.environ.get('DATAFOLDER'),
        TimeMode.BETWEEN,
        start_time="2020-01-01T00:00:00",
        end_time="2020-12-31T18:00:00",
        max_autoregression_steps=max_autoregression_steps,
        zarr_col_names=col_names
    )
    train_loader_params = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': os.cpu_count() // 2,
        'pin_memory': True
    }
    val_loader_params = {
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': os.cpu_count() // 2,
        'pin_memory': True
    }
    logger.info('Creating DataLoader')
    train_dl = DataLoader(train_ds, **train_loader_params)
    test_dl = DataLoader(test_ds, **val_loader_params)
    return train_dl, test_dl, train_ds.get_latitude_weights()


def get_autoregression_steps(autoregression_steps_epochs, epoch):
    smaller_values = [value for value in autoregression_steps_epochs.keys() if int(value) <= epoch]
    if not smaller_values:
        return 1

    return autoregression_steps_epochs[str(max(smaller_values))]


def train():
    with wandb.init() as run:
        config = run.config

        logger.info('Loading Clima Mean')
        clima_mean_dir = dotenv.dotenv_values().get('CLIMA_MEAN_DIR', "")
        clima_mean = None
        if os.path.exists(clima_mean_dir):
            # Shape (vars x lats x longs)
            clima_mean = torch.flatten(torch.stack([
                torch.load(os.path.join(clima_mean_dir, "temperature.pt")),
                torch.load(os.path.join(clima_mean_dir, "specific_humidity.pt")),
                torch.load(os.path.join(clima_mean_dir, "u_component_of_wind.pt")),
                torch.load(os.path.join(clima_mean_dir, "v_component_of_wind.pt")),
                torch.load(os.path.join(clima_mean_dir, "geopotential.pt"))
            ], 0), 0, 1)

        logger.info('Creating Model')
        channels = config.get('model_parameter')['channel']
        transformer_blocks = config.get('model_parameter')['transformer_blocks']
        transformer_heads = config.get('model_parameter')['heads']
        lr = config.get("init_learning_rate")

        model = FuXi(25, channels, transformer_blocks, transformer_heads, lr, clima_mean)
        wandb_logger = WandbLogger(id=run.id, resume='allow')
        wandb_logger.watch(model, log_freq=100)
        checkpoint_callback = ModelCheckpoint(dirpath=os.environ.get('MODEL_DIR', './models'),
                                              save_on_train_epoch_end=True,
                                              save_top_k=-1)
        trainer = L.Trainer(
            accelerator=device,
            logger=wandb_logger,
            callbacks=[checkpoint_callback,
                       StochasticWeightAveraging(swa_lrs=1e-2)],
            gradient_clip_val=0.5
        )

        epochs = 0
        for item in config.get('autoregression_steps_epochs'):
            autoregression_steps = item.get('steps')
            train_dl, test_dl, lat_weights = create_train_test_datasets(config.get('batch_size', 1),
                                                                        autoregression_steps)

            model.set_autoregression_steps(autoregression_steps)

            epochs += item.get('epochs')
            trainer.fit_loop.max_epochs = epochs
            trainer.fit(
                model=model,
                train_dataloaders=train_dl,
                val_dataloaders=test_dl
            )

        wandb_logger.experiment.unwatch(model)


if __name__ == '__main__':
    wandb.agent(getSweepID(), train)
