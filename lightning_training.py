import logging
import os
from typing import Tuple

import dotenv
import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from pytorch_lightning.strategies import DDPStrategy

import wandb
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
        dataset_path="/Users/ksoll/git/FuXiClimatePrediction/data/1958_1958.zarr",
        means_file="/Users/ksoll/git/FuXiClimatePrediction/data/mean_1958_1958.zarr"
    )
    # TODO Test DS File muss erstellt und hier geladen werden
    # TODO dafür muss die Zeit Dim noch berichtigt werden
    test_ds = ERA5Dataset(
        dataset_path="/Users/ksoll/git/FuXiClimatePrediction/data/1958_1958.zarr",
        means_file="/Users/ksoll/git/FuXiClimatePrediction/data/mean_1958_1958.zarr"
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
        # TODO Der Klima mean muss jetzt einfach über den mean zarr file gebaut werden
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

        model = FuXi(35, channels, transformer_blocks, transformer_heads, lr, clima_mean)
        wandb_logger = WandbLogger(id=run.id, resume='allow')
        wandb_logger.watch(model, log_freq=100)
        checkpoint_callback = ModelCheckpoint(dirpath=os.environ.get('MODEL_DIR', './models'),
                                              save_on_train_epoch_end=True,
                                              save_top_k=-1)

        if dotenv.dotenv_values().get('MULTI_GPU', False):
            strategy = DDPStrategy(find_unused_parameters=False)
            devices = config.get('devices')
            num_nodes = config.get('num_nodes')
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
