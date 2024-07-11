import logging
import os

import dotenv
import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from pytorch_lightning.strategies import DDPStrategy

import wandb
from src.Dataset.fuxi_dataset import FuXiDataset
from src.PyModel.fuxi_ligthning import FuXi
from src.sweep_config import getSweepID
from src.utils import get_dataloader_params, config_epoch_to_autoregression_steps
from src.wandb_utils import get_optimizer_config

dotenv.load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

device = 'auto'
if torch.backends.mps.is_available():
    device = 'cpu'


def train():
    with wandb.init() as run:
        config = run.config

        logger.info('Creating Model')
        channels = config.get('model_parameter')['channel']
        transformer_blocks = config.get('model_parameter')['transformer_blocks']
        transformer_heads = config.get('model_parameter')['heads']
        optimizer_config = get_optimizer_config()

        model = FuXi(
            35, channels = channels,transformer_blocks= transformer_blocks,transformer_heads=transformer_heads,
            autoregression_steps_config= config.get('autoregression_steps_epochs'),
            train_ds_path="/Users/xgxtphg/Documents/git/FuXiClimatePrediction/data/1958_1958.zarr",
            train_mean_ds_path="/Users/xgxtphg/Documents/git/FuXiClimatePrediction/data/mean_1958_1958.zarr",
            val_ds_path="/Users/xgxtphg/Documents/git/FuXiClimatePrediction/data/1958_1958.zarr",
            val_mean_ds_path="/Users/xgxtphg/Documents/git/FuXiClimatePrediction/data/mean_1958_1958.zarr",
            batch_size=config.get('batch_size'),
            optimizer_config = optimizer_config,
        )

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
            callbacks=[
                checkpoint_callback,
                StochasticWeightAveraging(swa_lrs=1e-2)
            ],
            gradient_clip_val=0.5,
            reload_dataloaders_every_n_epochs=1,
            max_epochs=config.get('max_epochs', None)
        )

        trainer.fit(model=model)

        wandb_logger.experiment.unwatch(model)


if __name__ == '__main__':
    wandb.agent(getSweepID(), train)
