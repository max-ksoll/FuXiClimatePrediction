import logging
import os

import dotenv
import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy

import wandb
from src.Dataset.FuXiDataModule import FuXiDataModule
from src.FuXiEval import FuXiEvaluator

from src.ModelEvaluator import ModelEvaluator
from src.PyModel.fuxi_ligthning import FuXi
from src.sweep_config import getSweepID
from src.wandb_utils import get_optimizer_config, get_model_parameter

dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = 'auto'
if torch.backends.mps.is_available():
    device = 'cpu'

def init_model(run):
    config = run.config

    logger.info('Creating Model')
    model_parameter = get_model_parameter()
    channels = model_parameter['channel']
    transformer_blocks = model_parameter['transformer_blocks']
    transformer_heads = model_parameter['heads']
    optimizer_config = get_optimizer_config()
    model = FuXi(
        35, channels, transformer_blocks, transformer_heads,
        config.get('autoregression_steps_epochs'),
        optimizer_config=optimizer_config,
    )
    return model


def start_eval(run = None):
    if run is None:
        with wandb.init() as run:
            ckpt_path: str = "/Users/xgxtphg/Documents/git/FuXiClimatePrediction/models/epoch=169-step=65579.ckpt"
            eval = FuXiEvaluator(
                 model_path=ckpt_path,
                autoregression_config= run.config.get('autoregression_steps_epochs'),
                optimizer_config= get_optimizer_config()
            )


def train():
    with wandb.init() as run:
        config = run.config
        model = init_model(run)

        wandb_logger = WandbLogger(id=run.id, resume='allow')
        wandb_logger.watch(model, log_freq=100)
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.environ.get('MODEL_DIR', './models'),
            save_on_train_epoch_end=True,
            save_top_k=-1
        )

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

        data_dir = os.environ.get('DATA_PATH', False)
        if not data_dir:
            raise ValueError("DATA_PATH muss in dem .env File gesetzt sein!")

        dm = FuXiDataModule(
            data_dir=data_dir,
            start_year=1958,
            end_year=1990,
            val_start_year=1991,
            val_end_year=2000,
            config=config,
            skip_data_preparing=os.environ.get('SKIP_DATA_PREPARATION', False),
        )
        dm.val_dataloader()
        trainer.fit(model, datamodule=dm)
        wandb_logger.experiment.unwatch(model)
        evaluator = ModelEvaluator(
            model=model,
            run=run,
            dataset_type="test",
            dm=dm
        )
        evaluator.evaluate()


if __name__ == '__main__':
    wandb.agent(getSweepID(), start_eval)
