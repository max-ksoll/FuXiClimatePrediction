from datetime import datetime
import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import wandb
from src.score_torch import compute_weighted_rmse, compute_weighted_acc, compute_weighted_mae
from src.sweep_config import getSweepID
from src.fuxi import FuXi
from src.era5_dataset import ERA5Dataset, TimeMode
import logging
from tqdm import tqdm
import dotenv

dotenv.load_dotenv()

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
if torch.backends.mps.is_available():
    device = 'cpu'

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def create_train_test_datasets(max_autoregression_steps) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    logger.info('Creating Dataset')
    train_ds = ERA5Dataset(
        os.environ.get('DATAFOLDER'),
        max_autoregression_steps,
        TimeMode.BEFORE,
        end_time="2011-01-11-30T18:00:00",
        max_autoregression_steps=max_autoregression_steps
    )
    test_ds = ERA5Dataset(
        os.environ.get('DATAFOLDER'),
        max_autoregression_steps,
        TimeMode.BETWEEN,
        start_time="2011-01-12-01T00:00:00",
        end_time="2011-01-12-31T18:00:00",
        max_autoregression_steps=max_autoregression_steps
    )
    loader_params = {'batch_size': None,
                     'batch_sampler': None,
                     'shuffle': False,
                     'num_workers': os.cpu_count() // 2,
                     'pin_memory': True}
    logger.info('Creating DataLoader')
    train_dl = DataLoader(train_ds, **loader_params, sampler=None)
    test_dl = DataLoader(test_ds, **loader_params, sampler=None)
    return train_dl, test_dl, train_ds.get_latitude_weights()


def train_epoch(model, optimizer, train_loader, autoregression_steps, epoch):
    model.train()
    whole_loss = []
    pbar = tqdm(train_loader, desc=f'Epoch: {epoch} - Train Loss: ', leave=True)
    for batch in pbar:
        optimizer.zero_grad()
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        loss = model.step(inputs, labels, autoregression_steps=autoregression_steps)
        loss.backward()
        optimizer.step()
        whole_loss.append(loss.detach().cpu().item())
        pbar.set_description(f'Train Loss: {loss.detach().cpu().item():.4f}')
    return np.mean(whole_loss)


def val_epoch(model, val_loader, autoregression_steps, weight_lat):
    model.eval()
    whole_loss = []
    pred = []
    label = []
    pbar = tqdm(val_loader, desc='Val Loss: ', leave=False)
    for batch in pbar:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        loss, outs = model.step(inputs, labels, autoregression_steps=autoregression_steps, return_out=True)

        pred.append(outs.cpu())
        label.append(labels.cpu())

        whole_loss.append(loss.detach().cpu().item())
        pbar.set_description(f'Val Loss: {loss.detach().cpu().item():.4f}')

    pred = torch.flatten(torch.stack(pred, dim=0), 0, 1)
    label = torch.flatten(torch.stack(label, dim=0), 0, 1)

    rmse = compute_weighted_rmse(pred, label, weight_lat.cpu())
    acc = compute_weighted_acc(pred, label, weight_lat.cpu())
    mae = compute_weighted_mae(pred, label, weight_lat.cpu())

    return np.mean(whole_loss), rmse, acc, mae


def get_autoregression_steps(autoregression_steps_epochs, epoch):
    smaller_values = [value for value in autoregression_steps_epochs.keys() if int(value) <= epoch]
    if not smaller_values:
        return 1

    return autoregression_steps_epochs[str(max(smaller_values))]


def train():
    with wandb.init() as run:
        config = run.config
        os.makedirs(os.environ.get('MODEL_DIR'), exist_ok=True)
        logger.info('Using {} device'.format(device))

        autoregression_steps_epochs = config.get('autoregression_steps_epochs')
        min_autoregression_steps = min(autoregression_steps_epochs.values())
        train_dl, test_dl, lat_weights = create_train_test_datasets(min_autoregression_steps)
        lat_weights = lat_weights.to(device)

        logger.info('Creating Model')
        model_parameter = config.get('model_parameter')
        model = FuXi(
            25,
            model_parameter['channel'],
            model_parameter['transformer_blocks'],
            121, 240,
            heads=model_parameter['heads'],
            lat_weights=lat_weights
        ).to(device)

        best_loss = float('inf')
        wandb.watch(model, log_freq=100)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("learning_rate"), betas=(0.9, 0.95),
                                      weight_decay=0.1)
        cur_ds_autoregression_steps = min_autoregression_steps

        for epoch in range(config.get("epochs")):
            autoregression_steps = get_autoregression_steps(autoregression_steps_epochs, epoch)
            if autoregression_steps > cur_ds_autoregression_steps:
                train_dl, test_dl, _ = create_train_test_datasets(autoregression_steps)
                cur_ds_autoregression_steps = autoregression_steps

            train_loss = train_epoch(model, optimizer, train_dl, autoregression_steps, epoch)
            test_loss, rmse, acc, mae = val_epoch(model, test_dl, autoregression_steps, lat_weights)

            val_df = pd.DataFrame({
                'rmse': rmse,
                'acc': acc,
                'mae': mae
            })

            run.log({
                'train_loss': train_loss,
                'test_loss': test_loss,
                'metrics': wandb.Table(data=val_df, columns=['rmse', 'acc', 'mae']),
                'mean_rmse': rmse.mean(),
                'mean_acc': acc.mean(),
                'mean_mae': mae.mean()
            })

            if test_loss < best_loss:
                current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
                filename = f'model_best_loss_{test_loss:.4f}_{current_time}.pth'
                save_path = os.path.join(os.environ.get('MODEL_DIR'), filename)
                torch.save(model.state_dict(), save_path)
                logger.info(f'New best model saved with loss: {test_loss:.4f}')
                best_loss = test_loss - test_loss * 0.1
        wandb.unwatch(model)


if __name__ == '__main__':
    wandb.agent(getSweepID(), train)
