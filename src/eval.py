import logging
import os

import dotenv
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.Dataset.era5_dataset import ERA5Dataset, TimeMode
from src.fuxi import FuXi
from src.score_torch import compute_weighted_acc, compute_weighted_mae, compute_weighted_rmse

dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    device = 'cuda'
if torch.backends.mps.is_available():
    device = 'cpu'


def evaluate(model_path, compute_metrik=True):
    # with wandb.init() as run:
    logger.info('Creating Model')
    model = FuXi(25, 768, 16, 121, 240, heads=4)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    logger.info('Load Model successfully')

    logger.info('Creating Eval Dataset')
    test_ds = ERA5Dataset(
        os.environ.get('DATAFOLDER'),
        1,
        TimeMode.AFTER,
        start_time="2011-01-12-28T23:59:59",
        max_autoregression_steps=1
    )
    lat_weights = test_ds.get_latitude_weights()
    loader_params = {'batch_size': None,
                     'batch_sampler': None,
                     'shuffle': False,
                     'num_workers': os.cpu_count() // 2,
                     'pin_memory': True}
    logger.info('Creating DataLoader')
    test_dl = DataLoader(test_ds, **loader_params, sampler=None)
    pbar = tqdm(test_dl, desc='Initialisierung')

    predicted_labels_list = []
    labels_list = []
    times = []
    i = 0
    for data in pbar:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        predicted_labels = model.forward(inputs)
        times.append(i)
        predicted_labels_list.append(predicted_labels.squeeze().detach().cpu())
        labels_list.append(labels.squeeze().detach().cpu())
        i += 1

    # Konvertieren in xarray DataArray
    # predicted_labels_xr = xr.DataArray(predicted_labels_list, dims=['time', 'variables', 'lat', 'lon'],
    #                                    coords={'time': times})
    # labels_xr = xr.DataArray(labels_list, dims=['time', 'variables', 'lat', 'lon'], coords={'time': times})
    #
    # predicted_labels_list = [torch.tensor(x) for x in predicted_labels_list]
    # labels_list = [torch.Tensor(x) for x in labels_list]

    predicted_labels_list = torch.stack(predicted_labels_list, 0)
    labels_list = torch.stack(labels_list, 0)

    if compute_metrik:
        print("RMSE:", compute_weighted_rmse(predicted_labels_list, labels_list, lat_weights))
        print("ACC:", compute_weighted_acc(predicted_labels_list, labels_list, lat_weights))
        print("MAE:", compute_weighted_mae(predicted_labels_list, labels_list, lat_weights))

    return predicted_labels_list, labels_list


if __name__ == '__main__':
    evaluate("/Users/xgxtphg/Documents/git/DL4WeatherAndClimate/model/model_best_loss_0.0804_20240406-103455.pth")
