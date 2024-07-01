import logging
import os
from typing import Tuple

import os
from typing import Tuple

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

from src.Dataset.dimensions import Dimension, ORAS_VARIABLES, ERA_SURFACE_VARIABLES, ERA_ATMOS_VARIABLES, \
    TIME_DIMENSION_NAME, LAT

logger = logging.getLogger('ERA5 Dataset')


class FuXiDataset(Dataset):

    def __init__(self, dataset_path: os.PathLike | str,
                 means_file: os.PathLike | str,
                 max_autoregression_steps: int = 1):
        super(FuXiDataset, self).__init__()
        store = zarr.DirectoryStore(dataset_path)
        self.sources = zarr.group(store=store)

        store = zarr.DirectoryStore(means_file)
        self.means = zarr.group(store=store)

        self.max_autoregression_steps = max_autoregression_steps + 2
        logger.debug(f"Retriving {self.max_autoregression_steps} items with every call")
        times = np.array(self.sources[TIME_DIMENSION_NAME])
        self.idxs = np.arange(self.sources[TIME_DIMENSION_NAME].shape[0])[times]
        if len(self.idxs) > 0:
            keep_idxs = self.idxs <= np.max(self.idxs) - self.max_autoregression_steps
            self.idxs = self.idxs[keep_idxs]
            logger.debug(f"Possible Idxs: {self.idxs}")
        self.len = self.idxs.shape[0]
        logger.debug(f"Number of Examples in DS {self.len}")

        self.surface_vars = ORAS_VARIABLES + ERA_SURFACE_VARIABLES
        self.atmos_vars = ERA_ATMOS_VARIABLES
        self.lat_weights = self.get_latitude_weights()[:, None]

        self.init_max_min()

    def init_max_min(self):
        logger.debug("Loading Min Tensor")
        stack = torch.stack([torch.tensor(np.array(self.means[var.name][0])) for var in self.surface_vars], 0)
        stack2 = torch.stack([torch.tensor(np.array(self.means[var.name][0, :])) for var in self.atmos_vars],
                             0).flatten()
        self.min = torch.cat([
            torch.stack([torch.tensor(np.array(self.means[var.name][0])) for var in self.surface_vars], 0),
            torch.stack([torch.tensor(np.array(self.means[var.name][0, :]))
                         for var in self.atmos_vars], 0).flatten()
        ], dim=0)

        logger.debug("Loading Max Tensor")
        max_val = torch.cat([
            torch.stack([torch.tensor(np.array(self.means[var.name][1])) for var in self.surface_vars], 0),
            torch.stack([torch.tensor(np.array(self.means[var.name][1, :]))
                         for var in self.atmos_vars], 0).flatten()
        ], dim=0)

        self.max_minus_min = max_val - self.min
        self.min = self.min[:, None, None]
        logger.debug(f"Min Tensor Shape: {self.min.shape}")
        self.max_minus_min = self.max_minus_min[:, None, None]
        logger.debug(f"Max-Min Tensor Shape: {self.max_minus_min.shape}")

    def get_latitude_weights(self):
        logger.debug("Calculating latitude weights")
        weights = np.cos(np.deg2rad(np.array(self.sources[LAT.name])))
        return torch.Tensor(weights)

    def __len__(self):
        return self.len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        logger.debug(f"Retrieving Data at Idx: {idx}")
        logger.debug(f"Loading Data")
        sources = [
            self.get_at_idx(i) for i in range(self.idxs[idx], self.idxs[idx] + self.max_autoregression_steps)
        ]
        sources = torch.stack(sources, dim=0)
        # Normalization
        logger.debug(f"Normalizing Data")
        # sources = (sources - self.min) / self.max_minus_min

        logger.debug(f"Shape: {sources.shape}, Min: {np.nanmin(sources.numpy())}, Max: {np.nanmax(sources.numpy())}")
        return sources, self.lat_weights

    def get_at_idx(self, idx_t: int) -> torch.Tensor:
        logger.debug(f"Retrieving Data at Datafile Idx: {idx_t}")
        return torch.cat([
            torch.stack([torch.tensor(np.array(self.sources[var.name][idx_t])) for var in self.surface_vars], 0),
            torch.stack([torch.tensor(np.array(self.sources[var.name][idx_t])) for var in self.atmos_vars],
                        0).flatten(start_dim=0, end_dim=1)
        ], dim=0)

    def get_var_name_and_level_at_idx(self, idx: int) -> Tuple[str, int]:
        if idx < len(self.surface_vars):
            return self.surface_vars[idx].name, 0
        idx -= len(self.surface_vars)
        return self.atmos_vars[idx // 5].name, idx % 5


if __name__ == '__main__':
    ds = FuXiDataset(
        dataset_path='/Users/ksoll/git/FuXiClimatePrediction/data/1958_1958.zarr',
        means_file='/Users/ksoll/git/FuXiClimatePrediction/data/mean_1958_1958.zarr'
    )
    for item in iter(ds):
        print(item[0].shape, item[1].shape)
        break
