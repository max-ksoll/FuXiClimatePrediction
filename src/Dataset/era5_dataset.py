import os
from typing import Tuple

import os
from typing import Tuple

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

from src.Dataset.dimensions import Dimension, VARIABLES


class ERA5Dataset(Dataset):

    def __init__(self, dataset_path: os.PathLike | str,
                 means_file: os.PathLike | str,
                 max_autoregression_steps: int = 1):
        super(ERA5Dataset, self).__init__()
        store = zarr.DirectoryStore(dataset_path)
        self.sources = zarr.group(store=store)

        store = zarr.DirectoryStore(means_file)
        self.means = zarr.group(store=store)

        self.max_autoregression_steps = max_autoregression_steps + 2

        times = np.array(self.sources[Dimension.TIME.value.name])

        self.idxs = np.arange(self.sources[Dimension.TIME.value.name].shape[0])[times]
        if len(self.idxs) > 0:
            keep_idxs = self.idxs <= np.max(self.idxs) - self.max_autoregression_steps
            self.idxs = self.idxs[keep_idxs]
        self.len = self.idxs.shape[0]

        self.surface_vars = list(filter(lambda x: x.value.isSurfaceVar, list(VARIABLES)))
        self.atmos_vars = list(filter(lambda x: not x.value.isSurfaceVar, list(VARIABLES)))
        self.lat_weights = self.get_latitude_weights()[:, None]

        self.init_max_min()

    def init_max_min(self):
        self.min = torch.cat([
            torch.stack([torch.tensor(np.array(self.means[var.value.name][:, 0])) for var in self.surface_vars], 1),
            torch.stack([torch.tensor(np.array(self.means[var.value.name][:, :, 0]))
                         for var in self.atmos_vars], 1).flatten(start_dim=1, end_dim=2)
        ], dim=1)

        max_val = torch.cat([
            torch.stack([torch.tensor(np.array(self.means[var.value.name][:, 1])) for var in self.surface_vars], 1),
            torch.stack([torch.tensor(np.array(self.means[var.value.name][:, :, 1]))
                         for var in self.atmos_vars], 1).flatten(start_dim=1, end_dim=2)
        ], dim=1)

        self.max_minus_min = max_val - self.min
        self.min = self.min[0, :, None, None]
        self.max_minus_min = self.max_minus_min[0, :, None, None]

    def get_latitude_weights(self):
        weights = np.cos(np.deg2rad(np.array(self.sources[Dimension.LAT.value.name])))
        return torch.Tensor(weights)

    def __len__(self):
        return self.len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sources = [
            self.get_at_idx(i) for i in range(self.idxs[idx], self.idxs[idx] + self.max_autoregression_steps)
        ]
        sources = torch.stack(sources, dim=0)
        # Normalization
        sources = (sources - self.min) / self.max_minus_min
        return sources, self.lat_weights

    def get_at_idx(self, idx_t: int) -> torch.Tensor:
        return torch.cat([
            torch.stack([torch.tensor(np.array(self.sources[var.value.name][idx_t])) for var in self.surface_vars], 0),
            torch.stack([torch.tensor(np.array(self.sources[var.value.name][idx_t])) for var in self.atmos_vars],
                        0).flatten(start_dim=0, end_dim=1)
        ], dim=0)
