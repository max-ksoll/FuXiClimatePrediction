import enum

import numpy as np
import pandas as pd
import torch
import zarr
import xarray as xr
from torch.utils.data import Dataset


class TimeMode(enum.Enum):
    ALL = 0
    AFTER = 1
    BEFORE = 2
    BETWEEN = 3


class ERA5Dataset(Dataset):

    def __init__(self, path_file,
                 time_mode: TimeMode,
                 max_autoregression_steps=1,
                 start_time="2011-01-01-01T00:00:00",
                 end_time="2011-01-12-31T18:00:00",
                 zarr_col_names="lessig",
                 use_xarray=False,
                 ds=None):
        super(ERA5Dataset, self).__init__()
        if not use_xarray:
            store = zarr.DirectoryStore(path_file)
            self.sources = zarr.group(store=store)
        else:
            self.sources = ds

        self.mins = torch.Tensor(
            [193.48901, -3.3835982e-05, -65.45247, -96.98215, -6838.8906]
        )
        self.maxs = torch.Tensor(
            [324.80637, 0.029175894, 113.785934, 89.834595, 109541.625]
        )
        self.max_minus_min = self.maxs - self.mins
        self.mins = self.mins[:, None, None]
        self.max_minus_min = self.max_minus_min[:, None, None]
        self.max_autoregression_steps = max_autoregression_steps + 2

        times = np.array(self.sources["time"])

        def stunden_zu_datum(stunden_array):
            basis_datum = pd.Timestamp('1959-01-01 00:00:00')
            datum_array = [basis_datum + pd.Timedelta(hours=int(h)) for h in stunden_array]
            return datum_array

        if times.dtype == np.int64:
            times = stunden_zu_datum(times)

        if time_mode == TimeMode.AFTER:
            times = times >= np.datetime64(start_time)
        elif time_mode == TimeMode.BEFORE:
            times = times <= np.datetime64(end_time)
        elif time_mode == TimeMode.BETWEEN:
            times_gt = times >= np.datetime64(start_time)
            times_ls = times <= np.datetime64(end_time)
            times = times_gt & times_ls
        else:
            times = np.ones_like(times)

        self.idxs = np.arange(self.sources["time"].shape[0])[times]
        if len(self.idxs) > 0:
            keep_idxs = self.idxs <= np.max(self.idxs) - self.max_autoregression_steps
            self.idxs = self.idxs[keep_idxs]
        self.len = self.idxs.shape[0]

        dim_names = {
            "lessig": {
                "cols": ["t", "q", "u", "v", "z", "lats"],
                "slice_idx": [0, 1, 2, 3, 4],
                "permutation": [0, 1, 2, 3]
            },
            "gcloud": {
                "cols": ["temperature", "specific_humidity", "u_component_of_wind", "v_component_of_wind",
                         "geopotential", "latitude"],
                "slice_idx": [3, 5, 7, 10, 12],
                "permutation": [0, 1, 3, 2]
            }
        }
        self.dim_names = dim_names[zarr_col_names]['cols']
        self.slice_idx = dim_names[zarr_col_names]['slice_idx']
        self.permutation = dim_names[zarr_col_names]['permutation']
        self.lat_weights = self.get_latitude_weights()[:, None]

    def get_latitude_weights(self):
        weights = np.cos(np.deg2rad(np.array(self.sources[self.dim_names[5]])))
        return torch.Tensor(weights)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sources = [
            self.get_at_idx(i) for i in range(self.idxs[idx], self.idxs[idx] + self.max_autoregression_steps)
        ]
        sources = torch.stack(sources, dim=0)
        # Normalization
        sources = (sources - self.mins) / self.max_minus_min
        sources = sources.flatten(start_dim=1, end_dim=2)
        return sources, self.lat_weights

    def get_at_idx(self, idx_t):
        stack = torch.stack([
            torch.tensor(np.array(self.sources[self.dim_names[0]][idx_t])),
            torch.tensor(np.array(self.sources[self.dim_names[1]][idx_t])),
            torch.tensor(np.array(self.sources[self.dim_names[2]][idx_t])),
            torch.tensor(np.array(self.sources[self.dim_names[3]][idx_t])),
            torch.tensor(np.array(self.sources[self.dim_names[4]][idx_t])),
        ], 1, )
        return torch.permute(stack[self.slice_idx, :, :, :], self.permutation)
