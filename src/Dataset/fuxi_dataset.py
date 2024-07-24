import logging
import os
from functools import lru_cache
from typing import Tuple

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

from src.Dataset.dimensions import (
    LAT,
    METRICS_ARRAY,
    SURFACE_VARIABLES,
    LEVEL_VARIABLES,
)
from src.global_vars import TIME_DIMENSION_NAME

logger = logging.getLogger("ERA5 Dataset")


class FuXiDataset(Dataset):
    def __init__(
        self,
        dataset_path: os.PathLike | str,
        means_file: os.PathLike | str,
        max_autoregression_steps: int = 1,
    ):
        self.dataset_path = dataset_path
        self.means_file = means_file
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path {dataset_path} does not exist")
        if not os.path.exists(means_file):
            raise ValueError(f"Means file {means_file} does not exist")
        logger.info(
            f"Creating FuXi Dataset with Autoregression: {max_autoregression_steps}"
        )
        super(FuXiDataset, self).__init__()
        store = zarr.DirectoryStore(dataset_path)
        self.sources = zarr.group(store=store)

        store = zarr.DirectoryStore(means_file)
        self.means = zarr.group(store=store)

        self.set_autoregression_steps(max_autoregression_steps)

        self.init_max_min()

    def set_autoregression_steps(self, max_autoregression_steps: int):
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

    def init_max_min(self):
        logger.debug("Loading Min Tensor")
        self.min = self.get_from_means_file("min")

        logger.debug("Loading Max Tensor")
        max_val = self.get_from_means_file("max")

        self.max_minus_min = max_val - self.min
        self.min = self.min[:, None, None]
        logger.debug(f"Min Tensor Shape: {self.min.shape}")
        self.max_minus_min = self.max_minus_min[:, None, None]
        logger.debug(f"Max-Min Tensor Shape: {self.max_minus_min.shape}")

    def __len__(self):
        return self.len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        logger.debug(f"Retrieving Data at Idx: {idx}")
        logger.debug(f"Loading Data")
        sources = [
            self.get_at_idx(i)
            for i in range(
                self.idxs[idx], self.idxs[idx] + self.max_autoregression_steps
            )
        ]
        sources = torch.stack(sources, dim=0)
        sources = self.normalize(sources)
        logger.debug(
            f"Shape: {sources.shape}, Min: {np.nanmin(sources.numpy())}, Max: {np.nanmax(sources.numpy())}"
        )
        return sources

    def get_at_idx(self, idx_t: int) -> torch.Tensor:
        logger.debug(f"Retrieving Data at Datafile Idx: {idx_t}")
        return torch.cat(
            [
                torch.stack(
                    [
                        torch.tensor(np.array(self.sources[var.name][idx_t]))
                        for var in SURFACE_VARIABLES
                    ],
                    0,
                ),
                torch.stack(
                    [
                        torch.tensor(np.array(self.sources[var.name][idx_t]))
                        for var in LEVEL_VARIABLES
                    ],
                    0,
                ).flatten(start_dim=0, end_dim=1),
            ],
            dim=0,
        )

    @staticmethod
    @lru_cache
    def get_var_name_and_level_at_idx(idx: int) -> Tuple[str, int]:
        if idx < len(SURFACE_VARIABLES):
            return SURFACE_VARIABLES[idx].name, -1
        idx -= len(SURFACE_VARIABLES)
        return LEVEL_VARIABLES[idx // 5].name, idx % 5

    def get_from_means_file(self, mode: str):
        mode_idx = METRICS_ARRAY.index(mode)
        return torch.cat(
            [
                torch.stack(
                    [
                        torch.tensor(np.array(self.means[var.name][mode_idx]))
                        for var in SURFACE_VARIABLES
                    ],
                    0,
                ),
                torch.stack(
                    [
                        torch.tensor(np.array(self.means[var.name][mode_idx, :]))
                        for var in LEVEL_VARIABLES
                    ],
                    0,
                ).flatten(),
            ],
            dim=0,
        )

    @lru_cache
    def get_clima_mean(self) -> torch.Tensor:
        return self.normalize(
            torch.cat(
                [
                    torch.mean(
                        torch.stack(
                            [
                                torch.tensor(np.array(self.sources[var.name]))
                                for var in SURFACE_VARIABLES
                            ],
                            0,
                        ),
                        dim=1,
                    ),
                    torch.mean(
                        torch.stack(
                            [
                                torch.tensor(np.array(self.sources[var.name]))
                                for var in LEVEL_VARIABLES
                            ],
                            0,
                        ),
                        dim=1,
                    ).flatten(start_dim=0, end_dim=1),
                ],
                dim=0,
            )
        )

    def normalize(self, inp) -> torch.Tensor:
        logger.debug(f"Normalizing Data")
        return (inp - self.min) / self.max_minus_min


if __name__ == "__main__":
    ds = FuXiDataset(
        dataset_path="/Users/ksoll/git/FuXiClimatePrediction/data/1958_1958.zarr",
        means_file="/Users/ksoll/git/FuXiClimatePrediction/data/mean_1958_1958.zarr",
    )
    for item in iter(ds):
        print(item[0].shape, item[1].shape)
        break
