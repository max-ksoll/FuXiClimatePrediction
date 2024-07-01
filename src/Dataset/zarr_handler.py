import logging
import os

import zarr

from src.Dataset.dimensions import *
from src.utils import get_metrics_array

logger = logging.getLogger("Zarr Handler")


class ZarrHandler:

    def __init__(self, directory: os.PathLike | str, start_year: int, end_year: int):
        self.root_path = os.path.join(directory, f'{start_year}_{end_year}.zarr')
        self.mean_root_path = os.path.join(directory, f'mean_{start_year}_{end_year}.zarr')
        self.store = zarr.DirectoryStore(self.root_path)
        self.mean_store = zarr.DirectoryStore(self.mean_root_path)
        self.root = zarr.group(store=self.store, overwrite=True)
        self.mean_root = zarr.group(store=self.mean_store, overwrite=True)
        self.end_year = end_year
        self.start_year = start_year
        self.lat_lon_level_dim = (LAT.size, LON.size, LEVEL.size)
        self.variables = ORAS_VARIABLES + ERA_SURFACE_VARIABLES + ERA_ATMOS_VARIABLES

    def build(self):
        logger.info("Building Zarr")
        self.create_dimensions()
        self.create_variables()

    def create_dimensions(self):
        logger.info("Creating dimensions")
        months = (self.end_year - self.start_year + 1) * 12
        time = Dimension[np.float64](TIME_DIMENSION_NAME, months, -99999, min_val=0, max_val=months - 1,
                                     oras_name="time_counter", era_name="time")
        self.create_dimension(time)
        self.create_dimension(LAT)
        self.create_dimension(LON)
        self.create_dimension(LEVEL)

        self.create_dimension(LEVEL, is_mean=True)
        self.create_dimension(METRICS, is_mean=True)

    def create_dimension(self, dim: Dimension, is_mean: bool = False):
        logger.debug(f"Creating Dimension {dim.name} in {'mean' if is_mean else 'normal'} File")
        root = self.mean_root if is_mean else self.root
        dimension = root.create_dataset(dim.name, shape=(0,), chunks=(dim.size,), dtype=dim.dtype, fill_value=dim.fill_value)
        dimension.attrs['_ARRAY_DIMENSIONS'] = [dim.name]
        if dim.values:
            dimension.append(dim.values)
        else:
            dimension.append(np.linspace(dim.min_val, dim.max_val, dim.size))

    def create_variables(self):
        logger.info("Creating Variables")
        for variable in self.variables:
            self.create_variable(variable)
            self.create_variable(variable, is_mean=True)

    def create_variable(self, var: Variable, is_mean: bool = False):
        logger.debug(f"Creating Variable {var.name} in {'mean' if is_mean else 'normal'} File")
        root = self.root
        lat_cnt, lon_cnt, level_cnt = self.lat_lon_level_dim
        attributes = MEAN_SURFACE_VAR_ATTRIBUTE if is_mean else SURFACE_VAR_ATTRIBUTE
        # TODO chunk größen überdenken Dataloading
        shape = [0, lat_cnt, lon_cnt]
        chunks = [12, lat_cnt, lon_cnt]

        if is_mean:
            root = self.mean_root
            shape = [METRICS.size]
            chunks = [METRICS.size]

        if not var.isSurfaceVar:
            shape.insert(1, level_cnt)
            chunks.insert(1, level_cnt)
            attributes = MEAN_LEVEL_VAR_ATTRIBUTE if is_mean else LEVEL_VAR_ATTRIBUTE

        variable = root.create_dataset(var.name, shape=shape, dtype=np.float32, chunks=chunks, fill_value=-99999)
        variable.attrs['_ARRAY_DIMENSIONS'] = attributes

    def append_data(self, var: Variable, data: np.ndarray):
        logger.debug(f"Appending Data of shape {data.shape} to Variable {var.name}")
        self.root[var.name].append(data)

    def write_means(self):
        logger.info("Calculating and Writing Means")
        for variable in self.variables:
            self.mean_root[variable.name] = get_metrics_array(self.root[variable.name])
            logger.debug(f"[min, max, mean, std] of {variable.name}: {self.mean_root[variable.name]}")

    def finish(self):
        logger.info("Finished writing")
        self.write_means()
        logger.debug("Consolidate Metadata")
        zarr.consolidate_metadata(self.store)
        zarr.consolidate_metadata(self.mean_store)


if __name__ == '__main__':
    builder = ZarrHandler(
        directory='/Users/ksoll/git/FuXiClimatePrediction/data',
        start_year=1958,
        end_year=1958
    )
    builder.build()
