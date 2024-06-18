import os

import zarr

from src.Dataset.dimensions import *


def get_zarr_root(directory: os.PathLike | str, start_year: int, end_year: int,
                  is_mean_file: bool = False) -> zarr.hierarchy.Group:
    path = os.path.join(directory, f'{"mean_" if is_mean_file else ""}{start_year}_{end_year}.zarr')
    store = zarr.DirectoryStore(path)
    return zarr.group(store=store, overwrite=True)


def create_zarr_file(root: zarr.hierarchy.Group, dimensions: List[Dimension], variables: List[Variable],
                     is_mean_file: bool = False):
    for dim in dimensions:
        create_dimension(root, dim)
        root[dim.value.name].append(list(dim.value))

    for var in variables:
        create_variable(root, var.value, is_mean_file)


def create_variable(root: zarr.hierarchy.Group, var: VariableInfo, is_mean_file: bool = False):
    if is_mean_file:
        create_mean_variable(root, var)
    else:
        create_var(root, var)


def create_mean_variable(root, var):
    level_cnt = Dimension.LEVEL.value.size
    metrics_cnt = Dimension.METRICS.value.size

    if var.isSurfaceVar:
        variable = root.create_dataset(var.name, shape=(0, metrics_cnt), dtype=np.float64,
                                       chunks=(1, metrics_cnt), fill_value=-99999)
        variable.attrs['_ARRAY_DIMENSIONS'] = MEAN_SURFACE_VAR_ATTRIBUTE
    else:
        variable = root.create_dataset(var.name, shape=(0, level_cnt, metrics_cnt), dtype=np.float64,
                                       chunks=(1, level_cnt, metrics_cnt), fill_value=-99999)
        variable.attrs['_ARRAY_DIMENSIONS'] = MEAN_LEVEL_VAR_ATTRIBUTE


def create_var(root: zarr.hierarchy.Group, var: VariableInfo):
    level_cnt = Dimension.LEVEL.value.size
    lat_cnt = Dimension.LAT.value.size
    lon_cnt = Dimension.LON.value.size
    # TODO chunk größen überdenken Dataloading
    if var.isSurfaceVar:
        variable = root.create_dataset(var.name, shape=(0, lat_cnt, lon_cnt), dtype=np.float64,
                                       chunks=(12, lat_cnt, lon_cnt), fill_value=-99999)
        variable.attrs['_ARRAY_DIMENSIONS'] = SURFACE_VAR_ATTRIBUTE
    else:
        variable = root.create_dataset(var.name, shape=(0, level_cnt, lat_cnt, lon_cnt), dtype=np.float64,
                                       chunks=(12, level_cnt, lat_cnt, lon_cnt), fill_value=-99999)
        variable.attrs['_ARRAY_DIMENSIONS'] = LEVEL_VAR_ATTRIBUTE


def create_dimension(root: zarr.hierarchy.Group, var: Dimension):
    dim = root.create_dataset(var.value.name, shape=(0,), chunks=(var.value.size,), dtype=var.value.dtype,
                        fill_value=var.value.fill_value)
    dim.attrs['_ARRAY_DIMENSIONS'] = [var.value.name]


if __name__ == '__main__':
    root = get_zarr_root("/Users/ksoll/git/FuXiClimatePrediction/data", 1958, 1958)
    create_zarr_file(root)
