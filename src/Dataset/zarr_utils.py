import os

import numpy as np
import zarr

from src.Dataset.dimensions import *


def get_zarr_root(directory: os.PathLike | str, start_year: int, end_year: int) -> zarr.hierarchy.Group:
    store = zarr.DirectoryStore(os.path.join(directory, f'{start_year}_{end_year}.zarr'))
    return zarr.group(store=store, overwrite=True)


def create_zarr_file(root: zarr.hierarchy.Group):
    for dim in list(Dimension):
        create_dimension(root, dim)
        fill_dimension(root, dim)

    variables = list(ORASVariable)
    variables.extend(list(ERAATMOSVariable))
    variables.extend(list(ERASURFACEVariable))
    for var in variables:
        create_variable(root, var.value)


def create_variable(root: zarr.hierarchy.Group, var: VariableInfo):
    level_cnt = Dimension.LEVEL.value.size
    lat_cnt = Dimension.LAT.value.size
    lon_cnt = Dimension.LON.value.size

    if var.isSurfaceVar:
        variable = root.create_dataset(var.name, shape=(0, lat_cnt, lon_cnt),
                                       dtype=np.float64, chunks=(12, lat_cnt, lon_cnt), fill_value=-99999)
        variable.attrs['_ARRAY_DIMENSIONS'] = SURFACE_VAR_ATTRIBUTE
    else:
        variable = root.create_dataset(var.name, shape=(0, level_cnt, lat_cnt, lon_cnt),
                                       dtype=np.float64, chunks=(12, level_cnt, lat_cnt, lon_cnt), fill_value=-99999)
        variable.attrs['_ARRAY_DIMENSIONS'] = LEVEL_VAR_ATTRIBUTE


def create_dimension(root: zarr.hierarchy.Group, var: Dimension):
    root.create_dataset(var.value.name, shape=(0,), chunks=(var.value.size,), dtype=var.value.dtype,
                        fill_value=var.value.fill_value)


def fill_dimension(root: zarr.hierarchy.Group, var: Dimension):
    if var == Dimension.TIME or var == Dimension.LEVEL:
        root[var.value.name].append(np.arange(var.value.size))
    if var == Dimension.LAT:
        root[var.value.name].append(np.linspace(-90, 90, var.value.size))
    if var == Dimension.LON:
        root[var.value.name].append(np.linspace(0, 358.5, var.value.size))


def write_to_zarr(root: zarr.hierarchy.Group, var: Variable, data: np.ndarray):
    pass


if __name__ == '__main__':
    root = get_zarr_root("/Users/ksoll/git/FuXiClimatePrediction/data", 1958, 1958)
    create_zarr_file(root)
