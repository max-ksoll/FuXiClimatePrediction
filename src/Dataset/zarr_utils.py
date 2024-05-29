import os

import zarr

from src.Dataset.dimensions import *


def create_zarr_file(directory: os.PathLike | str, start_year: int, end_year: int):
    store = zarr.DirectoryStore(os.path.join(directory, f'{start_year}_{end_year}.zarr'))
    root = zarr.group(store=store, overwrite=True)

    for dim in list(Dimension):
        create_dimension(root, dim)

    for var in list(Variable):
        create_variable(root, var)


def create_variable(root: zarr.hierarchy.Group, var: Variable):
    level_cnt = Dimension.LEVEL.value.size
    lat_cnt = Dimension.LAT.value.size
    lon_cnt = Dimension.LON.value.size
    variable = root.create_dataset(var.value.name, shape=(0, level_cnt, lat_cnt, lon_cnt),
                                   dtype=np.float64, chunks=(12, level_cnt, lat_cnt, lon_cnt), fill_value=-99999)
    if var.value.isSurfaceVar:
        variable.attrs['_ARRAY_DIMENSIONS'] = SURFACE_VAR_ATTRIBUTE
    else:
        variable.attrs['_ARRAY_DIMENSIONS'] = LEVEL_VAR_ATTRIBUTE


def create_dimension(root: zarr.hierarchy.Group, var: Dimension):
    root.create_dataset(var.value.name, shape=(0,), chunks=(var.value.size,), dtype=var.value.dtype,
                        fill_value=var.value.fill_value)


def write_to_zarr(root: zarr.hierarchy.Group, var: Variable, data: np.ndarray):
    pass


if __name__ == '__main__':
    create_zarr_file("/Users/ksoll/git/FuXiClimatePrediction/data", 1958, 1958)
