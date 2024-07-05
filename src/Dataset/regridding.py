import math
from typing import Tuple

import numpy as np
import scipy.spatial
from scipy.spatial import cKDTree
from src.utils import log_exec_time


def interpolate_tripolar_to_lat_lon(
    tripolar_data: np.ndarray, target_coords: np.ndarray, tree: scipy.spatial.cKDTree
):
    dist, idx = tree.query(target_coords)
    interpolated_values = tripolar_data.ravel()[idx]
    return interpolated_values


@log_exec_time
def transform_tripolar_to_lat_lon(
    tripolar_data: np.ndarray,
    tripolar_lat: np.ndarray,
    tripolar_lon: np.ndarray,
    lat_lon_grid_size: Tuple[float, float],
):
    num_lat_points, num_lon_points = lat_lon_grid_size
    num_lat_points = math.ceil(180 / num_lat_points)
    num_lon_points = math.ceil(360 / num_lon_points)
    lat_range = np.linspace(-90, 90, num_lat_points)
    lon_range = np.linspace(-180, 180, num_lon_points)

    # Erstellen eines KD-Baumes für die schnellen Nachbarschaftsabfragen
    tripolar_points = np.column_stack((tripolar_lat.ravel(), tripolar_lon.ravel()))
    tree = cKDTree(tripolar_points)

    # Erstellen eines Gitters von Zielkoordinaten
    target_lon, target_lat = np.meshgrid(lon_range, lat_range)
    target_coords = np.column_stack((target_lat.ravel(), target_lon.ravel()))

    # Interpolieren der Werte für alle Zielkoordinaten
    lat_lon_data = interpolate_tripolar_to_lat_lon(tripolar_data, target_coords, tree)
    lat_lon_data = lat_lon_data.reshape((num_lat_points, num_lon_points))

    return lat_lon_data


if __name__ == "__main__":
    # Beispielaufruf
    tripolar_data = np.random.rand(100, 100)  # Beispieldaten
    tripolar_lat = np.random.uniform(-90, 90, (100, 100))  # Beispielhafte Laten
    tripolar_lon = np.random.uniform(-180, 180, (100, 100))  # Beispielhafte Lonen
    lat_lon_grid_size = (721, 1440)  # Zielgittergröße (z.B. 1x1 Grad)

    lat_lon_data = transform_tripolar_to_lat_lon(
        tripolar_data, tripolar_lat, tripolar_lon, lat_lon_grid_size
    )

    print(lat_lon_data)
