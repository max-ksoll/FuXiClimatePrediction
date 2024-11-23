import numpy as np

from src.Dataset.dimensions import LEVEL_VARIABLES, SURFACE_VARIABLES


def get_unit_for_var_name(var_name):
    variables = LEVEL_VARIABLES + SURFACE_VARIABLES
    variables = list(filter(lambda x: x.name == var_name, variables))
    return variables[0].unit


def get_slice_for_lat_lon(
    lat_start: float, lat_end: float, lon_start: float, lon_end: float, tensor_shape
):
    """
    Berechnet die Slice-Indizes für gegebene Breitengrad- und Längengradbereiche basierend auf der Tensorform.

    Parameters:
    lat_start (float): Startwert des Breitengrads in Grad (-90 bis +90).
    lat_end (float): Endwert des Breitengrads in Grad (-90 bis +90).
    lon_start (float): Startwert des Längengrads in Grad (-180 bis +180).
    lon_end (float): Endwert des Längengrads in Grad (-180 bis +180).
    tensor_shape (tuple): Form des Tensors, beinhaltet die Größen der Latitude- und Longitude-Dimensionen.

    Returns:
    Tuple[Tuple[int, int], Tuple[int, int]]: ((lat_start_idx, lat_end_idx), (lon_start_idx, lon_end_idx))
    """
    # Extrahiere die Größen der Latitude- und Longitude-Dimensionen
    # Angenommen, der Tensor hat die Form [Batch Size, Autoregression, Variablen, Latitude, Longitude]
    lat_size = tensor_shape[3]
    lon_size = tensor_shape[4]

    # Definiere die Bereiche von Latitude und Longitude im Tensor
    lat_min = -90.0
    lat_max = 90.0
    lon_min = -180.0
    lon_max = 180.0

    # Berechne die Schrittweite (Auflösung) für Latitude und Longitude
    lat_step = (lat_max - lat_min) / (lat_size - 1)
    lon_step = (lon_max - lon_min) / (lon_size - 1)

    # Berechne die Indizes für Latitude
    lat_start_idx = int(round((lat_start - lat_min) / lat_step))
    lat_end_idx = int(round((lat_end - lat_min) / lat_step))
    lat_start_idx = max(0, min(lat_start_idx, lat_size - 1))
    lat_end_idx = max(0, min(lat_end_idx, lat_size - 1))
    if lat_start_idx > lat_end_idx:
        lat_start_idx, lat_end_idx = lat_end_idx, lat_start_idx

    # Berechne die Indizes für Longitude
    lon_start_idx = int(round((lon_start - lon_min) / lon_step))
    lon_end_idx = int(round((lon_end - lon_min) / lon_step))
    lon_start_idx = max(0, min(lon_start_idx, lon_size - 1))
    lon_end_idx = max(0, min(lon_end_idx, lon_size - 1))
    if lon_start_idx > lon_end_idx:
        lon_start_idx, lon_end_idx = lon_end_idx, lon_start_idx

    return (lat_start_idx, lat_end_idx), (lon_start_idx, lon_end_idx)


def get_cmap_for_var_name(var_name):
    variables = LEVEL_VARIABLES + SURFACE_VARIABLES
    variables = list(filter(lambda x: x.name == var_name, variables))
    return variables[0].cmap


def calculate_area_weights(latitudes):
    """
    Berechnet die Flächengewichte (areacello) basierend auf den Breitengraden.
    Die Fläche einer Zelle wird angenähert durch den Kosinus des Breitengrads.

    Parameters:
    latitudes (np.array): Ein Array von Breitengraden (in Grad), für die die Flächengewichte berechnet werden sollen.

    Returns:
    np.array: Ein 1D-Array von Flächengewichten.
    """
    # Konvertiere die Breitengrade in Radians
    lat_radians = np.radians(latitudes)
    # Kosinus des Breitengrads wird als Gewichte verwendet
    area_weights = np.cos(lat_radians)
    return area_weights
