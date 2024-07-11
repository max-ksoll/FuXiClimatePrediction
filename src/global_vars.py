from typing import TypeVar

OPTIMIZER_REQUIRED_KEYS = [
    "optimizer_config_lr",
    "optimizer_config_betas",
    "optimizer_config_weight_decay",
    "optimizer_config_T_0",
    "optimizer_config_T_mult",
    "optimizer_config_eta_min",
]
LAT_DIM = 121
LONG_DIM = 240

LEVEL_VAR_ATTRIBUTE = ["time", "level", "latitude", "longitude"]
MEAN_LEVEL_VAR_ATTRIBUTE = ["metric", "level"]
SURFACE_VAR_ATTRIBUTE = ["time", "latitude", "longitude"]
MEAN_SURFACE_VAR_ATTRIBUTE = ["metric"]

TIME_DIMENSION_NAME = "time"

T = TypeVar("T")

ORAS_LAT_LON_GRID_SIZE = (0.25, 0.25)
LAT_LON_GRID_SIZE = (1.5, 1.5)

ERA5_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "sea_surface_temperature",
    "soil_temperature_level_1",
    "surface_latent_heat_flux",
    "total_precipitation",
    "volumetric_soil_water_layer_2",
]

ORAS5_VARIABLES = [
    "ocean_heat_content_for_the_total_water_column",
    "ocean_heat_content_for_the_upper_300m",
]
