import dataclasses
from enum import Enum

import numpy as np

LEVEL_VAR_ATTRIBUTE = ['time', 'level', 'latitude', 'longitude']
SURFACE_VAR_ATTRIBUTE = ['time', 'latitude', 'longitude']


@dataclasses.dataclass
class VariableInfo:
    name: str
    isSurfaceVar: bool
    cdf_name: str


@dataclasses.dataclass
class DimensionInfo:
    name: str
    size: int
    dtype: np.dtype
    fill_value: int


class Dimension(Enum):
    LAT = DimensionInfo('latitude', 121, np.float64, -99999)
    LON = DimensionInfo('longitude', 240, np.float64, -99999)
    TIME = DimensionInfo('time', 1, np.float64, -99999)
    LEVEL = DimensionInfo('level', 5, np.float32, -99)


class Variable(Enum):
    V_WIND = VariableInfo("v_component_of_wind", False, "")
    U_WIND = VariableInfo("u_component_of_wind", False, "")
    TEMP = VariableInfo("temperature", False, "")
    HUMIDITY = VariableInfo("specific_humidity", False, "")
    GEOPOTENTIAL = VariableInfo("geopotential", False, "")
    V_WIND_10M = VariableInfo("10m_v_component_of_wind", True, "")
    U_WIND_10M = VariableInfo("10m_u_component_of_wind", True, "")
    SURFACE_TEMP = VariableInfo("2m_temperature", True, "")
    SEA_SURFACE_TEMP = VariableInfo("sea_surface_temperature", True, "")
    SOIL_TEMP_LV1 = VariableInfo("soil_temperature_level_1", True, "")
    HEAT_FLUX = VariableInfo("surface_latent_heat_flux", True, "")
    TOTAL_PRECIPITATION = VariableInfo("total_precipitation", True, "")
    VOL_SOIL_WATER_LV2 = VariableInfo("volumetric_soil_water_layer_2", True, "")
    COMPLETE_OCEAN_HEAT_CONTENT = VariableInfo("ocean_heat_content_for_the_total_water_column", True, "")
    OCEAN_HEAT_CONTENT_300M = VariableInfo("ocean_heat_content_for_the_upper_300m", True, "")
