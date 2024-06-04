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


@dataclasses.dataclass
class ReanalysisDimensionInfo(DimensionInfo):
    oras_name: str
    era_name: str


class Dimension(Enum):
    LAT = ReanalysisDimensionInfo('latitude', 121, np.float64, -99999, "nav_lat", "latitude")
    LON = ReanalysisDimensionInfo('longitude', 240, np.float64, -99999, "nav_lon", "longitude")
    TIME = ReanalysisDimensionInfo('time', 1, np.float64, -99999, "time_counter", "time")
    LEVEL = ReanalysisDimensionInfo('level', 5, np.float32, -99, "", "level")


class MeanDimension(Enum):
    MIN = DimensionInfo('min', 1, np.float64, -99999)
    MAX = DimensionInfo('max', 1, np.float64, -99999)
    MEAN = DimensionInfo('mean', 1, np.float64, -99999)
    STD = DimensionInfo('std', 1, np.float64, -99999)


class Variable(Enum):
    V_WIND = VariableInfo("v_component_of_wind", False, "v")
    U_WIND = VariableInfo("u_component_of_wind", False, "u")
    TEMP = VariableInfo("temperature", False, "t")
    HUMIDITY = VariableInfo("specific_humidity", False, "z")
    GEOPOTENTIAL = VariableInfo("geopotential", False, "q")
    V_WIND_10M = VariableInfo("10m_v_component_of_wind", True, "u10")
    U_WIND_10M = VariableInfo("10m_u_component_of_wind", True, "v10")
    SURFACE_TEMP = VariableInfo("2m_temperature", True, "t2m")
    SEA_SURFACE_TEMP = VariableInfo("sea_surface_temperature", True, "sst")
    SOIL_TEMP_LV1 = VariableInfo("soil_temperature_level_1", True, "stl1")
    HEAT_FLUX = VariableInfo("surface_latent_heat_flux", True, "slhf")
    TOTAL_PRECIPITATION = VariableInfo("total_precipitation", True, "tp")
    VOL_SOIL_WATER_LV2 = VariableInfo("volumetric_soil_water_layer_2", True, "swvl2")
    COMPLETE_OCEAN_HEAT_CONTENT = VariableInfo("ocean_heat_content_for_the_total_water_column", True, "sohtcbtm")
    OCEAN_HEAT_CONTENT_300M = VariableInfo("ocean_heat_content_for_the_upper_300m", True, "sohtc300")
