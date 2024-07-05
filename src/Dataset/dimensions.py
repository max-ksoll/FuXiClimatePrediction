import dataclasses
from enum import Enum
from typing import TypeVar, Generic, Optional, List

import numpy as np

LEVEL_VAR_ATTRIBUTE = ["time", "level", "latitude", "longitude"]
MEAN_LEVEL_VAR_ATTRIBUTE = ["metric", "level"]
SURFACE_VAR_ATTRIBUTE = ["time", "latitude", "longitude"]
MEAN_SURFACE_VAR_ATTRIBUTE = ["metric"]

TIME_DIMENSION_NAME = "time"

T = TypeVar("T")


class VariableType(Enum):
    ORAS = 0
    ERA_SURFACE = 1
    ERA_ATMOS = 2


@dataclasses.dataclass
class Variable:
    name: str
    isSurfaceVar: bool
    cdf_name: str
    var_type: VariableType


class Dimension(Generic[T]):
    def __init__(
        self,
        name: str,
        size: int,
        fill_value: T,
        min_val: T = None,
        max_val: T = None,
        oras_name: str = "",
        era_name: str = "",
        values: List[T] = None,
    ):
        self.name = name
        self.size = size
        self.dtype = type(fill_value)
        self.fill_value = fill_value
        self.min_val = min_val
        self.max_val = max_val
        self.oras_name = oras_name
        self.era_name = era_name
        self.values = values

    def __iter__(self):
        if self.values:
            for value in self.values:
                yield value
        elif self.min_val is not None and self.max_val is not None:
            for value in np.linspace(self.min_val, self.max_val, self.size):
                yield value
        else:
            raise ValueError(
                "Either 'values' or both 'min_val' and 'max_val' must be set to iterate."
            )


LAT = Dimension[np.float64](
    "latitude", 121, -99999, min_val=-90, max_val=90, oras_name="y", era_name="latitude"
)
LON = Dimension[np.float64](
    "longitude",
    240,
    -99999,
    min_val=0,
    max_val=358.5,
    oras_name="x",
    era_name="longitude",
)
LEVEL = Dimension[np.float32]("level", 5, -99, min_val=0, max_val=4, era_name="level")
# Die Reihenfolge der Values darf nicht geändert werden
METRICS_ARRAY = ['min', 'max', 'mean', 'std']
METRICS = Dimension[np.string_]('metric', 4, "", values=METRICS_ARRAY)

# ORAS
COMPLETE_OCEAN_HEAT_CONTENT = Variable(
    "ocean_heat_content_for_the_total_water_column", True, "sohtcbtm", VariableType.ORAS
)
OCEAN_HEAT_CONTENT_300M = Variable(
    "ocean_heat_content_for_the_upper_300m", True, "sohtc300", VariableType.ORAS
)

# ERA Surface
V_WIND_10M = Variable("10m_v_component_of_wind", True, "u10", VariableType.ERA_SURFACE)
U_WIND_10M = Variable("10m_u_component_of_wind", True, "v10", VariableType.ERA_SURFACE)
SURFACE_TEMP = Variable("2m_temperature", True, "t2m", VariableType.ERA_SURFACE)
SEA_SURFACE_TEMP = Variable(
    "sea_surface_temperature", True, "sst", VariableType.ERA_SURFACE
)
SOIL_TEMP_LV1 = Variable(
    "soil_temperature_level_1", True, "stl1", VariableType.ERA_SURFACE
)
HEAT_FLUX = Variable("surface_latent_heat_flux", True, "slhf", VariableType.ERA_SURFACE)
TOTAL_PRECIPITATION = Variable(
    "total_precipitation", True, "tp", VariableType.ERA_SURFACE
)
VOL_SOIL_WATER_LV2 = Variable(
    "volumetric_soil_water_layer_2", True, "swvl2", VariableType.ERA_SURFACE
)

# ERA Atmos
V_WIND = Variable("v_component_of_wind", False, "v", VariableType.ERA_ATMOS)
U_WIND = Variable("u_component_of_wind", False, "u", VariableType.ERA_ATMOS)
TEMP = Variable("temperature", False, "t", VariableType.ERA_ATMOS)
HUMIDITY = Variable("specific_humidity", False, "z", VariableType.ERA_ATMOS)
GEOPOTENTIAL = Variable("geopotential", False, "q", VariableType.ERA_ATMOS)

ORAS_VARIABLES = [COMPLETE_OCEAN_HEAT_CONTENT, OCEAN_HEAT_CONTENT_300M]

ERA_SURFACE_VARIABLES = [
    V_WIND_10M,
    U_WIND_10M,
    SURFACE_TEMP,
    SEA_SURFACE_TEMP,
    SOIL_TEMP_LV1,
    HEAT_FLUX,
    TOTAL_PRECIPITATION,
    VOL_SOIL_WATER_LV2,
]

ERA_ATMOS_VARIABLES = [V_WIND, U_WIND, TEMP, HUMIDITY, GEOPOTENTIAL]
