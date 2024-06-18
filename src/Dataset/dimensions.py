import dataclasses
from enum import Enum
from typing import TypeVar, Generic, Optional, List

import numpy as np

LEVEL_VAR_ATTRIBUTE = ['time', 'level', 'latitude', 'longitude']
MEAN_LEVEL_VAR_ATTRIBUTE = ['time', 'level', 'metric']
SURFACE_VAR_ATTRIBUTE = ['time', 'latitude', 'longitude']
MEAN_SURFACE_VAR_ATTRIBUTE = ['time', 'metric']

T = TypeVar("T")


@dataclasses.dataclass
class VariableInfo:
    name: str
    isSurfaceVar: bool
    cdf_name: str


class DimensionInfo(Generic[T]):

    def __init__(self, name: str, size: int, fill_value: T, min_val: T = None, max_val: T = None,
                 oras_name: str = "", era_name: str = "", values: List[T] = None):
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
            raise ValueError("Either 'values' or both 'min_val' and 'max_val' must be set to iterate.")


class Dimension(Enum):
    LAT = DimensionInfo[np.float64]('latitude', 121, -99999, min_val=-90, max_val=90,
                                    oras_name="y", era_name="latitude")
    LON = DimensionInfo[np.float64]('longitude', 240, -99999, min_val=0, max_val=358.5,
                                    oras_name="x", era_name="longitude")
    # TODO hier muss noch irgendwie gelöst werden, dass man nicht händisch die menge der Zeitschritte ändern muss
    TIME = DimensionInfo[np.float64]('time', 12, -99999, min_val=0, max_val=11,
                                     oras_name="time_counter", era_name="time")
    LEVEL = DimensionInfo[np.float32]('level', 5, -99, min_val=0, max_val=4, era_name="level")

    # Ich glaube hier darf die reihenfolge nicht geändert werden von den Werte in values
    METRICS = DimensionInfo[np.string_]('metric', 4, "", values=['min', 'max', 'mean', 'std'])

    # TODO das hier ist jetzt nicht die schönste Lösung, aber mir fällt auf Anhieb gerade nicht ein wie ich es besser
    # TODO machen kann, deswegen mache ich es erstmal so.
    METRICS_TIME = DimensionInfo[np.float64]('time', 1, -99999, min_val=0, max_val=1,
                                             oras_name="time_counter", era_name="time")


class Variable(Enum):
    pass


class ORASVariable(Variable):
    COMPLETE_OCEAN_HEAT_CONTENT = VariableInfo("ocean_heat_content_for_the_total_water_column", True, "sohtcbtm")
    OCEAN_HEAT_CONTENT_300M = VariableInfo("ocean_heat_content_for_the_upper_300m", True, "sohtc300")


class ERAATMOSVariable(Variable):
    V_WIND = VariableInfo("v_component_of_wind", False, "v")
    U_WIND = VariableInfo("u_component_of_wind", False, "u")
    TEMP = VariableInfo("temperature", False, "t")
    HUMIDITY = VariableInfo("specific_humidity", False, "z")
    GEOPOTENTIAL = VariableInfo("geopotential", False, "q")


class ERASURFACEVariable(Variable):
    V_WIND_10M = VariableInfo("10m_v_component_of_wind", True, "u10")
    U_WIND_10M = VariableInfo("10m_u_component_of_wind", True, "v10")
    SURFACE_TEMP = VariableInfo("2m_temperature", True, "t2m")
    SEA_SURFACE_TEMP = VariableInfo("sea_surface_temperature", True, "sst")
    SOIL_TEMP_LV1 = VariableInfo("soil_temperature_level_1", True, "stl1")
    HEAT_FLUX = VariableInfo("surface_latent_heat_flux", True, "slhf")
    TOTAL_PRECIPITATION = VariableInfo("total_precipitation", True, "tp")
    VOL_SOIL_WATER_LV2 = VariableInfo("volumetric_soil_water_layer_2", True, "swvl2")


VariableDict = {
    'ORAS': ORASVariable,
    'ERA5SURFACE': ERASURFACEVariable,
    'ERA5ATMOS': ERAATMOSVariable,
}
VARIABLES = []
for variable in VariableDict.values():
    VARIABLES.extend(list(variable))