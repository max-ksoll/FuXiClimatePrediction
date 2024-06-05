import logging
import os
import time
from typing import Set, List, Tuple, Callable

logger = logging.getLogger('Timing Logger')


def log_exec_time(func: Callable):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds")
        return result

    return wrapper


def get_nc_files(directory: os.PathLike | str) -> Set[str]:
    if not os.path.exists(directory):
        return {}
    return {f for f in os.listdir(directory) if f.endswith('.nc')}


def get_decades(start: int, end: int) -> List[int]:
    decades = (((end // 10) * 10) - ((start // 10) * 10)) // 10 + 1
    starting_decade = (start // 10) * 10
    return [starting_decade + 10 * i for i in range(decades)]


def get_date_strings(start_year: int, end_year: int) -> Tuple[List[str], List[int]]:
    date_strings = []
    decades = get_decades(start_year, end_year)
    for decade in decades:
        sy = decade
        ey = decade + 10
        if start_year > decade:
            sy = start_year
        if end_year < ey:
            ey = end_year + 1
        date = '/'.join([f'{year}{month:02d}01' for year in range(sy, ey) for month in range(1, 2)])
        date_strings.append(date)
    return date_strings, decades


def get_years_as_strings(start_year: int, end_year: int) -> List[str]:
    return [str(year) for year in range(start_year, end_year + 1)]


def get_month_as_strings(start_year: int, end_year: int) -> List[str]:
    return [f'{m:02d}' for m in range(1, 13)]
