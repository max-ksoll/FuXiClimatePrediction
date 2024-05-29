import os.path
import logging

logger = logging.getLogger(__name__)


class Dataset:
    def __init__(self, data_dir: os.PathLike | str, start_year: int, end_year: int,
                 force_download: bool = False):
        pass