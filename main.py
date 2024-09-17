import logging

from src.Dataset.create_data import DataBuilder

logging.basicConfig(level=logging.DEBUG)
start_year = 1958
end_year = 2014
data_dir = "/home/stud1/data/fuxi"

builder = DataBuilder(data_dir, start_year, end_year)
builder.generate_data()
