import os
import sys

sys.path.append(os.environ["MODULE_PATH"])

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
import cartopy
import cartopy.crs as ccrs

from src.Dataset.fuxi_dataset import FuXiDataset
from src.PyModel.fuxi_ligthning import FuXi

cartopy.config["pre_existing_data_dir"] = os.environ["CARTOPY_DIR"]


class ModelEvaluator:
    def __init__(
        self,
        model_path,
        eval_start_year,
        autoregression_years,
        data_path,
        output_path,
        fps=20,
        frame_size=(1920, 1080),
    ):
        model_ckpt = torch.load(model_path)
        self.model = FuXi.load_from_checkpoint(model_ckpt)
        self.model.eval()
        self.autoregression_steps = autoregression_years * 12
        self.model.autoregression_steps = self.autoregression_steps
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        os.makedirs(output_path, exist_ok=True)

        data_file_path = ModelEvaluator.find_file_with_start_year(
            data_path, eval_start_year
        )[0]
        mean_file_path = os.path.join(
            data_path, "mean_" + data_file_path.split("/")[-1]
        )
        self.ds = FuXiDataset(data_file_path, mean_file_path)

    @staticmethod
    def find_file_with_start_year(data_path, start_year):
        return_files = []
        for file in os.listdir(data_path):
            filename = file.split("/")[-1]
            is_zarr = ".zarr" in filename
            is_mean = "mean" in filename
            print(filename, is_mean, is_zarr)
            if is_zarr and not is_mean:
                filename = filename.split(".")[0]
                sy, ey = filename.split("_")
                if start_year in list(range(int(sy), int(ey) + 1)):
                    return_files.append(os.path.join(data_path, file))
        return return_files

    @staticmethod
    def plot_data(data, var_idx, time_idx):
        data = data[time_idx][var_idx]
        fig, ax = plt.subplots(
            figsize=(12, 8), subplot_kw={"projection": ccrs.PlateCarree()}
        )
        ax.coastlines()
        lons = np.linspace(-180, 180, data.shape[1])
        lats = np.linspace(-90, 90, data.shape[0])
        im = ax.pcolormesh(
            lons, lats, data, transform=ccrs.PlateCarree(), shading="auto"
        )
        plt.colorbar(im, ax=ax, orientation="vertical")
        name, level = FuXiDataset.get_var_name_and_level_at_idx(var_idx)
        ax.set_title(f"Var: {name} at Level: {level} at Time idx: {time_idx}")
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = cv2.cvtColor(data, cv2.COLOR_RGBA2BGR)
        plt.close(fig)
        return data

    @torch.no_grad()
    def evaluate(self):
        model_input = self.ds[0]
        model_out = self.model(model_input)

        for var_idx in range(1):
            name, level = FuXiDataset.get_var_name_and_level_at_idx(var_idx)
            path = os.path.join(self.output_path, f"{name}_{level}.mp4")
            out = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.frame_size
            )

            for step in range(self.autoregression_steps - 1):
                img = ModelEvaluator.plot_data(model_out, 0, step)
                bild_resized = cv2.resize(img, self.frame_size)
                out.write(bild_resized)
            out.release()


if __name__ == "__main__":
    model_path = os.environ["MODEL_FILE"]
    data_path = os.environ["DATA_PATH"]
    eval_start_year = os.environ["EVAL_START_YEAR"]
    autoregression_years = os.environ["AUTOREGRESSION_YEARS"]
    output_path = os.environ["OUTPUT_PATH"]

    model_evaluator = ModelEvaluator(
        model_path,
        eval_start_year,
        autoregression_years,
        data_path,
        output_path,
        fps=20,
        frame_size=(1920, 1080),
    )
