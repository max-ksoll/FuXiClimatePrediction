import logging
import os
import sys
from functools import lru_cache

sys.path.append(os.environ["MODULE_PATH"])

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
import cartopy
import cartopy.crs as ccrs

from src.Dataset.fuxi_dataset import FuXiDataset
from src.PyModel.fuxi_ligthning import FuXi
from src.Dataset.dimensions import LEVEL_VARIABLES, SURFACE_VARIABLES

cartopy.config["pre_existing_data_dir"] = os.environ["CARTOPY_DIR"]
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    def __init__(
        self,
        model_path: os.PathLike | str,
        dataset: FuXiDataset,
        autoregression_steps: int,
        start_offset: int,
        output_path,
        fps=20,
        frame_size=(1920, 1080),
    ):
        self.model = FuXi.load_from_checkpoint(model_path)
        self.model.eval()
        self.autoregression_steps = autoregression_steps
        self.model.autoregression_steps = self.autoregression_steps + 2
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        os.makedirs(output_path, exist_ok=True)

        self.offset = start_offset
        self.dataset = dataset

        self.month_after_data = min(
            0, (2014 + 1 - 1958) * 12 - start_offset - autoregression_steps
        )

    @staticmethod
    @lru_cache
    def get_unit_for_var_name(var_name):
        variables = LEVEL_VARIABLES + SURFACE_VARIABLES
        variables = list(filter(lambda x: x.name == var_name, variables))
        return variables[0].unit

    @staticmethod
    def plot_data(data, var_idx, time_idx, d_min, d_max):
        data = data[0, time_idx, var_idx]
        fig, ax = plt.subplots(
            figsize=(12, 8), subplot_kw={"projection": ccrs.PlateCarree()}
        )
        ax.coastlines()
        lons = np.linspace(-180, 180, data.shape[1])
        lats = np.linspace(-90, 90, data.shape[0])
        im = ax.pcolormesh(
            lons,
            lats,
            data,
            transform=ccrs.PlateCarree(),
            shading="auto",
            vmin=d_min,
            vmax=d_max,
        )

        name, level = FuXiDataset.get_var_name_and_level_at_idx(var_idx)

        clb = plt.colorbar(im, ax=ax, orientation="vertical")
        clb.ax.set_ylabel(f"{ModelEvaluator.get_unit_for_var_name(name)}", rotation=270)

        ax.set_title(f"Var: {name} at Level: {level} at Time idx: {time_idx}")
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = cv2.cvtColor(data, cv2.COLOR_RGBA2BGR)
        plt.close(fig)
        return data

    @torch.no_grad()
    def evaluate(self):
        model_input: torch.Tensor = self.dataset[self.offset]
        model_input = model_input.unsqueeze(0)
        model_input = model_input.to(self.model.device)
        model_out = self.model(model_input, None).cpu()
        # bs x auto_step x var x lat x lon
        model_minus_correct = model_out.clone()

        for idx, elem in enumerate(iter(self.dataset)):
            if idx >= self.autoregression_steps:
                break
            model_minus_correct[:, idx] -= elem[-1]

        if self.month_after_data < 0:
            model_minus_correct[:, self.month_after_data] = 0

        self.dataset.denormalize(model_out)
        self.dataset.denormalize(model_minus_correct)

        for var_idx in range(35):
            name, level = FuXiDataset.get_var_name_and_level_at_idx(var_idx)
            path = os.path.join(self.output_path, f"{name}_{level}.mp4")
            diff_path = os.path.join(self.output_path, f"diff_{name}_{level}.mp4")
            out = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.frame_size
            )
            diff_out = cv2.VideoWriter(
                diff_path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.frame_size
            )

            out_min, out_max = (
                model_out[:, :, var_idx].min(),
                model_out[:, :, var_idx].max(),
            )
            diff_out_min, diff_out_max = (
                model_minus_correct[:, :, var_idx].min(),
                model_minus_correct[:, :, var_idx].max(),
            )

            for step in range(self.autoregression_steps - 1):
                img = ModelEvaluator.plot_data(
                    model_out, var_idx, step, out_min, out_max
                )
                bild_resized = cv2.resize(img, self.frame_size)
                out.write(bild_resized)

                img = ModelEvaluator.plot_data(
                    model_minus_correct, var_idx, step, diff_out_min, diff_out_max
                )
                bild_resized = cv2.resize(img, self.frame_size)
                diff_out.write(bild_resized)
            out.release()
            diff_out.release()


if __name__ == "__main__":
    model_path = os.environ["MODEL_FILE"]
    data_path = os.environ["DATA_PATH"]
    mean_data_path = os.environ["MEAN_DATA_PATH"]
    offset = int(os.environ["INFERENCE_START_OFFSET_MONTH_FROM_JAN_1958"])
    autoregression_steps = int(os.environ["AUTOREGRESSION_STEPS"])
    output_path = os.environ["OUTPUT_PATH"]
    fps = int(os.environ["FPS"])
    frame_size = eval(os.environ["FRAME_SIZE"])

    dataset = FuXiDataset(data_path, mean_data_path)

    model_evaluator = ModelEvaluator(
        model_path,
        dataset,
        autoregression_steps,
        offset,
        output_path,
        fps=fps,
        frame_size=frame_size,
    )
    model_evaluator.evaluate()
