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
TASK_ID = os.environ.get("SLURM_ARRAY_TASK_ID", -1)
logging.basicConfig(level=logging.INFO)
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
        load_tensor_if_available=True,
        only_create_model_output=False,
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
        self.load_tensor_if_available = load_tensor_if_available
        model = model_path.split("/")[-1]
        self.tensor_path = os.path.join(
            output_path, f"{model}_{start_offset}_{autoregression_steps}.pt"
        )
        self.only_create_model_output = only_create_model_output

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

        ax.set_title(f"Var: {name} at Level: {level} at Time idx: {time_idx + 1}")
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = cv2.cvtColor(data, cv2.COLOR_RGBA2BGR)
        plt.close(fig)
        return data

    @staticmethod
    def get_slice_for_lat_lon(
        lat_start: float, lat_end: float, lon_start: float, lon_end: float, tensor_shape
    ):
        """
        Berechnet die Slice-Indizes für gegebene Breitengrad- und Längengradbereiche basierend auf der Tensorform.

        Parameters:
        lat_start (float): Startwert des Breitengrads in Grad (-90 bis +90).
        lat_end (float): Endwert des Breitengrads in Grad (-90 bis +90).
        lon_start (float): Startwert des Längengrads in Grad (-180 bis +180).
        lon_end (float): Endwert des Längengrads in Grad (-180 bis +180).
        tensor_shape (tuple): Form des Tensors, beinhaltet die Größen der Latitude- und Longitude-Dimensionen.

        Returns:
        Tuple[Tuple[int, int], Tuple[int, int]]: ((lat_start_idx, lat_end_idx), (lon_start_idx, lon_end_idx))
        """
        # Extrahiere die Größen der Latitude- und Longitude-Dimensionen
        # Angenommen, der Tensor hat die Form [Batch Size, Autoregression, Variablen, Latitude, Longitude]
        lat_size = tensor_shape[3]
        lon_size = tensor_shape[4]

        # Definiere die Bereiche von Latitude und Longitude im Tensor
        lat_min = -90.0
        lat_max = 90.0
        lon_min = -180.0
        lon_max = 180.0

        # Berechne die Schrittweite (Auflösung) für Latitude und Longitude
        lat_step = (lat_max - lat_min) / (lat_size - 1)
        lon_step = (lon_max - lon_min) / (lon_size - 1)

        # Berechne die Indizes für Latitude
        lat_start_idx = int(round((lat_start - lat_min) / lat_step))
        lat_end_idx = int(round((lat_end - lat_min) / lat_step))
        lat_start_idx = max(0, min(lat_start_idx, lat_size - 1))
        lat_end_idx = max(0, min(lat_end_idx, lat_size - 1))
        if lat_start_idx > lat_end_idx:
            lat_start_idx, lat_end_idx = lat_end_idx, lat_start_idx

        # Berechne die Indizes für Longitude
        lon_start_idx = int(round((lon_start - lon_min) / lon_step))
        lon_end_idx = int(round((lon_end - lon_min) / lon_step))
        lon_start_idx = max(0, min(lon_start_idx, lon_size - 1))
        lon_end_idx = max(0, min(lon_end_idx, lon_size - 1))
        if lon_start_idx > lon_end_idx:
            lon_start_idx, lon_end_idx = lon_end_idx, lon_start_idx

        return (lat_start_idx, lat_end_idx), (lon_start_idx, lon_end_idx)

    def create_videos(self, model_out, model_minus_correct):
        for variable in range(model_out.shape[2]):
            var_idx = variable if TASK_ID == -1 else 0
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

            for step in range(self.autoregression_steps):
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

    @torch.no_grad()
    def evaluate(self):
        model_out = self.get_model_out()
        if self.only_create_model_output:
            return

        if TASK_ID != -1:
            model_out = model_out[:, :, TASK_ID, :, :]
            model_out = model_out.unsqueeze(2)

        # bs x auto_step x var x lat x lon
        model_minus_correct = model_out.clone()
        correct = model_out.clone()

        for idx in range(self.autoregression_steps):
            if self.offset + idx >= len(self.dataset):
                model_minus_correct[:, idx:] = 0
                correct[:, idx:] = 0
                break
            model_minus_correct[:, idx] -= self.dataset[self.offset + idx][-1]
            correct[:, idx] = self.dataset[self.offset + idx][-1]

        self.dataset.denormalize(model_out)
        self.dataset.denormalize(model_minus_correct)

        self.create_videos(model_out, model_minus_correct)
        self.create_temp_curve(model_out, correct)
        self.create_el_nino_curve(model_out, correct)

    def create_temp_curve(self, model_out, correct):
        if not (TASK_ID == 4 or TASK_ID == -1):
            return

        temp_variable_idx = 4 if TASK_ID == -1 else 0
        (lat_start, lat_end), (
            lon_start,
            lon_end,
        ) = ModelEvaluator.get_slice_for_lat_lon(-180, 180, 30, 60, model_out.shape)
        x = np.arange(model_out.shape[1])
        pred = (
            model_out[
                0,
                :,
                temp_variable_idx,
                lat_start : lat_end + 1,
                lon_start : lon_end + 1,
            ]
            .mean(axis=(-1, -2))
            .numpy()
        )
        corr = (
            correct[
                0,
                :,
                temp_variable_idx,
                lat_start : lat_end + 1,
                lon_start : lon_end + 1,
            ]
            .mean(axis=(-1, -2))
            .numpy()
        )

        plt.figure(figsize=(40, 12))
        plt.plot(x, pred, label="Prediction")
        plt.plot(x, corr, label="Ground Truth")
        plt.savefig(os.path.join(self.output_path, "temp_curve.png"))

    def get_model_out(self) -> torch.Tensor:
        if os.path.exists(self.tensor_path) and self.load_tensor_if_available:
            model_out = torch.load(self.tensor_path, map_location=torch.device("cpu"))
        else:
            model_input: torch.Tensor = self.dataset[self.offset]
            model_input = model_input.unsqueeze(0)
            model_input = model_input.to(self.model.device)
            model_out = self.model(model_input, None).cpu()
            torch.save(model_out, self.tensor_path)

        return model_out

    @staticmethod
    def calculate_area_weights(latitudes):
        """
        Berechnet die Flächengewichte (areacello) basierend auf den Breitengraden.
        Die Fläche einer Zelle wird angenähert durch den Kosinus des Breitengrads.

        Parameters:
        latitudes (np.array): Ein Array von Breitengraden (in Grad), für die die Flächengewichte berechnet werden sollen.

        Returns:
        np.array: Ein 1D-Array von Flächengewichten.
        """
        # Konvertiere die Breitengrade in Radians
        lat_radians = np.radians(latitudes)
        # Kosinus des Breitengrads wird als Gewichte verwendet
        area_weights = np.cos(lat_radians)
        return area_weights

    def create_el_nino_curve(self, model_out, correct):
        if not (TASK_ID == 5 or TASK_ID == -1):
            return
        sst_idx = 5 if TASK_ID == -1 else 0

        (lat_start, lat_end), (
            lon_start,
            lon_end,
        ) = ModelEvaluator.get_slice_for_lat_lon(-5, 5, 190, 240, model_out.shape)
        pred = model_out[
            0,
            :,
            sst_idx,
            lat_start : lat_end + 1,
            lon_start : lon_end + 1,
        ].numpy()
        corr = correct[
            0,
            :,
            sst_idx,
            lat_start : lat_end + 1,
            lon_start : lon_end + 1,
        ].numpy()
        lats = np.linspace(
            -5, 5, lat_end - lat_start + 1
        )  # Breitengrade in der NINO 3.4 Region
        lons = np.linspace(
            -170, -120, lon_end - lon_start + 1
        )  # Längengrade in der NINO 3.4 Region
        area_weights = ModelEvaluator.calculate_area_weights(lats)
        area_weights_2d = np.tile(area_weights[:, np.newaxis], (1, len(lons)))

        tos_nino34_anom = pred - pred.mean(axis=0)

        weighted_anom_pred = np.average(
            tos_nino34_anom, weights=area_weights_2d, axis=(1, 2)
        )
        weighted_anom_corr = np.average(
            corr - corr.mean(axis=0), weights=area_weights_2d, axis=(1, 2)
        )

        x = np.arange(pred.shape[0])  # Zeitachse
        plt.figure(figsize=(12, 6))
        plt.plot(
            x,
            weighted_anom_pred,
            label="Weighted SST Anomaly Prediction (NINO 3.4)",
            color="r",
        )
        plt.plot(
            x,
            weighted_anom_corr,
            label="Weighted SST Anomaly Ground Truth (NINO 3.4)",
            color="b",
        )
        plt.axhline(0, color="black", linestyle="--")
        plt.title("SST Anomalies in the NINO 3.4 Region (El Niño/La Niña Indicator)")
        plt.xlabel("Time Steps")
        plt.ylabel("SST Anomaly (°C)")
        plt.legend()
        plt.savefig(os.path.join(self.output_path, "el_nino_curve.png"))


if __name__ == "__main__":
    model_path = os.environ["MODEL_FILE"]
    data_path = os.environ["DATA_PATH"]
    mean_data_path = os.environ["MEAN_DATA_PATH"]
    offset = int(os.environ["INFERENCE_START_OFFSET_MONTH_FROM_JAN_1958"])
    autoregression_steps = int(os.environ["AUTOREGRESSION_STEPS"])
    output_path = os.environ["OUTPUT_PATH"]
    fps = int(os.environ["FPS"])
    frame_size = eval(os.environ["FRAME_SIZE"])
    only_create_tensor = (
        os.environ.get("ONLY_CREATE_TENSORS", "False").lower() == "true"
    )

    dataset = FuXiDataset(data_path, mean_data_path)

    model_evaluator = ModelEvaluator(
        model_path,
        dataset,
        autoregression_steps,
        offset,
        output_path,
        fps=fps,
        frame_size=frame_size,
        only_create_model_output=only_create_tensor,
    )
    model_evaluator.evaluate()
