import torch
import os
import wandb
from src.Dataset.FuXiDataModule import FuXiDataModule
from src.PyModel.fuxi_ligthning import FuXi
from src.PyModel.score_torch import weighted_rmse, weighted_acc, weighted_mae
from src.sweep_config import get_sweep_config
from src.wandb_utils import get_optimizer_config


class ModelEvaluator:
    def __init__(self, clima_mean, dataloader, fig_path):
        self.dataloader = dataloader

    def set_dl(self, dataloader):
        self.dataloader = dataloader

    def reset(self):
        ...

    def update(self, ts, lat_weights, outs):
        ...

    def evaluate_batch(self, ts, lat_weights, outs, withImg=True):
        return {
            "acc": 0,
            "mae": 0,
            "rmse": 0,
            "img": {
                "average_difference_over_time": {
                    "specific_humidity": ["fig_path"],
                    "temperature": ["fig_path"],
                    "u_component_of_wind": ["fig_path"],
                    "v_component_of_wind": ["fig_path"],
                },
                "model_out_minus_clim": {
                    "specific_humidity": ["fig_path"],
                    "temperature": ["fig_path"],
                    "u_component_of_wind": ["fig_path"],
                    "v_component_of_wind": ["fig_path"],
                },
            },
        }


class ModelEvaluator2:
    def __init__(
        self,
        model_path=None,
        model=None,
        run=None,
        dataset_type="test",
        dm=None,
        data_dir="",
        autoregression_steps=1,
        start_year=None,
        end_year=None,
        val_start_year=None,
        val_end_year=None,
        test_start_year=None,
        test_end_year=None,
    ):
        self.model_path = model_path
        self.model = model
        self.run = run
        self.dataset_type = dataset_type
        self.autoregression_steps = autoregression_steps
        if model_path is not None:
            self.model = self.load_model(model_path)
        elif model is not None:
            self.model = model
        else:
            raise ValueError("Either model_path or model must be provided.")

        if dm is None and run is not None:
            self.dm = FuXiDataModule(
                data_dir=data_dir,
                start_year=start_year,
                end_year=end_year,
                val_start_year=val_start_year,
                val_end_year=val_end_year,
                test_start_year=test_start_year,
                test_end_year=test_end_year,
                config=run.config,
                skip_data_preparing=os.environ.get("SKIP_DATA_PREPARATION", False),
            )
        elif dm is not None:
            self.dm = dm
        else:
            raise NotImplementedError("Noch nicht fertig")

    def load_model(self, model_path, optimizer_config=None, autoregression_config=None):
        map_location = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        optimizer_config = optimizer_config or get_optimizer_config()
        autoregression_config = (
            autoregression_config or wandb.config["autoregression_steps_epochs"]
        )
        if os.path.isfile(model_path):
            model = FuXi.load_from_checkpoint(
                checkpoint_path=model_path,
                map_location=map_location,
                optimizer_config=optimizer_config,
                autoregression_config=autoregression_config,
            )
            return model
        else:
            raise FileNotFoundError(f"No model found at {model_path}")

    def get_data_loader(self):
        if self.dataset_type == "test":
            return self.dm.test_dataloader(
                autoregression_steps=self.autoregression_steps
            )
        elif self.dataset_type == "val":
            return self.dm.val_dataloader(
                autoregression_steps=self.autoregression_steps
            )
        else:
            raise ValueError("dataset_type must be either 'test' or 'val'")

    def evaluate(self):
        data_loader = self.get_data_loader()
        self.model.eval()
        all_rmses = []
        all_accs = []
        all_maes = []
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(data_loader):
                ts, lat_weights = sample_batched
                label = torch.clone(ts[:, 2:, :, :, :])
                args = [ts, lat_weights]
                outs = self.model.forward(
                    args,
                    autoregression_steps=self.autoregression_steps,
                )
                rmse = weighted_rmse(outs, label, lat_weights)
                all_rmses.append(rmse.item())
                mae = weighted_mae(outs, label, lat_weights)
                all_maes.append(mae.item())
                if hasattr(data_loader.dataset, "get_clima_mean"):
                    clima_mean = data_loader.dataset.get_clima_mean()
                    acc = weighted_acc(outs, label, lat_weights, clima_mean.to(outs))
                    all_accs.append(acc.item())
        results = {
            f"weighted_rmse_{self.dataset_type}": sum(all_rmses) / len(all_rmses),
            f"weighted_acc_{self.dataset_type}": sum(all_accs) / len(all_accs)
            if len(all_accs) > 0
            else None,
            f"weighted_mae_{self.dataset_type}": sum(all_maes) / len(all_maes),
        }
        if self.run is not None:
            wandb.log(results)
        return results
