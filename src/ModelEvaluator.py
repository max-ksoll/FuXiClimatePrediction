import torch
from torch.utils.data import DataLoader
import os
import wandb
import argparse

from src.Dataset.FuXiDataModule import FuXiDataModule
from src.PyModel.fuxi_ligthning import FuXi
from src.PyModel.score_torch import weighted_rmse
from src.sweep_config import getSweepID, get_sweep_config
from src.wandb_utils import get_model_parameter, get_optimizer_config


class ModelEvaluator:
    def __init__(
        self,
        model_path=None,
        model=None,
        run=None,
        dataset_type="test",
        dm=None,
        data_dir="",
        autoregression_steps=2,
    ):
        self.model_path = model_path
        self.model = model
        self.run = run
        self.dataset_type = dataset_type
        self.autoregression_steps = autoregression_steps

        print(get_sweep_config())
        if self.run:
            self.model = self.init_model(run)
        elif model_path is not None:
            self.model = self.load_model(model_path)
        elif model is not None:
            self.model = model
        else:
            raise ValueError("Either model_path or model must be provided.")

        if dm is None and run is not None:
            self.dm = FuXiDataModule(
                data_dir=data_dir,
                start_year=1958,
                end_year=1990,
                val_start_year=1991,
                val_end_year=2000,
                config=run.config,
                skip_data_preparing=os.environ.get("SKIP_DATA_PREPARATION", False),
            )
        elif dm is not None:
            self.dm = dm
        else:
            raise NotImplementedError("Noch nicht fertig")

    def load_model(self, model_path):
        map_location = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path, map_location=map_location)
            model = FuXi.load_from_checkpoint(
                checkpoint_path=model_path, map_location=map_location
            )
            return model
        else:
            raise FileNotFoundError(f"No model found at {model_path}")

    def init_model(self, run):
        config = run.config

        print("Creating Model")
        model_parameter = get_model_parameter()
        channels = model_parameter["model_parameter_channel"]
        transformer_blocks = model_parameter["model_parameter_transformer_blocks"]
        transformer_heads = model_parameter["model_parameter_heads"]
        optimizer_config = get_optimizer_config()
        model = FuXi(
            35,
            channels,
            transformer_blocks,
            transformer_heads,
            config.get("autoregression_steps_epochs"),
            optimizer_config=optimizer_config,
        )
        return model

    def get_data_loader(self):
        if self.dataset_type == "test":
            return self.dm.test_dataloader()
        elif self.dataset_type == "val":
            return self.dm.val_dataloader(
                autoregression_steps=self.autoregression_steps
            )
        else:
            raise ValueError("dataset_type must be either 'test' or 'val'")

    def evaluate(self, batch_size=32):
        data_loader = self.get_data_loader()
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for i_batch, sample_batched in enumerate(data_loader):
                print(i_batch)
                print(sample_batched)
                ts, lat_weights = sample_batched
                label = torch.clone(ts[:, 2:, :, :, :])
                args = [ts, lat_weights]
                outs = self.model.forward(
                    args,
                    autoregression_steps=self.autoregression_steps,
                )
                rmse = weighted_rmse(outs, label, lat_weights)
                print(rmse)

        # return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a model on the test or validation set"
    )
    parser.add_argument("--model_path", type=str, help="Path to the model checkpoint")
    parser.add_argument(
        "--use_wandb", action="store_true", help="Use WandB for logging"
    )
    parser.add_argument("--wandb_project", type=str, help="WandB project name")
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["test", "val"],
        default="test",
        help="Dataset type to use for evaluation",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for data loading"
    )
    args = parser.parse_args()

    evaluator = ModelEvaluator(
        model_path="/Users/xgxtphg/Documents/git/FuXiClimatePrediction/models/epoch=169-step=65579.ckpt",
        use_wandb=True,
        dataset_type="val",
    )

    results = evaluator.evaluate(batch_size=args.batch_size)
    print(results)
