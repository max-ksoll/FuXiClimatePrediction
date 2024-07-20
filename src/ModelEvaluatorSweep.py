import os

import wandb

from src.ModelEvaluator import ModelEvaluator
from src.sweep_config import getSweepID


def start_eval():
    with wandb.init() as run:
        data_dir = os.environ.get("DATA_PATH", False)
        if not data_dir:
            raise ValueError("DATA_PATH muss in dem .env File gesetzt sein!")
        evaluator = ModelEvaluator(
            model_path="/Users/xgxtphg/Documents/git/FuXiClimatePrediction/models/epoch=169-step=65579.ckpt",
            run=run,
            dataset_type="test",
            data_dir=data_dir,
            start_year=1958,
            end_year=1990,
            val_start_year=1991,
            val_end_year=2000,
            test_start_year=1991,
            test_end_year=2000,
        )
        evaluator.evaluate()


if __name__ == "__main__":
    wandb.agent(getSweepID(), start_eval)
