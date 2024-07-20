import wandb

from src.ModelEvaluator import ModelEvaluator
from src.sweep_config import getSweepID


def start_eval():
    with wandb.init() as run:
        evaluator = ModelEvaluator(
            model_path="/Users/xgxtphg/Documents/git/FuXiClimatePrediction/models/epoch=169-step=65579.ckpt",
            run=run,
            dataset_type="val",
            data_dir="/Users/xgxtphg/Documents/git/FuXiClimatePrediction/data/",
        )

    results = evaluator.evaluate(batch_size=2)
    print(results)


if __name__ == "__main__":
    wandb.agent(getSweepID(), start_eval)
