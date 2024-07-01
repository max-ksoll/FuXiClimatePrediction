from datetime import datetime

import yaml

import wandb


def get_sweep():
    executionTime = datetime.now().strftime("%d/%m/%Y, %H:%M")
    name = str("1st_test " + executionTime)
    sweep_config = {"method": "grid", "name": name}
    # metric = {"name": "mse", "goal": "minimize"}
    # sweep_config["metric"] = metric
    return sweep_config


def get_parameters_dict():
    parameters_dict = {
        "batch_size": {"value": 1},
        "init_learning_rate": {"value": 2.5e-4},
        "devices": {"value": 1},
        "num_nodes": {"value": 1},
        "model_parameter": {
            "value": {"channel": 32, "transformer_blocks": 1, "heads": 1}
        },
        "autoregression_steps_epochs": {
            "value": [
                {"epochs": 10, "steps": 1},
            ]
        },
    }
    return parameters_dict


def getSweepID():
    sweep_config = get_sweep()
    sweep_config["parameters"] = get_parameters_dict()
    sweep_config["name"] = sweep_config["name"]
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project="FuXiClimPred",
    )
    return sweep_id
