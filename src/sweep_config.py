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
            "value": {"channel": 2, "transformer_blocks": 1, "heads": 1}
        },
        "autoregression_steps_epochs": {
            "value": {
                1: 1,  # until epoch 10 -> 1 Autoregression step
                2: 2,  # until epoch 20 -> 2 Autoregression step
                3: 4,  # until epoch 20 -> 2 Autoregression step
                -1: 6  # otherwise 4
            }
        },
        "max_epochs": {
            "value": 50
        }
    }
    return parameters_dict


def getSweepID():
    sweep_config = get_sweep()
    sweep_config["parameters"] = get_parameters_dict()
    sweep_config["name"] = sweep_config["name"]
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project="Test",
    )
    return sweep_id
