from datetime import datetime

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
        "optimizer_config_lr": {"value": 1e-5},
        "optimizer_config_betas": {"value": [(0.9, 0.95)]},
        "optimizer_config_weight_decay": {"value": 0.1},
        "optimizer_config_T_0": {"value": 2},
        "optimizer_config_eta_min": {"value": 1e-7},
        "optimizer_config_T_mult": {"value": 2},
        "devices": {"value": 1},
        "num_nodes": {"value": 1},
        "model_parameter": {
            "value": {"channel": 256, "transformer_blocks": 8, "heads": 8}
        },
        "autoregression_steps_epochs": {
            "value": {
                10: 1,  # until epoch n -> m autoregression step
                20: 2,
                30: 4,
                40: 6,
                50: 8,
                -1: 10,
            }
        },
        "max_epochs": {"value": 200},
    }
    return parameters_dict


def getSweepID():
    sweep_config = get_sweep()
    sweep_config["parameters"] = get_parameters_dict()
    sweep_config["name"] = sweep_config["name"]
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project="FuXiClimateFirstTests",
    )
    return sweep_id
