import wandb


def get_specific_config(specific_key):
    optimizer_config = {}
    for key in wandb.config.keys():
        if key.startswith(specific_key):
            optimizer_config[key] = wandb.config[key]
    return optimizer_config


def get_optimizer_config():
    return get_specific_config("optimizer_config")


def get_model_parameter():
    return get_specific_config("model_parameter")


def log_eval_dict(model_eval, ds_type):
    log_data = {
        f"{ds_type}_img": {},
    }
    for eval_type in ["average_difference_over_time", "model_out_minus_clim"]:
        log_data[f"{ds_type}_img"][eval_type] = {}
        for var, paths in model_eval[f"img"][eval_type].items():
            log_data[f"{ds_type}_img"][eval_type][var] = [
                wandb.Image(path) for path in paths
            ]

    wandb.log(log_data)
