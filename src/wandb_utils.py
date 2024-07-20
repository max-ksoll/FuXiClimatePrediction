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
