import torch


def compute_weighted_rmse(forecast, label, lat_weights):
    error = forecast - label
    rmse = torch.sqrt((error ** 2 * lat_weights).mean(dim=(-1, -2)))
    return rmse.mean()


def compute_weighted_acc(forecast, labels, lat_weights, clim):
    # input (bs, auto, 25, 121, 240)
    # clim (25, 121, 240)
    # lat_weights (121, 1)

    forecast_error = (forecast - clim) * lat_weights
    label_error = (labels - clim) * lat_weights

    forecast_mean_error = torch.mean(forecast_error, dim=(-1, -2))[:, :, :, None, None]
    label_mean_error = torch.mean(label_error, dim=(-1, -2))[:, :, :, None, None]

    upper = torch.mean((forecast_error - forecast_mean_error) * (label_error - label_mean_error), dim=(-1, -2))
    lower_left = torch.sqrt(torch.mean((forecast_error - forecast_mean_error) ** 2, dim=(-1, -2)))
    lower_right = torch.sqrt(torch.mean((label_error - label_mean_error) ** 2, dim=(-1, -2)))
    return torch.mean(upper / (lower_left * lower_right))


def compute_weighted_mae(forecast, labels, lat_weights):
    error = forecast - labels
    mae = (torch.abs(error) * lat_weights).mean()
    return mae


if __name__ == '__main__':
    label = torch.rand((1, 1, 25, 121, 240))
    forecast = label + torch.rand((1, 1, 25, 121, 240))
    clim = torch.rand((25, 121, 240))
    import numpy as np

    lat_weights = torch.Tensor(np.cos(np.deg2rad(np.linspace(-90, 90, 121))))[:, None]

    print(compute_weighted_acc(label, label, lat_weights, clim))
    print(compute_weighted_acc(-label, label, lat_weights, clim))
    print(compute_weighted_acc(forecast, label, lat_weights, clim))
