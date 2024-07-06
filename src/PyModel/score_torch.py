from typing import Dict

import torch


def weighted_rmse(forecast: torch.Tensor, labels: torch.Tensor, lat_weights: torch.Tensor) -> torch.Tensor:
    mask = ~torch.isnan(labels)
    error = (forecast - labels)
    error[~mask] = 0
    weighted_squared_error = (error ** 2) * lat_weights
    rmse = torch.sqrt(weighted_squared_error.sum() / mask.sum())
    return rmse


def weighted_acc(forecast: torch.Tensor, labels: torch.Tensor, lat_weights: torch.Tensor,
                 clima_mean: torch.Tensor) -> torch.Tensor:
    # input (bs, auto, 25, 121, 240)
    # clim (25, 121, 240)
    # lat_weights (121, 1)
    mask = ~torch.isnan(labels)
    if clima_mean.dim() == 1:
        clima_mean = clima_mean[:, None, None]
    forecast_error = (forecast - clima_mean) * lat_weights
    forecast_error[~mask] = 0  # Set forecast errors to 0 where mask is True
    label_error = (labels - clima_mean) * lat_weights
    label_error[~mask] = 0

    forecast_mean_error = (
            torch.sum(forecast_error, dim=(-1, -2))[:, :, :, None, None] / mask.sum()
    )
    label_mean_error = (
            torch.sum(label_error, dim=(-1, -2))[:, :, :, None, None] / mask.sum()
    )

    upper = torch.mean(
        (forecast_error - forecast_mean_error) * (label_error - label_mean_error),
        dim=(-1, -2),
    )
    lower_left = torch.sqrt(
        torch.mean((forecast_error - forecast_mean_error) ** 2, dim=(-1, -2))
    )
    lower_right = torch.sqrt(
        torch.mean((label_error - label_mean_error) ** 2, dim=(-1, -2))
    )
    assert torch.all(
        (lower_left * lower_right) != 0
    ), "lower_left * lower_right is zero, causing division by zero error."
    return torch.mean(upper / (lower_left * lower_right))


def weighted_mae(forecast: torch.Tensor, labels: torch.Tensor, lat_weights: torch.Tensor) -> torch.Tensor:
    mask = ~torch.isnan(labels)
    error = forecast - labels
    weighted_error = torch.abs(error) * lat_weights
    weighted_error[~mask] = 0
    mae = weighted_error.sum() / mask.sum()
    return mae

# def calculate_metrics(forecast: torch.Tensor, labels: torch.Tensor, lat_weights: torch.Tensor,
#                       clima_mean: torch.Tensor) -> Dict[str, torch.Tensor]:
#     return {
#         'rmse': compute_weighted_rmse(forecast, labels, lat_weights),
#         'acc': compute_weighted_acc(forecast, labels, lat_weights, clima_mean),
#         'mae': compute_weighted_mae(forecast, labels, lat_weights),
#     }
