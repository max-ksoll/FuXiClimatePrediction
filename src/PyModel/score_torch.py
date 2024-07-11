from typing import Dict, Tuple

import torch


def weighted_rmse(
    forecast: torch.Tensor,
    labels: torch.Tensor,
    lat_weights: torch.Tensor,
    reduce_axis: int | Tuple[int, ...] = None,
) -> torch.Tensor:
    mask = ~torch.isnan(labels)
    error = forecast - labels
    error[~mask] = 0
    weighted_squared_error = (error**2) * lat_weights
    rmse = torch.sqrt(weighted_squared_error.sum(reduce_axis) / mask.sum(reduce_axis))
    return rmse


def weighted_acc(
    forecast: torch.Tensor,
    labels: torch.Tensor,
    lat_weights: torch.Tensor,
    clima_mean: torch.Tensor,
    reduce_axis: int | Tuple[int, ...] = None,
) -> torch.Tensor:
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
        (forecast_error - forecast_mean_error) * (label_error - label_mean_error)
    )

    lower_left = torch.sqrt(torch.mean((forecast_error - forecast_mean_error) ** 2))

    lower_right = torch.sqrt(torch.mean((label_error - label_mean_error) ** 2))

    return torch.mean(upper / (lower_left * lower_right), dim=reduce_axis)


def weighted_mae(
    forecast: torch.Tensor,
    labels: torch.Tensor,
    lat_weights: torch.Tensor,
    reduce_axis: int | Tuple[int, ...] = None,
) -> torch.Tensor:
    mask = ~torch.isnan(labels)
    error = forecast - labels
    weighted_error = torch.abs(error) * lat_weights
    weighted_error[~mask] = 0
    mae = weighted_error.sum(dim=reduce_axis) / mask.sum(dim=reduce_axis)
    return mae
