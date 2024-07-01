import torch
import numpy as np


class WeightedMetrics:
    def compute_weighted_rmse(self, forecast, labels, lat_weights):
        mask = ~torch.isnan(labels)
        error = forecast - labels
        rmse = torch.sqrt((error**2 * lat_weights * mask).mean(dim=(-1, -2)))
        return rmse.mean()

    def compute_weighted_acc(self, forecast, labels, clim, lat_weights):
        mask = ~torch.isnan(labels)

        forecast_error = (forecast - clim) * lat_weights * mask
        label_error = (labels - clim) * lat_weights * mask

        forecast_mean_error = torch.mean(forecast_error, dim=(-1, -2))[
            :, :, :, None, None
        ]
        label_mean_error = torch.mean(label_error, dim=(-1, -2))[:, :, :, None, None]

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
        return torch.mean(upper / (lower_left * lower_right))

    def compute_weighted_mae(self, forecast, labels, lat_weights):
        mask = ~torch.isnan(labels)
        error = forecast - labels
        mae = (torch.abs(error) * lat_weights * mask).mean()
        return mae
