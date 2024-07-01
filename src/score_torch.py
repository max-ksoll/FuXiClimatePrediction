import torch
import numpy as np


class WeightedMetrics:
    def __init__(self, lat_weights):
        self.lat_weights = lat_weights

    def compute_weighted_rmse(self, forecast, labels):
        mask = ~torch.isnan(labels)
        error = forecast - labels
        rmse = torch.sqrt((error**2 * self.lat_weights * mask).mean(dim=(-1, -2)))
        return rmse.mean()

    def compute_weighted_acc(self, forecast, labels, clim):
        mask = ~torch.isnan(labels)

        forecast_error = (forecast - clim) * self.lat_weights * mask
        label_error = (labels - clim) * self.lat_weights * mask

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

    def compute_weighted_mae(self, forecast, labels):
        mask = ~torch.isnan(labels)
        error = forecast - labels
        mae = (torch.abs(error) * self.lat_weights * mask).mean()
        return mae


if __name__ == "__main__":
    label = torch.rand((1, 1, 25, 121, 240))
    forecast = label + torch.rand((1, 1, 25, 121, 240))
    clim = torch.rand((25, 121, 240))
    import numpy as np

    lat_weights = torch.Tensor(np.cos(np.deg2rad(np.linspace(-90, 90, 121))))[:, None]

    metrics = WeightedMetrics(lat_weights)
    print(metrics.compute_weighted_rmse(forecast, label))
    print(metrics.compute_weighted_rmse_neu(forecast, label))
