import torch


class Score:
    def compute_weighted_rmse(self, forecast, labels, lat_weights):
        mask = ~torch.isnan(labels)
        error = (forecast - labels).nan_to_num(nan=0)
        weighted_squared_error = (error**2) * lat_weights
        rmse = torch.sqrt(weighted_squared_error.sum() / mask.sum())
        return rmse

    def compute_weighted_acc(self, forecast, labels, clim, lat_weights):
        # input (bs, auto, 25, 121, 240)
        # clim (25, 121, 240)
        # lat_weights (121, 1)
        mask = ~torch.isnan(labels)
        forecast_error = (forecast - clim) * lat_weights
        forecast_error[~mask] = 0  # Set forecast errors to 0 where mask is True
        label_error = (labels - clim) * lat_weights
        label_error = label_error.nan_to_num(nan=0)

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

    def compute_weighted_mae(self, forecast, labels, lat_weights):
        mask = ~torch.isnan(labels)
        error = forecast - labels
        mae = ((torch.abs(error) * lat_weights).nan_to_num(nan=0)).sum() / mask.sum(
            dim=(-1, -2)
        )
        return mae
