import unittest
from src.Model.score_torch import *


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.lat_weights = torch.tensor([[1.0], [1.0]])  # lat x 1

    def test_compute_weighted_rmse_hight_error(self):
        forecast = torch.tensor([[[[[2.0, 2.0], [2.0, 2.0]]]]])
        labels = torch.tensor([[[[[1.0, 1.0], [float("nan"), 1.0]]]]])
        result = Score().compute_weighted_rmse(forecast, labels, self.lat_weights)
        expected = torch.tensor(1.0)
        self.assertTrue(
            torch.isclose(result, expected), f"Expected {expected}, got {result}"
        )

    def test_compute_weighted_rmse_no_error(self):
        forecast = torch.tensor([[[[[2.0, 2.0], [2.0, 2.0]]]]])
        labels = torch.tensor([[[[[2.0, 2.0], [float("nan"), 2.0]]]]])
        result = Score().compute_weighted_rmse(forecast, labels, self.lat_weights)
        expected = torch.tensor(0.0)
        self.assertTrue(
            torch.isclose(result, expected), f"Expected {expected}, got {result}"
        )

    def test_compute_weighted_acc_no_error(self):
        forecast = torch.tensor([[[[[2.0, 2.0], [2.0, 3]]]]])
        clim = torch.tensor([[[[[2, 3], [2, 2]]]]])
        labels = torch.tensor([[[[[2.0, 2], [float("nan"), 3]]]]])
        result = Score().compute_weighted_acc(forecast, labels, clim, self.lat_weights)
        expected = torch.tensor(1.0)  # Adjust this based on accurate calculation
        self.assertTrue(
            torch.isclose(result, expected, atol=1e-6),
            f"Expected {expected}, got {result}",
        )

    def test_compute_weighted_acc_hight_error(self):
        forecast = torch.tensor([[[[[200.0, 200.0], [200.0, 200]]]]])
        clim = torch.tensor([[[[[2, 3], [2, 2]]]]])
        labels = torch.tensor([[[[[2.0, 2], [float("nan"), 3]]]]])
        result = Score().compute_weighted_acc(forecast, labels, clim, self.lat_weights)
        expected = torch.tensor(0.0)  # Adjust this based on accurate calculation
        self.assertTrue(
            torch.isclose(result, expected, atol=1e-1),
            f"Expected {expected}, got {result}",
        )

    def test_compute_weighted_mae_hight_error(self):
        forecast = torch.tensor([[[[[1.0, 1.0], [1.0, 1.0]]]]])
        labels = torch.tensor([[[[[2.0, 2.0], [float("nan"), 2.0]]]]])
        result = Score().compute_weighted_mae(forecast, labels, self.lat_weights)
        expected = torch.tensor(1.0)
        self.assertTrue(
            torch.isclose(result, expected), f"Expected {expected}, got {result}"
        )

    def test_compute_weighted_mae_no_error(self):
        forecast = torch.tensor([[[[[2.0, 2.0], [5.0, 2.0]]]]])
        labels = torch.tensor([[[[[2.0, 2.0], [float("nan"), 2.0]]]]])
        result = Score().compute_weighted_mae(forecast, labels, self.lat_weights)
        expected = torch.tensor(0.0)
        self.assertTrue(
            torch.isclose(result, expected), f"Expected {expected}, got {result}"
        )
