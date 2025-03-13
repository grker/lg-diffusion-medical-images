import numpy as np
import torch

from .metrics_wrapper import BettiNumberMetric


class TopologicalMetric:
    needed_inputs: list[str] = []


class DigitBettiNumberMetric(BettiNumberMetric, TopologicalMetric):
    needed_inputs = ["betti_0", "betti_1"]

    def __init__(
        self,
        connectivity: int = 1,
        num_labels: int = 10,
        include_background: bool = False,
    ):
        self.connectivity = connectivity
        self.num_labels = num_labels
        self.include_background = include_background

        self.logging_names = [
            "digit_betti_number_labels_0",
            "digit_betti_number_labels_1",
        ]

    def get_needed_inputs(self):
        return self.needed_inputs

    def __call__(
        self, y_pred: torch.Tensor, betti_0: torch.Tensor, betti_1: torch.Tensor
    ):
        y_pred = y_pred.detach().cpu().numpy()
        betti_0 = betti_0.detach().cpu().numpy()
        betti_1 = betti_1.detach().cpu().numpy()
        scores = torch.zeros((2, y_pred.shape[0]), device="cpu")

        for idx in range(y_pred.shape[0]):
            b0, b1 = self.betti_number_per_pred(y_pred[idx])

            scores[0, idx] = abs(b0 - betti_0[idx])
            scores[1, idx] = abs(b1 - betti_1[idx])

        return scores

    def betti_number_per_pred(self, pred: torch.Tensor):
        pred = (pred == 1).squeeze(0)

        if not np.any(pred):
            return 0, 0

        b0, b1 = self.betti_numbers_image(pred)
        return b0, b1
