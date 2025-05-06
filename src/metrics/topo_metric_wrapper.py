import numpy as np
import torch

from .metrics_wrapper import BettiNumberMetric


class TopologicalMetric:
    needed_inputs: list[str] = []


class DigitBettiNumberMetric(BettiNumberMetric, TopologicalMetric):
    needed_inputs = ["betti_0", "betti_1"]
    logging_names = ["digit_betti_number_0", "digit_betti_number_1"]

    def __init__(
        self,
        connectivity: int = 1,
        num_labels: int = 10,
        include_background: bool = False,
        only_components: bool = False,
    ):
        self.connectivity = connectivity
        self.num_labels = num_labels
        self.include_background = include_background
        self.only_components = only_components
        self.only_components = False

    def get_needed_inputs(self):
        return self.needed_inputs

    def __call__(
        self, y_pred: torch.Tensor, betti_0: torch.Tensor, betti_1: torch.Tensor = None
    ):
        y_pred = y_pred.detach().cpu().numpy()
        scores = torch.zeros((len(self.logging_names), y_pred.shape[0]), device="cpu")

        for idx in range(y_pred.shape[0]):
            b0, b1 = self.betti_number_per_pred(y_pred[idx])

            # print(f"b0: {b0}, betti_0[idx]: {betti_0[idx]}")
            # print(f"b1: {b1}, betti_1[idx]: {betti_1[idx]}")

            scores[0, idx] = abs(b0 - betti_0[idx])

            if not self.only_components:
                scores[1, idx] = abs(b1 - betti_1[idx])

        return scores

    def betti_number_per_pred(self, pred: torch.Tensor):
        pred = (pred == 1).squeeze(0)

        if not np.any(pred):
            return 0, 0

        b0, b1 = self.betti_numbers_image(pred)
        return b0, b1


class ComponentBettiNumberMetric_0(DigitBettiNumberMetric):
    needed_inputs = ["betti_0"]
    logging_names = ["digit_betti_number_0"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.only_components = True
