import numpy as np
import torch

from .metrics_wrapper import BettiNumberMetric


class TopologicalMetric:
    needed_inputs: list[str] = []


class DigitBettiNumberMetric(BettiNumberMetric, TopologicalMetric):
    needed_inputs = ["labels"]

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
        self.betti_numbers_0 = torch.tensor([1] * num_labels)
        self.betti_numbers_1 = torch.tensor([1, 0, 0, 0, 1, 0, 1, 0, 2, 1])

    def get_needed_inputs(self):
        return self.needed_inputs

    def __call__(self, y_pred: torch.Tensor, labels: torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
        labels = labels.detach().cpu()
        scores = torch.zeros((2, y_pred.shape[0]), device="cpu")

        betti_0_per_sample, betti_1_per_sample = self.get_betti_numbers(labels)

        for idx in range(y_pred.shape[0]):
            b0, b1 = self.betti_number_per_pred(y_pred[idx])

            scores[0, idx] = abs(b0 - betti_0_per_sample[idx])
            scores[1, idx] = abs(b1 - betti_1_per_sample[idx])

        return scores

    def get_betti_numbers(self, labels):
        """
        This function returns the betti numbers for each sample in the batch. This is achieved by summing the betti numbers of the labels.
        params:
            labels: torch.Tensor, shape (batch_size, num_labels)
        returns:
            betti_0_per_label: torch.Tensor, shape (batch_size,)
            betti_1_per_label: torch.Tensor, shape (batch_size,)
        """
        betti_numbers_0 = self.betti_numbers_0.to(labels.device)
        betti_numbers_1 = self.betti_numbers_1.to(labels.device)

        betti_0_per_sample = (betti_numbers_0[labels] * (labels >= 0)).sum(dim=1)
        betti_1_per_sample = (betti_numbers_1[labels] * (labels >= 0)).sum(dim=1)

        return betti_0_per_sample.numpy(), betti_1_per_sample.numpy()

    def betti_number_per_pred(self, pred: torch.Tensor):
        pred = (pred == 1).squeeze(0)

        if not np.any(pred):
            return 0, 0

        b0, b1 = self.betti_numbers_image(pred)
        return b0, b1
