import torch


class BirthDeathLoss(torch.nn.Module):
    def __init__(self, betti_numbers: dict, normalize_mode: str = "sum", **kwargs):
        super().__init__(**kwargs)

        self.good_intervals_0 = [betti_numbers[i][0] for i in range(len(betti_numbers))]
        self.good_intervals_1 = [betti_numbers[i][1] for i in range(len(betti_numbers))]

        self.normalize_mode = normalize_mode

    def forward(
        self,
        prediction: torch.Tensor,
        intervals_comp_0: list[list[torch.Tensor]] = None,
        intervals_comp_1: list[list[torch.Tensor]] = None,
    ):
        """
        params:
            prediction: torch.Tensor, shape (batch_size, num_classes, height, width)
            intervals_comp_0: list[list[torch.Tensor]], s.t. len(intervals_comp_0) == batch_size, the tensors have shape (num_intervals, 2, 2)
            intervals_comp_1: list[list[torch.Tensor]], s.t. len(intervals_comp_1) == batch_size, the tensors have shape (num_intervals, 2, 2)
        returns:
            torch.Tensor, shape (0,)
        """
        if intervals_comp_0 is not None:
            loss_0 = self._compute_interval_diff(
                prediction, intervals_comp_0, self.good_intervals_0
            )
        else:
            loss_0 = 0

        if intervals_comp_1 is not None:
            loss_1 = self._compute_interval_diff(
                prediction, intervals_comp_1, self.good_intervals_1
            )
        else:
            loss_1 = 0

        return loss_0 + loss_1

    def _compute_interval_diff(
        self,
        prediction: torch.Tensor,
        intervals: list[list[torch.Tensor]],
        num_comps: list[int],
    ):
        device = prediction.device
        num_samples = prediction.shape[0]
        num_classes = prediction.shape[1]

        assert num_classes == len(num_comps)

        num_intervals_list = [
            len(intervals[i][j]) for i in range(num_samples) for j in range(num_classes)
        ]

        concated_intervals = [
            torch.cat(intervals[i], dim=0) for i in range(num_samples)
        ]
        concated_intervals = torch.cat(concated_intervals, dim=0)

        sample_indices = [
            torch.full(
                (sum(num_intervals_list[i * num_classes : (i + 1) * num_classes]),), i
            )
            for i in range(num_samples)
        ]
        sample_indices = torch.cat(sample_indices, dim=0)

        class_indices = [
            torch.full((num_intervals_list[i],), i % num_classes)
            for i in range(num_classes * num_samples)
        ]
        class_indices = torch.cat(class_indices, dim=0)

        good_intervals = [
            torch.cat(
                (
                    torch.ones(
                        min(num_comps[i % num_classes], num_intervals_list[i]),
                        device=device,
                    ),
                    torch.zeros(
                        num_intervals_list[i]
                        - min(num_comps[i % num_classes], num_intervals_list[i]),
                        device=device,
                    ),
                ),
                dim=0,
            )
            for i in range(num_samples * num_classes)
        ]
        good_intervals = torch.cat(good_intervals, dim=0)

        print(f"intervals shape: {intervals.shape}")

        birth_indices_x = concated_intervals[:, 0, 0]
        birth_indices_y = concated_intervals[:, 0, 1]
        death_indices_x = concated_intervals[:, 1, 0]
        death_indices_y = concated_intervals[:, 1, 1]

        birth_values = prediction[
            sample_indices, class_indices, birth_indices_x, birth_indices_y
        ]
        death_values = prediction[
            sample_indices, class_indices, death_indices_x, death_indices_y
        ]

        interval_diff = (birth_values - death_values) ** 2
        return self.normalize(interval_diff, good_intervals, num_samples, num_classes)

    def normalize(
        self,
        interval_diff: torch.Tensor,
        good_intervals: torch.Tensor,
        num_samples: int = 1,
        num_classes: int = 1,
    ):
        interval_diff = torch.where(
            good_intervals == 1, 1 - interval_diff, interval_diff
        )

        if self.normalize_mode == "sum":
            return torch.sum(interval_diff)
        elif self.normalize_mode == "batch_mean":
            return torch.sum(interval_diff) / num_samples
        elif self.normalize_mode == "batch_class_mean":
            return torch.sum(interval_diff) / (num_samples * num_classes)
        elif self.normalize_mode == "interval_mean":
            return torch.mean(interval_diff)
        else:
            raise ValueError(f"Invalid normalize mode: {self.normalize_mode}")


class BirthDeathIntervalLoss(torch.nn.Module):
    def __init__(self, betti_numbers: dict, alpha: float = 0.5, beta: float = 0.5):
        super().__init__()

        self.good_intervals_0 = [betti_numbers[i][0] for i in range(len(betti_numbers))]
        self.good_intervals_1 = [betti_numbers[i][1] for i in range(len(betti_numbers))]

        print(f"comp 0: {self.good_intervals_0}")
        print(f"comp 1: {self.good_intervals_1}")

        self.alpha = alpha  # weighting factor of loss_0 and loss_1
        self.beta = beta  # weighting factor between good and bad intervals sum

    def forward(
        self,
        prediction: torch.Tensor,
        intervals_comp_0: list[list[torch.Tensor]] = None,
        intervals_comp_1: list[list[torch.Tensor]] = None,
    ):
        """
        params:
            prediction: torch.Tensor, shape (batch_size, num_classes, height, width)
            intervals_comp_0: list[list[torch.Tensor]], s.t. len(intervals_comp_0) == batch_size, the tensors have shape (num_intervals, 2, 2)
            intervals_comp_1: list[list[torch.Tensor]], s.t. len(intervals_comp_1) == batch_size, the tensors have shape (num_intervals, 2, 2)
        returns:
            torch.Tensor, shape (0,)
        """
        if intervals_comp_0 is not None:
            loss_0 = self._compute_interval_diff(
                prediction, intervals_comp_0, self.good_intervals_0
            )
        else:
            loss_0 = 0

        if intervals_comp_1 is not None:
            loss_1 = self._compute_interval_diff(
                prediction, intervals_comp_1, self.good_intervals_1
            )
        else:
            loss_1 = 0

        print(f"loss_0: {loss_0}")
        print(f"loss_1: {loss_1}")

        return self.alpha * loss_0 + (1 - self.alpha) * loss_1

    def _compute_interval_diff(
        self,
        prediction: torch.Tensor,
        intervals: list[list[torch.Tensor]],
        num_comps: list[int],
    ):
        num_samples = prediction.shape[0]
        num_classes = prediction.shape[1]

        assert num_classes == len(num_comps)
        assert num_samples == len(intervals)

        total_loss = 0.0
        for sample_idx in range(num_samples):
            sample_loss = 0.0
            for class_idx in range(num_classes):
                if intervals[sample_idx][class_idx].shape[0] > 0:
                    sample_loss += self.interval_diff_sum(
                        prediction[sample_idx, class_idx],
                        intervals[sample_idx][class_idx],
                        num_comps[class_idx],
                    )

            total_loss += sample_loss / num_classes

        return total_loss

    def interval_diff_sum(
        self, prediction: torch.Tensor, intervals: torch.Tensor, num_comps: int
    ):
        """
        params:
            prediction: torch.Tensor, shape (height, width)
            intervals: torch.Tensor, shape (num_intervals, 2, 2)
            num_comps: int, number of components in the interval
        returns:
            torch.Tensor, shape (0,)
        """
        total_intervals = intervals.shape[0]
        num_bad_intervals = max(total_intervals - num_comps, 0)
        num_good_intervals = min(total_intervals, num_comps)

        birth_values_x = intervals[:, 0, 0]
        birth_values_y = intervals[:, 0, 1]
        death_values_x = intervals[:, 1, 0]
        death_values_y = intervals[:, 1, 1]

        birth_values = prediction[birth_values_x, birth_values_y]
        death_values = prediction[death_values_x, death_values_y]

        interval_diff = (birth_values - death_values) ** 2

        loss = (
            self.beta * (1 - interval_diff[:num_good_intervals].mean())
            if num_good_intervals > 0
            else 0.0
        )

        loss += (
            (1 - self.beta) * interval_diff[num_good_intervals:].mean()
            if num_bad_intervals > 0
            else 0.0
        )

        return loss
