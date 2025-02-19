import torch


def likelyhood_temperature_scaling(
    x_softmax: torch.Tensor, likelyhood: torch.Tensor, alpha: float = 1.0
):
    """
    Likelihood temperature scaling of the softmax output.
    params:
        x_softmax: torch.Tensor, shape (batch_size, num_classes, height, width)
        likelyhood: torch.Tensor, shape (batch_size, num_classes, height, width)
        alpha: float, between 0 and 1, default 1.0
    returns:
        torch.Tensor, shape (batch_size, num_classes, height, width)
    """
    # second part does not make sense !!
    x_softmax = x_softmax / (1 - likelyhood) * alpha + torch.softmax(
        x_softmax, dim=1
    ) * (1 - alpha)

    return x_softmax / torch.sum(x_softmax, dim=1, keepdim=True)


class Birth_Death_Loss(torch.nn.Module):
    def __init__(self, betti_numbers: dict, **kwargs):
        super().__init__(**kwargs)

        self.good_intervals_0 = [betti_numbers[i][0] for i in range(len(betti_numbers))]
        self.good_intervals_1 = [betti_numbers[i][1] for i in range(len(betti_numbers))]

        print(f"good_intervals_0: {self.good_intervals_0}")
        print(f"good_intervals_1: {self.good_intervals_1}")

    def forward(
        self,
        prediction: torch.Tensor,
        intervals_comp_0: torch.Tensor = None,
        intervals_comp_1: torch.Tensor = None,
    ):
        """
        params:
            prediction: torch.Tensor, shape (batch_size, num_classes, height, width)
            intervals_comp_0: torch.Tensor, shape (batch_size, num_classes, num_intervals, 2, 2)
            intervals_comp_1: torch.Tensor, shape (batch_size, num_classes, num_intervals, 2, 2)
            good_intervals_0: list[int], s.t. len(good_intervals_0) == num_classes
            good_intervals_1: list[int], s.t. len(good_intervals_1) == num_classes
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
        interval_diff = torch.where(
            good_intervals == 1, 1 - interval_diff, interval_diff
        )

        return torch.sum(interval_diff)
