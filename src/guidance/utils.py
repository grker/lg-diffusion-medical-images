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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(
        self,
        prediction: torch.Tensor,
        intervals_comp_0: torch.Tensor,
        intervals_comp_1: torch.Tensor,
        good_intervals_0: list[int],
        good_intervals_1: list[int],
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

        loss_0 = self._compute_interval_diff(
            prediction, intervals_comp_0, good_intervals_0
        )
        loss_1 = self._compute_interval_diff(
            prediction, intervals_comp_1, good_intervals_1
        )

        return loss_0 + loss_1

    def _compute_interval_diff(
        self,
        prediction: torch.Tensor,
        intervals: list[list[torch.Tensor]],
        num_comps: list[int],
    ):
        num_samples = prediction.shape[0]
        num_classes = prediction.shape[1]

        assert num_classes == len(num_comps)

        num_intervals_list = [
            len(intervals[i][j]) for i in range(num_samples) for j in range(num_classes)
        ]
        concated_intervals = torch.cat(
            [torch.cat(intervals[i], dim=0) for i in range(num_samples)], dim=0
        )

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
                    torch.ones(min(num_comps[i % num_classes], num_intervals_list[i])),
                    torch.zeros(
                        num_intervals_list[i]
                        - min(num_comps[i % num_classes], num_intervals_list[i])
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

    # def _compute_interval_diff(
    #     self, prediction: torch.Tensor, intervals: torch.Tensor, good_intervals: int
    # ):
    #     """
    #     params:
    #         prediction: torch.Tensor, shape (batch_size, num_classes, height, width)
    #         intervals: torch.Tensor, shape (batch_size, num_classes, num_intervals, 2, 2)
    #     returns:
    #         torch.Tensor, shape (0,)
    #     """
    #     samples, classes, num_intervals, _, _ = intervals.shape
    #     good_intervals = min(good_intervals, num_intervals)

    #     birth_indices_x = intervals[:, :, :, 0, 0]
    #     birth_indices_y = intervals[:, :, :, 0, 1]
    #     death_indices_x = intervals[:, :, :, 1, 0]
    #     death_indices_y = intervals[:, :, :, 1, 1]

    #     sample_indices = [
    #         torch.full((classes * num_intervals,), i) for i in range(samples)
    #     ]
    #     sample_indices = torch.cat(sample_indices, dim=0)

    #     class_indices = [
    #         torch.full((num_intervals,), i % classes) for i in range(classes * samples)
    #     ]
    #     class_indices = torch.cat(class_indices, dim=0)

    #     interval_indices = [
    #         torch.arange(num_intervals) for i in range(samples * classes)
    #     ]
    #     interval_indices = torch.cat(interval_indices, dim=0)

    #     birth_values = prediction[
    #         sample_indices, class_indices, birth_indices_x, birth_indices_y
    #     ]
    #     death_values = prediction[
    #         sample_indices, class_indices, death_indices_x, death_indices_y
    #     ]

    #     interval_diff = (birth_values - death_values) ** 2

    #     interval_diff[:good_intervals] = 1 - interval_diff[:good_intervals]

    #     return interval_diff.sum()
