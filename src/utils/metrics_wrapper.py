import monai.metrics
import torch
import logging
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from monai.networks.utils import one_hot

import numpy as np


logger = logging.getLogger(__name__)


class ClassWiseDiceMetric(monai.metrics.DiceMetric):
    num_classes: int = None

    def __init__(self, **kwargs):
        if "num_classes" not in kwargs:
            raise ValueError(
                "num_classes must be specified for the metric class 'ClassWiseDiceMetric'"
            )

        num_classes = kwargs.get("num_classes", None)
        del kwargs["num_classes"]
        self.logging_names = [f"dice_class_{i}" for i in range(1, num_classes)]

        if not "include_background" in kwargs or kwargs["include_background"]:
            self.logging_names.insert(0, "dice_background")

        self.logging_names.append("DiceMetric")  # mean

        super().__init__(**kwargs)

        self.num_classes = num_classes

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor):
        scores = super().__call__(
            one_hot(y_pred, num_classes=self.num_classes),
            one_hot(y, num_classes=self.num_classes),
        )

        metric_list = [scores[:, i] for i in range(len(self.logging_names) - 1)]
        if self.ignore_empty:
            metric_list.append(self.average_over_classes_ignore_nan(scores))
        else:
            metric_list.append(scores.mean(dim=1))
        return metric_list

    def average_over_classes_ignore_nan(self, scores: torch.Tensor):
        not_nan_map = ~torch.isnan(scores)
        scores = torch.where(not_nan_map, scores, 0)

        return torch.sum(scores, dim=1) / not_nan_map.sum(dim=1)


# class DiceMetric(monai.metrics.DiceMetric):
#     num_classes: int = None

#     def __init__(self, **kwargs):
#         self.num_classes = kwargs.get("num_classes", None)
#         super().__init__(**kwargs)

#     def __call__(self, y_pred: torch.Tensor, y: torch.Tensor):
#         scores = super().__call__(
#             one_hot(y_pred, num_classes=self.num_classes),
#             one_hot(y, num_classes=self.num_classes),
#         )
#         scores_over_samples = torch.mean(scores, dim=1)
#         return scores_over_samples


class DiceMetric(monai.metrics.DiceMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor):
        return super().__call__(y_pred, y)


class HausdorffDistanceMetric2(monai.metrics.HausdorffDistanceMetric):
    num_classes: int

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor):

        def one_hot_encode(
            tensor: torch.Tensor, one_hot_shape: tuple[int, int, int, int]
        ):
            with torch.no_grad():
                tensor = tensor.type(torch.int64)
                one_hot_tensor = torch.zeros(
                    one_hot_shape, device=tensor.device
                ).scatter_(1, tensor, 1)
            return one_hot_tensor

        if self.num_classes is None:
            logger.warning("num_classes not set, defaulting to 2")
            self.num_classes = 2

        num_samples = y_pred.shape[0]

        one_hot_shape = (
            y_pred.shape[0],
            self.num_classes,
            y_pred.shape[2],
            y_pred.shape[3],
        )
        y_pred = one_hot_encode(y_pred, one_hot_shape)
        y = one_hot_encode(y, one_hot_shape)

        hd_per_class = super().__call__(y_pred, y)
        if self.num_classes > 2 or self.include_background:
            mean = torch.mean(hd_per_class, dim=1, keepdim=True)
            return mean
        else:
            return hd_per_class
        # return torch.zeros(num_samples, device="cpu")


class BettiNumberMetric:
    connectivity: int
    num_classes: int
    include_background: bool
    background_label: int = 0  # background label has to be 0!
    logging_names: list[str] = ["betti_number_0", "betti_number_1"]
    class_wise: bool = False

    def __init__(self, **kwargs):
        self.connectivity = kwargs.get("connectivity", 1)
        self.num_classes = kwargs.get("num_classes", 2)
        self.include_background = kwargs.get("include_background", False)

        self.class_wise = kwargs.get("class_wise", False)
        if self.class_wise:
            logging_names_0 = [
                f"betti_number_0_class_{i}" for i in range(self.num_classes)
            ]
            logging_names_0.append("betti_number_0")

            logging_names_1 = [
                f"betti_number_1_class_{i}" for i in range(self.num_classes)
            ]
            logging_names_1.append("betti_number_1")

            self.logging_names = logging_names_0 + logging_names_1

            print(f"logging_names: {self.logging_names}")

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor):
        if self.class_wise:
            return self.betti_number_0_1_class_wise(y_pred, y)
        else:
            return self.betti_number_0_1(y_pred, y)

    def betti_number_0_1(self, y_pred: torch.Tensor, y: torch.Tensor):

        y_pred_np = y_pred.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        betti_errors_0 = torch.empty((y_pred_np.shape[0],), device="cpu")
        betti_errors_1 = torch.empty((y_pred_np.shape[0],), device="cpu")
        for idx in range(y_pred_np.shape[0]):
            err_0 = 0.0
            err_1 = 0.0
            labels = self.extract_labels(y_pred_np[idx], y_np[idx])
            for label in labels:
                b0, b1 = self.betti_number_per_label(y_pred_np[idx], y_np[idx], label)
                err_0 += b0
                err_1 += b1

            if len(labels) > 0:
                betti_errors_0[idx] = err_0 / len(labels)
                betti_errors_1[idx] = err_1 / len(labels)
            else:
                betti_errors_0[idx] = 0.0
                betti_errors_1[idx] = 0.0

        return betti_errors_0, betti_errors_1

    def betti_number_0_1_class_wise(self, y_pred: torch.Tensor, y: torch.Tensor):
        scores = (
            torch.ones((y_pred.shape[0], len(self.logging_names)), device="cpu") * -1
        )

        y_pred_np = y_pred.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        betti_errors_0 = torch.empty((y_pred_np.shape[0],), device="cpu")
        betti_errors_1 = torch.empty((y_pred_np.shape[0],), device="cpu")
        for idx in range(y_pred_np.shape[0]):
            err_0 = 0.0
            err_1 = 0.0
            labels = self.extract_labels(y_pred_np[idx], y_np[idx])
            for label in labels:
                b0, b1 = self.betti_number_per_label(y_pred_np[idx], y_np[idx], label)
                err_0 += b0
                err_1 += b1

                scores[idx, label] = b0
                scores[idx, label + self.num_classes + 1] = b1

            if len(labels) > 0:
                scores[idx, self.num_classes] = err_0 / len(labels)
                scores[idx, -1] = err_1 / len(labels)

        scores = torch.where(scores != -1, scores, torch.nan)
        print(f"scores shape: {scores.shape}")
        return scores

    def betti_number_per_label(self, pred: np.ndarray, gt: np.ndarray, label: int):
        label_pred = (pred == label).squeeze(0)
        label_gt = (gt == label).squeeze(0)

        if not np.any(label_gt):
            # logger.warning(f"Label {label} not present in ground truth")
            b0_gt, b1_gt = 0, 0
        else:
            b0_gt, b1_gt = self.betti_numbers_image(label_gt)

        if not np.any(label_pred):
            # logger.warning(f"Label {label} not present in prediction")
            b0_pred, b1_pred = 0, 0
        else:
            b0_pred, b1_pred = self.betti_numbers_image(label_pred)

        return abs(b0_gt - b0_pred), abs(b1_gt - b1_pred)

    def betti_numbers_image(self, image: np.ndarray):
        """
        Computes the betti number 0 and betti number 1 for a single image.
        :param image: 2D numpy array
        :return: (int, int): tuple of betti number 0 and betti number 1
        """
        b0 = self.connected_components(image)
        skeleton = skeletonize(image)
        regions = regionprops(label(skeleton))
        if regions:
            euler_characteristic = regions[0].euler_number
        else:
            euler_characteristic = 0

        return b0, b0 - euler_characteristic

    def extract_labels(self, y_pred: np.ndarray, y: np.ndarray):
        """
        Adapted from https://github.com/CoWBenchmark/TopCoW_Eval_Metrics/blob/master/metric_functions.py#L18.
        :param y_pred: 2D numpy array
        :param y: 2D numpy array
        :return: list[int]: list of labels appearing in at least one of the two images
        """
        labels_gt = np.unique(y)
        labels_pred = np.unique(y_pred)
        labels = list(set().union(labels_gt, labels_pred))
        labels = [int(x) for x in labels]
        if not self.include_background and self.background_label in labels:
            labels.remove(self.background_label)
        return labels

    def connected_components(self, img: np.ndarray):
        """
        Computes the number of connected components in a 2D image.
        :param img: 2D numpy array
        :return: int: number of connected components
        """
        assert (img.ndim == 2, "Image must be 2D")
        assert (
            img.ndim >= self.connectivity,
            "Connectivity must be less than or equal to the dimension of the image",
        )

        _, num_components = label(img, connectivity=self.connectivity, return_num=True)
        return num_components


class BettiNumberMetric_0(BettiNumberMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logging_names = ["betti_number_0"]

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor):
        error, _ = super().__call__(y_pred, y)
        return error


class BettiNumberMetric_1(BettiNumberMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logging_names = ["betti_number_1"]

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor):
        _, error = super().__call__(y_pred, y)
        return error
