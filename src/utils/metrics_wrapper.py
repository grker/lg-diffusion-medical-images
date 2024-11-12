import monai.metrics
import torch
import logging
from skimage.measure import label
import numpy as np


logger = logging.getLogger(__name__)


class DiceMetric(monai.metrics.DiceMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor):
        return super().__call__(y_pred, y)
    

class HausdorffDistanceMetric(monai.metrics.HausdorffDistanceMetric):
    num_classes: int=None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor):
        
        def one_hot_encode(tensor: torch.Tensor, one_hot_shape: tuple[int, int, int, int]):
            tensor = tensor.type(torch.int64)
            return torch.zeros(one_hot_shape, device=tensor.device).scatter_(1, tensor, 1)

        if self.num_classes is None:
            logger.warning("num_classes not set, defaulting to 2")
            self.num_classes = 2
        
        one_hot_shape = (y_pred.shape[0], self.num_classes, y_pred.shape[2], y_pred.shape[3])
        y_pred = one_hot_encode(y_pred, one_hot_shape)
        y = one_hot_encode(y, one_hot_shape)

        hd_per_class = super().__call__(y_pred, y)
        if self.num_classes > 2 or self.include_background:
            return torch.mean(hd_per_class, dim=1, keepdim=True)
        else:
            return hd_per_class    


class BettiNumberMetric():
    number: int
    connectivity: int
    num_classes: int
    include_background: bool
    background_label: int=0 # background label has to be 0!

    def __init__(self, **kwargs):
        self.number = kwargs.get("number", 0)
        self.connectivity = kwargs.get("connectivity", 1)
        self.num_classes = kwargs.get("num_classes", 2)
        self.include_background = kwargs.get("include_background", False)

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor):
        if self.number == 0:
            return self.betti_number_0(y_pred, y)
        else:
            raise ValueError(f"Betti number {self.number} not supported")
    
    def betti_number_0(self, y_pred: torch.Tensor, y: torch.Tensor):
        y_pred_np = y_pred.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        betti_errors = torch.empty((y_pred_np.shape[0],))
        
        for idx in range(y_pred_np.shape[0]):
            err = 0.0
            labels = self.extract_labels(y_pred_np[idx], y_np[idx])
            for label in labels:
                err += self.betti_number_0_label(y_pred_np[idx], y_np[idx], label)

            if len(labels) > 0:
                betti_errors[idx] = err / len(labels)
            else:
                betti_errors[idx] = 0.0

        return betti_errors
    

    def extract_labels(self, y_pred: np.ndarray, y: np.ndarray):
        """
        Adapted from https://github.com/CoWBenchmark/TopCoW_Eval_Metrics/blob/master/metric_functions.py#L18.
        """
        labels_gt = np.unique(y)
        labels_pred = np.unique(y_pred)
        labels = list(set().union(labels_gt, labels_pred))
        labels = [int(x) for x in labels]
        if not self.include_background and self.background_label in labels:
            labels.remove(self.background_label)
        return labels
    
    def betti_number_0_label(self, pred: np.ndarray, gt: np.ndarray, label: int):
        label_pred = (pred == label).squeeze(0)
        label_gt = (gt == label).squeeze(0)

        if not np.any(label_gt):
            # logger.warning(f"Label {label} not present in ground truth")
            num_components_gt = 0
        else:
            num_components_gt = self.connected_components(label_gt)

        if not np.any(label_pred):
            # logger.warning(f"Label {label} not present in prediction")
            num_components_pred = 0
        else:
            num_components_pred = self.connected_components(label_pred)


        return abs(num_components_gt - num_components_pred)
        

    def connected_components(self, img: np.ndarray):
        assert(img.ndim == 2, "Image must be 2D")
        assert(img.ndim >= self.connectivity, "Connectivity must be less than or equal to the dimension of the image")

        _, num_components = label(img, connectivity=self.connectivity, return_num=True)
        return num_components
    
