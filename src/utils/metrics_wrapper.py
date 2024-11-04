import monai.metrics
import torch
import logging

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


    

