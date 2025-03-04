import torch

from loss import single_loss_fn
from utils.helper import check_topofeatures
from utils.hydra_config import BettiGuiderConfig, GuiderConfig


class LossGuider:
    """
    Base Loss Guider class. This class is a helper class for the loss guidance.
    It defines and computes the loss used in the guidance step. It also contains any logic that is needed to modify the model output and to create a pseudo ground truth which can than be used to compute the loss.
    """

    def __init__(self, guider_config: GuiderConfig):
        self.guider_config = guider_config
        self.num_classes = guider_config.num_classes

    def pseudo_gt(self, x_softmax: torch.Tensor, t: int = None, batch_idx: int = None):
        raise NotImplementedError("Pseudo GT not implemented")

    def guidance_loss(
        self, model_output: torch.Tensor, t: int = None, batch_idx: int = None
    ):
        raise NotImplementedError("Guidance loss not implemented")


class LossGuiderBetti(LossGuider):
    """
    Superclass for all loss guiders optimizing the betti number metric/error.
    """

    def __init__(self, guider_config: BettiGuiderConfig):
        super().__init__(guider_config)

        self.topo_features = check_topofeatures(
            guider_config.topo_features, guider_config.num_classes
        )

        if guider_config.loss:
            self.loss_fn = single_loss_fn(guider_config.loss)
            self.loss_name = next(iter(guider_config.loss.loss_fns_config.keys()))
            print(f"loss_name: {self.loss_name}")
        else:
            from torch.nn import CrossEntropyLoss

            self.loss_fn = CrossEntropyLoss()
            self.loss_name = "CrossEntropyLoss"
