import torch

from loss import single_loss_fn
from utils.helper import check_topofeatures, get_fixed_betti_numbers
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

        self.fixed_betti_numbers = guider_config.fixed_betti_numbers
        self.topo_features = check_topofeatures(
            guider_config.topo_features, guider_config.num_classes
        )
        self.betti_0, self.betti_1 = get_fixed_betti_numbers(
            self.topo_features, guider_config.num_classes
        )

        if guider_config.loss:
            loss_name = next(iter(guider_config.loss.loss_fns_config.keys()))

            if self.check_loss_option(loss_name):
                self.loss_fn = single_loss_fn(guider_config.loss)
                self.loss_name = loss_name
            else:
                raise ValueError(
                    f"Loss function {loss_name} is not supported for the loss guider {guider_config.name}"
                )
        else:
            raise ValueError(
                f"Loss function not specified for the loss guider {guider_config.name}"
            )

    def check_loss_option(self, name: str):
        if hasattr(self, "possible_losses"):
            losses = getattr(self, "possible_losses")
            return name in losses
        else:
            return True

    def batched_betti(
        self, batch_size: int, only_betti_0: bool = False, **kwargs: dict
    ):
        if self.fixed_betti_numbers:
            betti_0_batched = self.betti_0.unsqueeze(0).repeat(batch_size, 1)
            betti_1_batched = self.betti_1.unsqueeze(0).repeat(batch_size, 1)
        else:
            betti_0_batched = kwargs.get("betti_0", None)
            betti_1_batched = kwargs.get("betti_1", None)

        if betti_0_batched is None:
            raise ValueError("betti_0 must be provided")

        if betti_1_batched is None and not only_betti_0:
            raise ValueError("betti_1 must be provided")

        if len(betti_0_batched.shape) == 1:
            betti_0_batched = betti_0_batched.unsqueeze(1)

        if only_betti_0:
            return betti_0_batched, None

        if len(betti_1_batched.shape) == 1:
            betti_1_batched = betti_1_batched.unsqueeze(1)

        return betti_0_batched, betti_1_batched
