import torch

from utils.hydra_config import BettiGuiderConfig

from ..loss_guider_base import LossGuiderBetti


class TopoGuider(LossGuiderBetti):
    possible_losses = ["TopoLoss"]

    def __init__(self, guider_config: BettiGuiderConfig):
        super().__init__(guider_config)

    def set_up_persistence_layer(self, size):
        self.loss_fn.set_up_persistence_layer(size)

    def guidance_loss(
        self, model_output: torch.Tensor, t: int, batch_idx: int, **kwargs: dict
    ):
        height, width = model_output.shape[-2], model_output.shape[-1]
        self.set_up_persistence_layer((height, width))

        betti_0, betti_1 = self.batched_betti(model_output.shape[0], **kwargs)
        loss = self.loss_fn(model_output, betti_0, betti_1)
        return loss


class TopoGuider_0(TopoGuider):
    possible_losses = ["TopoLoss_0"]

    def __init__(self, guider_config: BettiGuiderConfig):
        super().__init__(guider_config)

    def guidance_loss(
        self, model_output: torch.Tensor, t: int, batch_idx: int, **kwargs: dict
    ):
        height, width = model_output.shape[-2], model_output.shape[-1]
        self.set_up_persistence_layer((height, width))

        betti_0, _ = self.batched_betti(
            model_output.shape[0], only_betti_0=True, **kwargs
        )
        loss = self.loss_fn(model_output, betti_0)
        return loss
