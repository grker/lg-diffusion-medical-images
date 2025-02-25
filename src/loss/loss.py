import monai.losses
import torch

import loss
from utils.hydra_config import LossConfig


def single_loss_fn(config: LossConfig):
    name = list(config.loss_fns_config.keys())[0]
    items = config.loss_fns_config[name]
    args = items.args if items.args is not None else {}

    if hasattr(torch.nn, name):
        return getattr(torch.nn, name)(**args)
    elif hasattr(monai.losses, name):
        return getattr(monai.losses, name)(**args)
    elif hasattr(loss, name):
        return getattr(loss, name)(**args)
    else:
        raise ValueError(f"Unknown loss function {name}")


def generate_loss_fns(config: LossConfig):
    loss_fns = {}
    scales = {}

    for name, kwargs in config.loss_fns_config.items():
        scale = kwargs.scale if kwargs.scale is not None else 1.0
        args = kwargs.args if kwargs.args is not None else {}

        if hasattr(torch.nn, name):
            loss_fns[name] = getattr(torch.nn, name)(**args)
        elif hasattr(monai.losses, name):
            loss_fns[name] = getattr(monai.losses, name)(**args)
        elif hasattr(loss, name):
            loss_fns[name] = getattr(loss, name)(**args)
        else:
            raise ValueError(f"Unknown loss function {name}")

        scales[name] = scale

    if loss_fns.keys():
        return loss_fns, scales
    else:
        raise NotImplementedError("No Loss Function defined")


class CustomLoss(torch.nn.Module):
    loss_fns: dict
    scales: dict

    def __init__(self, config: LossConfig):
        super().__init__()
        self.loss_fns, self.scales = generate_loss_fns(config)
        self.log_loss_parts = config.log_loss_parts

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        logger=None,
        phase: str = None,
    ):
        assert torch.equal(
            torch.Tensor(list(prediction.shape)), torch.Tensor(list(target.shape))
        ), "Shape of prediction and target have to match!"

        loss = torch.zeros(1, device=prediction.device)
        for loss_name, loss_fn in self.loss_fns.items():
            loss_tmp = loss_fn(prediction, target)

            if logger is not None and self.log_loss_parts and phase is not None:
                logger(f"{phase}_{loss_name}", loss_tmp)

            loss += self.scales[loss_name] * loss_tmp

        return loss
