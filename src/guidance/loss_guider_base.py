import omegaconf
import torch

from loss import single_loss_fn
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

        self.topo_features = self.check_topofeatures(
            guider_config.topo_features, guider_config.num_classes
        )

        if guider_config.loss:
            for loss_name in guider_config.loss.loss_fns_config.keys():
                guider_config.loss.loss_fns_config[loss_name]["args"][
                    "betti_numbers"
                ] = self.topo_features

            self.loss_fn = single_loss_fn(guider_config.loss)
        else:
            from torch.nn import CrossEntropyLoss

            self.loss_fn = CrossEntropyLoss()

    def check_topofeatures(self, topo_features: dict, num_classes: int):
        """
        Check if the topo_features are valid.
        """

        if len(topo_features) != num_classes:
            raise ValueError(
                f"Expected {num_classes} topo_features definitions, but got {len(topo_features)}"
            )

        idx_list = [i for i in range(num_classes)]

        for class_idx, topo_feature in topo_features.items():
            if not isinstance(topo_feature, omegaconf.dictconfig.DictConfig):
                raise ValueError(
                    f"Topo feature for class {class_idx} is not a dictionary"
                )
            if class_idx in idx_list:
                idx_list.remove(class_idx)
            else:
                raise ValueError(
                    f"Topo feature for class {class_idx} is not in the idx list of the classes"
                )

            if (
                0 in topo_feature.keys()
                and type(topo_feature[0]) is int
                and topo_feature[0] >= 0
            ):
                if (
                    1 in topo_feature.keys()
                    and type(topo_feature[1]) is int
                    and topo_feature[1] >= 0
                ):
                    continue
                else:
                    raise ValueError(
                        f"Topo feature for class {class_idx} does not contain homology dimension for class 1"
                    )
            else:
                raise ValueError(
                    f"Topo feature for class {class_idx} does not contain homology dimension for class 0"
                )

        return dict(topo_features)
