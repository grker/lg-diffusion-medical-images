import torch
import omegaconf

from utils.loss import CustomLoss
from utils.hydra_config import LossGuidanceConfig


class LossGuider:
    """
    Base Loss Guider class. This class is a helper class for the loss guidance.
    It defines and computes the loss used in the guidance step. It also contains any logic that is needed to modify the model output and to create a pseudo ground truth which can than be used to compute the loss.
    """

    def __init__(self, loss_guidance_config: LossGuidanceConfig):
        self.loss_guidance_config = loss_guidance_config

    def get_starting_step(self):
        return self.loss_guidance_config.starting_step

    def get_gamma(self):
        return self.loss_guidance_config.gamma

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

    def __init__(self, loss_guidance_config: LossGuidanceConfig):
        super().__init__(loss_guidance_config)

        pgt_config = loss_guidance_config.pseudo_gt_generator

        self.topo_features = self.check_topofeatures(
            pgt_config.topo_features, pgt_config.num_classes
        )

        if self.loss_guidance_config.loss:
            self.loss_fn = CustomLoss(self.loss_guidance_config.loss)
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

        print(f"topo items: {topo_features.items()}")

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
                and type(topo_feature[0]) == int
                and topo_feature[0] >= 0
            ):
                if (
                    1 in topo_feature.keys()
                    and type(topo_feature[1]) == int
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

        return topo_features
