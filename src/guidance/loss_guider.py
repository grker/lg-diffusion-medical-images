import torch
import omegaconf
from torch.nn import CrossEntropyLoss

from utils.hydra_config import LossGuidanceConfig
from utils.loss import CustomLoss


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
        self, x_softmax: torch.Tensor, t: int = None, batch_idx: int = None
    ):
        raise NotImplementedError("Guidance loss not implemented")


class LossGuiderBetti(LossGuider):
    """
    Superclass for all loss guiders optimizing the betti number metric/error.
    """

    def __init__(self, loss_guidance_config: LossGuidanceConfig):
        print(f"loss_guidance_config: {loss_guidance_config}")
        pgt_config = loss_guidance_config.pseudo_gt_generator

        print(f"type of pgt_config: {type(pgt_config.topo_features)}")
        print(f"pgt_config: {pgt_config.topo_features}")

        self.topo_features = self.check_topofeatures(
            pgt_config.topo_features, pgt_config.num_classes
        )

        print(f"topo_features: {self.topo_features}")

        # if pgt_config.topo_features and isinstance(pgt_config.topo_features, dict):
        #     self.topo_features = self.check_topofeatures(
        #         pgt_config.topo_features, pgt_config.num_classes
        #     )
        # else:
        #     raise ValueError(
        #         "The pseudo generator config is not valid. Either the attribute 'topo_features' is not defined or it is not a dictionary.",
        #     )

        super().__init__(loss_guidance_config)

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


class LossGuiderSegmentationComponents(LossGuiderBetti):
    """
    This loss guider is a simple loss guider that tries to correct the amount of components in the guidance process.
    """

    def __init__(self, loss_guidance_config: LossGuidanceConfig):
        super().__init__(loss_guidance_config)

        self.base_prob = self.loss_guidance_config.pseudo_gt_generator.base_prob
        self.num_classes = self.loss_guidance_config.pseudo_gt_generator.num_classes

        if self.loss_guidance_config.loss:
            self.loss_fn = CustomLoss(self.loss_guidance_config.loss)
        else:
            self.loss_fn = CrossEntropyLoss()

    def pseudo_gt(self, x_softmax: torch.Tensor, t: int, batch_idx: int):

        prediction = torch.argmax(x_softmax, dim=1).unsqueeze(1)
        prediction = torch.zeros_like(x_softmax).scatter_(1, prediction, 1)

        binary_component_map = torch.zeros_like(prediction)

        for sample_idx in range(prediction.shape[0]):
            for class_idx in range(self.num_classes):
                component_map = self.component_map(
                    prediction[sample_idx, class_idx], self.topo_features[class_idx][0]
                )
                binary_component_map[sample_idx, class_idx] = component_map.squeeze(0)

        # from utils.visualize import visualize_component_map

        # visualize_component_map(
        #     binary_component_map[0].unsqueeze(0),
        #     f"binary_component_map_timestep_{t}",
        #     batch_idx=batch_idx,
        #     merged=False,
        # )

        likelihood = binary_component_map * x_softmax
        likelihood = torch.where(likelihood > 0, likelihood, self.base_prob)

        return likelihood

    def component_map(self, prediction: torch.Tensor, num_components: int):
        """
        Generate a component map for the given prediction. Pixels belonging to the same component are assigned the same value. Background is 0.
        params:
            prediction: torch.Tensor, shape (height, width)
            num_components: int
        returns:
            torch.Tensor, shape (1, height, width)
        """
        width, height = prediction.shape[0], prediction.shape[1]
        prediction = prediction.unsqueeze(0)
        component_map = (
            torch.arange(width * height, device=prediction.device)
            .reshape(height, width)
            .unsqueeze(0)
            * prediction
        )

        for i in range(2 * max(width, height)):
            component_map = (
                torch.max_pool2d(component_map, kernel_size=3, stride=1, padding=1)
                * prediction
            )

        filtered_component_map = component_map

        largest_value_in_map = torch.max(component_map)
        components = []

        while largest_value_in_map > 0:
            component = filtered_component_map == largest_value_in_map
            component_size = torch.sum(component)

            components.append((largest_value_in_map, component_size))

            filtered_component_map = torch.where(
                component_map == largest_value_in_map,
                0,
                filtered_component_map,
            )

            largest_value_in_map = torch.max(filtered_component_map)

        components.sort(key=lambda x: x[1], reverse=True)
        num_components = min(num_components, len(components))
        numbers = [components[i][0] for i in range(num_components)]

        binary_map = torch.zeros_like(component_map, device=component_map.device)
        for i in range(num_components):
            binary_map = torch.add(binary_map, component_map == numbers[i])

        return binary_map

    def guidance_loss(self, model_output: torch.Tensor, t: int, batch_idx: int):
        x_softmax = torch.softmax(torch.clamp(model_output, -1, 1), dim=1).detach()
        pseudo_gt = self.pseudo_gt(x_softmax, t, batch_idx)

        loss = self.loss_fn(model_output, pseudo_gt)
        return loss

    def get_loss_gradient(
        self,
        model_output: torch.Tensor,
        x_softmax: torch.Tensor,
        noisy_mask: torch.Tensor,
        t: int,
        batch_idx: int,
    ):

        print(f"has requires grad: {noisy_mask.requires_grad}")
        print(f"x_softmax requires grad: {x_softmax.requires_grad}")
        pseudo_gt = self.pseudo_gt(x_softmax, t, batch_idx)
        loss = self.loss_fn(model_output, x_softmax)
        loss.backward()
        return noisy_mask.grad
