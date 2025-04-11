import torch

from utils.helper import get_fixed_betti_numbers
from utils.hydra_config import BettiSegmentationGuiderConfig

from ..loss_guider_base import LossGuiderBetti


class SegBasedBettiGuidance(LossGuiderBetti):
    """
    Superclass for all loss guiders optimizing the betti number metric/error using the actual segmentation of the output to create a pseudo ground truth.
    """

    def __init__(self, guider_config: BettiSegmentationGuiderConfig):
        super().__init__(guider_config)

        self.base_prob = guider_config.base_prob

    def component_map(self, prediction: torch.Tensor, num_components: int):
        """
        Generate a component map for the given prediction. Pixels belonging to the same component are assigned the same value. Background is 0. All used torch operations can be executed on the GPU, no transfer to the CPU is needed.
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


class LossGuiderSegmentationComponents(SegBasedBettiGuidance):
    """
    This loss guider is a simple loss guider that tries to correct the amount of components in the guidance process.
    All used torch operations can be executed on the GPU, no transfer to the CPU is needed.
    """

    def __init__(self, guider_config: BettiSegmentationGuiderConfig):
        super().__init__(guider_config)

    def get_segmentation(self, x_softmax: torch.Tensor):
        prediction = torch.argmax(x_softmax, dim=1).unsqueeze(1)
        prediction = torch.zeros_like(x_softmax).scatter_(1, prediction, 1)
        return prediction

    def pseudo_gt(
        self, x_softmax: torch.Tensor, t: int, batch_idx: int, betti_0: torch.Tensor
    ):
        prediction = self.get_segmentation(x_softmax)

        binary_component_map = torch.zeros_like(prediction)

        for sample_idx in range(prediction.shape[0]):
            for class_idx in range(prediction.shape[1]):
                component_map = self.component_map(
                    prediction[sample_idx, class_idx], betti_0[sample_idx][class_idx]
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

    def guidance_loss(
        self, model_output: torch.Tensor, t: int, batch_idx: int, **kwargs: dict
    ):
        betti_0, _ = self.batched_betti(
            model_output.shape[0], only_betti_0=True, **kwargs
        )

        assert betti_0.shape[:2] == model_output.shape[:2]

        x_softmax = torch.softmax(torch.clamp(model_output, -1, 1), dim=1)
        pseudo_gt = self.pseudo_gt(x_softmax.detach(), t, batch_idx, betti_0)

        loss = self.loss_fn(model_output, pseudo_gt)
        print(f"Loss: {loss}")
        return loss


class LossGuiderSegmentationComponentsDigits(LossGuiderSegmentationComponents):
    possible_losses = ["BCELoss"]

    def __init__(self, guider_config: BettiSegmentationGuiderConfig):
        super().__init__(guider_config)

    def get_segmentation(self, x_softmax: torch.Tensor):
        return (x_softmax > 0.5).to(torch.float32)

    def guidance_loss(
        self, model_output: torch.Tensor, t: int, batch_idx: int, **kwargs: dict
    ):
        betti_0, _ = self.batched_betti(
            model_output.shape[0], only_betti_0=True, **kwargs
        )

        assert betti_0.shape[:2] == model_output.shape[:2]

        x_softmax = torch.sigmoid(model_output)
        pseudo_gt = self.pseudo_gt(x_softmax.detach(), t, batch_idx, betti_0).to(
            device=x_softmax.device
        )

        loss = self.loss_fn(x_softmax, pseudo_gt)
        print(f"Loss: {loss}")
        return loss


class LossGuiderSegmentationCycles(SegBasedBettiGuidance):
    """
    This loss guider is a simple loss guider that tries to correct the amount of cycles and the amount of components in the guidance process. All used torch operations can be executed on the GPU, no transfer to the CPU is needed.
    """

    def __init__(self, guider_config: BettiSegmentationGuiderConfig):
        super().__init__(guider_config)

        self.betti_0, self.betti_1 = get_fixed_betti_numbers(
            self.topo_features, self.num_classes
        )

    def get_segmentation(self, x_softmax: torch.Tensor):
        prediction = torch.argmax(x_softmax, dim=1).unsqueeze(1)
        prediction = torch.zeros_like(x_softmax).scatter_(1, prediction, 1)
        return prediction

    def pseudo_gt(
        self,
        x_softmax: torch.Tensor,
        t: int,
        batch_idx: int,
        betti_0: torch.Tensor,
        betti_1: torch.Tensor,
    ):
        prediction = self.get_segmentation(x_softmax)

        binary_component_map = torch.zeros_like(prediction)

        for sample_idx in range(prediction.shape[0]):
            for class_idx in range(prediction.shape[1]):
                holes_map, holes = self.holes_map(
                    prediction[sample_idx, class_idx],
                )

                comp_map = self.component_map(
                    prediction[sample_idx, class_idx], betti_0[sample_idx, class_idx]
                )

                binary_component_map[sample_idx, class_idx] = (
                    self.combined_comp_holes_map(
                        holes_map,
                        comp_map,
                        holes,
                        betti_1[sample_idx, class_idx],
                    )
                )

        # from utils.visualize import visualize_component_map

        # if t % 10 == 0 or t < 4:
        #     visualize_component_map(
        #         binary_component_map[0].unsqueeze(0),
        #         f"binary_component_map_timestep_{t}",
        #         batch_idx=batch_idx,
        #         merged=False,
        #     )

        likelihood = binary_component_map * x_softmax
        likelihood = torch.where(likelihood > 0, likelihood, self.base_prob)
        return likelihood

    def holes_map(self, class_prediction: torch.Tensor):
        """
        Generate a map containing the holes for the given prediction. Pixels belonging to the same holes are assigned the same value. Background is 0.
        params:
            class_prediction: torch.Tensor, shape (height, width), binary map --> 1: pixel belonging to class, 0: pixel does belong to another class
            num_components: int
        returns:
            torch.Tensor, shape (height, width)
        """

        class_prediction = class_prediction.unsqueeze(0)
        reversed_map = class_prediction != 1
        width, height = class_prediction.shape[1], class_prediction.shape[2]

        holes_map = (
            torch.arange(width * height, device=class_prediction.device)
            .reshape(height, width)
            .unsqueeze(0)
            * reversed_map
        ).to(dtype=torch.float32)

        for i in range(2 * max(width, height)):
            holes_map = (
                torch.max_pool2d(holes_map, kernel_size=3, stride=1, padding=1)
                * reversed_map
            )

        border = torch.cat(
            (
                holes_map[:, 0, :],
                holes_map[:, -1, :],
                holes_map[:, :, 0],
                holes_map[:, :, -1],
            ),
            dim=1,
        )

        largest_border_value = torch.max(border)
        holes = []

        # eliminate the incorrect holes. Incorrect holes are the components that have border pixels belonging to it.
        while largest_border_value > 0:
            holes_map = torch.where(holes_map == largest_border_value, 0, holes_map)
            border = torch.where(border == largest_border_value, 0, border)
            largest_border_value = torch.max(border)

        # iterate over the correct holes
        largest_value = torch.max(holes_map)
        holes_map_copy = holes_map.clone()

        while largest_value > 0:
            size = torch.sum(holes_map_copy == largest_value)
            holes.append((largest_value, size))
            holes_map_copy = torch.where(
                holes_map_copy == largest_value, 0, holes_map_copy
            )
            largest_value = torch.max(holes_map_copy)

        holes.sort(key=lambda x: x[1], reverse=True)

        return holes_map, holes

    def combined_comp_holes_map(
        self,
        holes_map: torch.Tensor,
        component_map: torch.Tensor,
        holes: list[tuple[int, int]],
        num_holes: int,
    ):
        component_map_copy = component_map.clone()
        max_value_hole = torch.max(holes_map)
        component_map = component_map * (max_value_hole + 1)
        component_map = component_map + holes_map

        binary_map = component_map > 0

        width, height = component_map.shape[1], component_map.shape[2]

        for i in range(2 * max(width, height)):
            component_map = (
                torch.max_pool2d(component_map, kernel_size=3, stride=1, padding=1)
                * binary_map
            )

        holes_within_component = (component_map == (max_value_hole + 1)) * holes_map

        good_holes = 0
        idx = 0

        while good_holes < num_holes and idx < len(holes):
            hole_value = holes[idx][0]

            if hole_value in holes_within_component:
                good_holes += 1

                holes_within_component = torch.where(
                    holes_within_component == hole_value, 0, holes_within_component
                )

            idx += 1

        holes_within_component = holes_within_component > 0

        return component_map_copy + holes_within_component

    def guidance_loss(
        self, model_output: torch.Tensor, t: int, batch_idx: int, **kwargs: dict
    ):
        print(f"timestep: {t}")
        print(f"model_output max: {model_output.max()}")
        print(f"model_output min: {model_output.min()}")
        # x_softmax = torch.softmax(torch.clamp(model_output, -1, 1), dim=1).detach()
        # x_softmax = max_min_normalization(model_output).detach()

        betti_0_batched, betti_1_batched = self.batched_betti(
            model_output.shape[0], **kwargs
        )

        assert betti_0_batched.shape[:2] == model_output.shape[:2]
        assert betti_1_batched.shape[:2] == model_output.shape[:2]

        pseudo_gt = self.pseudo_gt(
            model_output, t, batch_idx, betti_0_batched, betti_1_batched
        )
        loss = self.loss_fn(model_output, pseudo_gt)
        return loss


class LossGuiderSegmenationCyclesDigits(LossGuiderSegmentationCycles):
    possible_losses = ["BCELoss"]

    def __init__(self, guider_config: BettiSegmentationGuiderConfig):
        super().__init__(guider_config)

    def get_segmentation(self, x_softmax: torch.Tensor):
        return (x_softmax > 0.5).to(torch.float32)

    def guidance_loss(
        self, model_output: torch.Tensor, t: int, batch_idx: int, **kwargs: dict
    ):
        betti_0, betti_1 = self.batched_betti(model_output.shape[0], **kwargs)

        assert betti_0.shape[:2] == model_output.shape[:2]
        assert betti_1.shape[:2] == model_output.shape[:2]

        x_softmax = torch.sigmoid(model_output)
        pseudo_gt = self.pseudo_gt(
            x_softmax.detach(), t, batch_idx, betti_0, betti_1
        ).to(device=x_softmax.device)

        loss = self.loss_fn(x_softmax, pseudo_gt)
        return loss
