import torch

from betti_matching.BettiMatching import CubicalPersistence
from utils.hydra_config import (
    BettiBirthDeathGuiderConfig,
    BettiPersHomologyGuiderConfig,
    Dim0_CompsScalerGuiderConfig,
)

from ..loss_guider_base import LossGuiderBetti
from ..utils import likelyhood_temperature_scaling, max_min_normalization


class PersHomologyBettiGuidance(LossGuiderBetti):
    """
    Superclass for all loss guiders optimizing the betti number metric/error using persistence homology analysis of the output to create a pseudo ground truth.
    """

    def __init__(self, guider_config: BettiPersHomologyGuiderConfig):
        super().__init__(guider_config)


class PersHomologyBettiGuidanceDim0_Comps(PersHomologyBettiGuidance):
    # TODO: refactor it

    def __init__(self, guider_config: Dim0_CompsScalerGuiderConfig):
        super().__init__(guider_config)
        self.scaling_function = (
            lambda softmax, likelyhood: likelyhood_temperature_scaling(
                softmax, likelyhood, guider_config.scaling_function.alpha
            )
        )

        self.analysis_config = guider_config.analysis
        self.num_classes = guider_config.num_classes
        self.with_softmax = guider_config.with_softmax
        self.scaling = guider_config.scaling

        super().__init__(guider_config)

    def pseudo_gt(self, x_softmax: torch.Tensor, t: int, batch_idx: int):
        """
        Generate a pseudo ground truth for the given softmax output.
        params:
            x_softmax: torch.Tensor, shape (batch_size, num_classes, height, width)
        returns:
            torch.Tensor, shape (batch_size, num_classes, height, width)
        """

        device = x_softmax.device
        likelihood = torch.zeros_like(x_softmax, device=device)

        for sample_idx in range(x_softmax.shape[0]):
            for class_idx in range(self.num_classes):
                topo_feature_0 = self.topo_features[class_idx][0]
                likelihood[sample_idx, class_idx] = self.likelihood_map(
                    x_softmax[sample_idx, class_idx], topo_feature_0
                )
        if not self.scaling:
            return likelihood
        else:
            return self.scaling_function(x_softmax, likelihood)

    def likelihood_map(self, class_probs: torch.Tensor, num_components: int):
        """
        Generate a likelihood map for the given softmax output.
        params:
            x_softmax: torch.Tensor, shape (height, width)
        returns:
            torch.Tensor, shape (height, width)
        """

        device = class_probs.device
        likelihood = torch.zeros_like(class_probs, device=device)
        cp = CubicalPersistence(
            class_probs.cpu(),
            relative=False,
            reduced=False,
            filtration="superlevel",
            construction="V",
            birth_UF=True,
        )

        # define the threshold
        if self.analysis_config.name == "fixed_threshold":
            intervals_and_threshold = cp.fixed_threshold_analysis(
                num_components, self.analysis_config.fixed_threshold
            )
        elif self.analysis_config.name == "polynomial":
            intervals_and_threshold = cp.threshold_analysis_dim0_components(
                num_components=num_components,
                num_bins=self.analysis_config.num_bins,
                degree=self.analysis_config.poly_degree,
                minimal_threshold=self.analysis_config.minimal_threshold,
            )
        else:
            raise ValueError(f"Unknown analysis type: {self.analysis_config.name}")

        for interval, threshold in intervals_and_threshold:
            print(
                f"using interval: {interval}, threshold: {threshold} for copmonent map"
            )
            component_map = cp.component_map(
                threshold, interval[0], base_prob=0.1, device=device
            )
            component_map = component_map.to(device=likelihood.device)
            likelihood = torch.max(likelihood, component_map)

        return likelihood

    def guidance_loss(self, model_output: torch.Tensor, t: int, batch_idx: int):
        model_output = max_min_normalization(model_output)

        if self.with_softmax:
            x_softmax = torch.softmax(model_output, dim=1)
        else:
            x_softmax = model_output

        pseudo_gt = self.pseudo_gt(x_softmax, t, batch_idx)
        loss = self.loss_fn(x_softmax, pseudo_gt)

        return loss


class Birth_Death_Guider(PersHomologyBettiGuidance):
    def __init__(self, guider_config: BettiBirthDeathGuiderConfig):
        super().__init__(guider_config)

        self.num_classes = guider_config.num_classes
        self.downsampling = guider_config.downsampling
        self.downsampling_factor = tuple(guider_config.downsampling_factor)
        self.downsampling_mode = guider_config.downsampling_mode

        self.modifier = guider_config.modifier

    def pseudo_gt(self, x_softmax: torch.Tensor, t: int, batch_idx: int):
        intervals_0 = []
        intervals_1 = []

        for sample_idx in range(x_softmax.shape[0]):
            sample_intervals_0 = []
            sample_intervals_1 = []
            for class_idx in range(self.num_classes):
                interval_0, interval_1 = self.get_intervals(
                    x_softmax[sample_idx, class_idx],
                )
                sample_intervals_0.append(interval_0)
                sample_intervals_1.append(interval_1)

            intervals_0.append(sample_intervals_0)
            intervals_1.append(sample_intervals_1)
        return intervals_0, intervals_1

    def get_intervals(self, class_probs: torch.Tensor):
        device = class_probs.device
        cp = CubicalPersistence(
            class_probs.cpu(),
            relative=False,
            reduced=False,
            filtration="superlevel",
            construction="V",
            birth_UF=False,
        )

        return cp.birth_death_pixels(0, start=0).to(
            device=device
        ), cp.birth_death_pixels(1, start=0).to(device=device)

    def guidance_loss(
        self, model_output: torch.Tensor, t: int = None, batch_idx: int = None
    ):
        print(f"*** Timestep: {t} ***")
        if self.downsampling:
            model_output = self.downsample_model_output(
                model_output, self.downsampling_factor, self.downsampling_mode
            )

        model_output = max_min_normalization(model_output)
        x_softmax = torch.softmax(model_output, dim=1)

        intervals_0, intervals_1 = self.pseudo_gt(
            x_softmax.detach(),
            t,
            batch_idx,
        )

        loss = self.loss_fn(
            x_softmax,
            intervals_comp_0=intervals_0,
            intervals_comp_1=intervals_1,
        )
        return loss

    def downsample_model_output(
        self,
        model_output: torch.Tensor,
        scale_factor: tuple[float, float],
        mode: str = "bilinear",
    ):
        from torch.nn.functional import interpolate

        return interpolate(
            model_output,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=False,
        )


class Birth_Death_Guider_Dim0(Birth_Death_Guider):
    def __init__(self, guider_config: BettiBirthDeathGuiderConfig):
        super().__init__(guider_config)

    def pseudo_gt(self, x_softmax: torch.Tensor, t: int, batch_idx: int):
        intervals = []

        for sample_idx in range(x_softmax.shape[0]):
            sample_intervals = []
            for class_idx in range(self.num_classes):
                sample_intervals.append(
                    self.get_intervals(
                        x_softmax[sample_idx, class_idx],
                        self.topo_features[class_idx][0],
                    )
                )
            intervals.append(sample_intervals)
        return intervals, None

    def get_intervals(self, class_probs: torch.Tensor, num_comps: int):
        device = class_probs.device
        cp = CubicalPersistence(
            class_probs.cpu(),
            relative=False,
            reduced=False,
            filtration="superlevel",
            construction="V",
            birth_UF=False,
        )

        return cp.birth_death_pixels(0, start=0).to(device=device)


class BirthDeathGuider(PersHomologyBettiGuidance):
    possible_losses = ["BirthDeathLoss", "BirthDeathIntervalLoss"]

    def __init__(self, guider_config: BettiBirthDeathGuiderConfig):
        super().__init__(guider_config)

        self.num_classes = guider_config.num_classes
        self.downsampling = guider_config.downsampling
        self.downsampling_factor = tuple(guider_config.downsampling_factor)
        self.downsampling_mode = guider_config.downsampling_mode

        self.modifier = guider_config.modifier

    def pseudo_gt(self, x_softmax: torch.Tensor, t: int, batch_idx: int):
        intervals_0 = []
        intervals_1 = []

        for sample_idx in range(x_softmax.shape[0]):
            sample_intervals_0 = []
            sample_intervals_1 = []
            for class_idx in range(x_softmax.shape[1]):
                interval_0, interval_1 = self.get_intervals(
                    x_softmax[sample_idx, class_idx],
                )
                sample_intervals_0.append(interval_0)
                sample_intervals_1.append(interval_1)

            intervals_0.append(sample_intervals_0)
            intervals_1.append(sample_intervals_1)
        return intervals_0, intervals_1

    def get_intervals(self, class_probs: torch.Tensor):
        device = class_probs.device
        cp = CubicalPersistence(
            class_probs.cpu(),
            relative=False,
            reduced=False,
            filtration="superlevel",
            construction="V",
            birth_UF=False,
        )

        return cp.birth_death_pixels(0, start=0).to(
            device=device
        ), cp.birth_death_pixels(1, start=0).to(device=device)

    def prepare_model_output(self, model_output: torch.Tensor):
        if self.num_classes == 2 and model_output.shape[1] == 1:  # binary case
            model_output = torch.sigmoid(model_output)
        else:
            model_output = max_min_normalization(model_output)
            model_output = torch.softmax(model_output, dim=1)
        return model_output

    def guidance_loss(
        self, model_output: torch.Tensor, t: int, batch_idx: int, **kwargs: dict
    ):
        betti_0_batched, betti_1_batched = self.batched_betti(
            model_output.shape[0], kwargs
        )

        assert betti_0_batched.shape[:2] == model_output.shape[:2]
        assert betti_1_batched.shape[:2] == model_output.shape[:2]

        if self.downsampling:
            model_output = self.downsample_model_output(
                model_output, self.downsampling_factor, self.downsampling_mode
            )

        x_softmax = self.prepare_model_output(model_output)

        intervals_0, intervals_1 = self.pseudo_gt(
            x_softmax.detach(),
            t,
            batch_idx,
        )

        loss = self.loss_fn(
            x_softmax,
            intervals_comp_0=intervals_0,
            intervals_comp_1=intervals_1,
            betti_0=betti_0_batched,
            betti_1=betti_1_batched,
        )
        return loss

    def downsample_model_output(
        self,
        model_output: torch.Tensor,
        scale_factor: tuple[float, float],
        mode: str = "bilinear",
    ):
        from torch.nn.functional import interpolate

        return interpolate(
            model_output,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=False,
        )

class TopoLossGuider(PersHomologyBettiGuidance):
    pass
