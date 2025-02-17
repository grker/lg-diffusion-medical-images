import torch

from utils.hydra_config import LossGuidanceConfig
from betti_matching.BettiMatching import CubicalPersistence

from ..loss_guider_base import LossGuiderBetti
from ..utils import likelyhood_temperature_scaling


class PersHomologyBettiGuidance(LossGuiderBetti):
    """
    Superclass for all loss guiders optimizing the betti number metric/error using persistence homology analysis of the output to create a pseudo ground truth.
    """

    def __init__(self, loss_guidance_config: LossGuidanceConfig):
        super().__init__(loss_guidance_config)


class PersHomologyBettiGuidanceDim0_Comps(PersHomologyBettiGuidance):

    def __init__(self, loss_guidance_config: LossGuidanceConfig):
        super().__init__(loss_guidance_config)

        pgt_config = loss_guidance_config.pseudo_gt_generator

        self.scaling_function = (
            lambda softmax, likelyhood: likelyhood_temperature_scaling(
                softmax, likelyhood, pgt_config.scaling_function.alpha
            )
        )
        self.analysis = pgt_config.analysis
        self.fixed_threshold = 0.4

        super().__init__(pgt_config)
        print(f"pgt gt object created with config: {pgt_config}")

    def pseudo_gt(
        self, x_softmax: torch.Tensor, t: int, batch_idx: int, no_scaling: bool = False
    ):
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
        if no_scaling:
            return likelihood
        else:
            scaled_version = self.scaling_function(x_softmax, likelihood)
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
        intervals_and_threshold = cp.threshold_analysis_dim0_components(
            num_components=num_components,
            num_bins=self.analysis.num_bins,
            degree=self.analysis.poly_degree,
            minimal_threshold=self.analysis.minimal_threshold,
        )

        for interval, threshold in intervals_and_threshold:
            print(
                f"using interval: {interval}, threshold: {threshold} for copmonent map"
            )
            print(f"using fixed threshold of: {self.fixed_threshold}")
            threshold = self.fixed_threshold
            component_map = cp.component_map(
                threshold, interval[0], base_prob=0.1, device=device
            )
            component_map = component_map.to(device=likelihood.device)
            likelihood = torch.max(likelihood, component_map)

        return likelihood
