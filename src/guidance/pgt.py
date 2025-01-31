import torch
import omegaconf

from betti_matching.BettiMatching import CubicalPersistence
from utils.hydra_config import PseudoGTConfig, PseudoGTDim0_CompsConfig


def alpha_smoothing_uniform(x_softmax: torch.Tensor, alpha: int = 0.5):
    """
    Simple smoothing of the softmax output.
    params:
        x_softmax: torch.Tensor, shape (num_classes, height, width)
        alpha: float, between 0 and 1, default 0.5
    returns:
        torch.Tensor, shape (num_classes, height, width)
    """
    uniform_dist_value = 1 / x_softmax.shape[0]
    return x_softmax * alpha + (1 - alpha) * uniform_dist_value


def alpha_smoothing(x_softmax: torch.Tensor, alpha: int = 0.5):
    """
    Smoothing of the softmax output.
    params:
        x_softmax: torch.Tensor, shape (num_classes, height, width)
        alpha: float, between 0 and 1, default 0.5
    returns:
        torch.Tensor, shape (num_classes, height, width)
    """
    num_classes = x_softmax.shape[0]
    max_values, max_indices = torch.max(x_softmax, dim=0)
    max_values_one_hot = alpha * torch.zeros_like(x_softmax).scatter_(
        0, max_indices.unsqueeze(0), max_values.unsqueeze(0)
    )

    return (
        x_softmax
        - max_values_one_hot
        + (torch.sum(max_values_one_hot, dim=0) / num_classes).unsqueeze(0)
    )


def laplace_smoothing(
    x_softmax: torch.Tensor, alpha: float = 1.0, neighborhood: list[torch.Tensor] = None
):
    """
    Laplace smoothing of the softmax output.
    params:
        x_softmax: torch.Tensor, shape (num_classes, height, width)
        neighborhood: list[torch.Tensor], len(neighborhood) = num_classes
    returns:
        torch.Tensor, shape (num_classes, height, width)
    """
    num_classes = x_softmax.shape[0]
    neighborhood_mean = torch.zeros_like(x_softmax)

    class_list = torch.arange(num_classes)
    for class_idx in range(num_classes):
        neighbors = (
            neighborhood[class_idx]
            if neighborhood is not None
            else class_list[class_idx != class_idx]
        )
        neighborhood_mean[class_idx] = torch.mean(x_softmax[neighbors], dim=0)

    return torch.softmax(neighborhood_mean * alpha + (1 - alpha) * x_softmax, dim=0)


def power_scaling(x_softmax: torch.Tensor, alpha: float = 1.0):
    """
    Sharpening of the softmax output.
    params:
        x_softmax: torch.Tensor, shape (num_classes, height, width)
        alpha: float, >= 1
    returns:
        torch.Tensor, shape (num_classes, height, width)
    """
    x_softmax = x_softmax.pow(alpha)
    return x_softmax / torch.sum(x_softmax, dim=0)


def temperature_scaling(x_softmax: torch.Tensor, temperature: float = 0.2):
    """
    Temperature scaling of the softmax output.
    params:
        x_softmax: torch.Tensor, shape (num_classes, height, width)
        temperature: float, < 1
    returns:
        torch.Tensor, shape (num_classes, height, width)
    """
    return torch.softmax(x_softmax / temperature, dim=0)


def likelyhood_temperature_scaling(
    x_softmax: torch.Tensor, likelyhood: torch.Tensor, alpha: float = 1.0
):
    """
    Likelihood temperature scaling of the softmax output.
    params:
        x_softmax: torch.Tensor, shape (batch_size, num_classes, height, width)
        likelyhood: torch.Tensor, shape (batch_size, num_classes, height, width)
        alpha: float, between 0 and 1, default 1.0
    returns:
        torch.Tensor, shape (batch_size, num_classes, height, width)
    """
    # second part does not make sense
    x_softmax = x_softmax / (1 - likelyhood) * alpha + torch.softmax(
        x_softmax, dim=1
    ) * (1 - alpha)
    return x_softmax / torch.sum(x_softmax, dim=1)


class AdjustProbs:
    def __init__(
        self,
        alpha_smoothing: float,
        alpha_sharpening: float,
        smoothing_function: str,
        sharpening_function: str,
        neighborhood: list[torch.Tensor] = None,
    ):
        self.set_smoothing_function(smoothing_function, alpha_smoothing, neighborhood)
        self.set_sharpening_function(sharpening_function, alpha_sharpening)

    def set_smoothing_function(
        self,
        smoothing_function: str,
        alpha_smoothing: float,
        neighborhood: list[torch.Tensor] = None,
    ):
        if smoothing_function == "alpha_smoothing":
            self.smoothing = lambda x: alpha_smoothing(x, alpha_smoothing)
        elif smoothing_function == "alpha_smoothing_uniform":
            self.smoothing = lambda x: alpha_smoothing_uniform(x, alpha_smoothing)
        elif smoothing_function == "laplace_smoothing":
            self.smoothing = lambda x: laplace_smoothing(
                x, alpha_smoothing, neighborhood
            )
        elif smoothing_function == "power_scaling":
            self.smoothing = lambda x: power_scaling(x, alpha_smoothing)
        else:
            raise ValueError(
                f"Invalid smoothing function: {smoothing_function}. Choose from: 'alpha_smoothing', 'alpha_smoothing_uniform', 'laplace_smoothing', 'power_scaling'"
            )

    def set_sharpening_function(
        self, sharpening_function: str, alpha_sharpening: float
    ):
        if sharpening_function == "power_scaling":
            self.sharpening = lambda x: power_scaling(x, alpha_sharpening)
        elif sharpening_function == "temperature_scaling":
            self.sharpening = lambda x: temperature_scaling(x, alpha_sharpening)
        else:
            raise ValueError(
                f"Invalid sharpening function: {sharpening_function}. Choose from: 'power_scaling', 'temperature_scaling'"
            )

    def __call__(self, x_softmax: torch.Tensor, scale_map: torch.Tensor):
        if self.smoothing is None or self.sharpening is None:
            raise ValueError(
                "Smoothing or sharpening function not set. Call set_smoothing_function and set_sharpening_function before calling __call__."
            )
        return self.adjust_probs(x_softmax, scale_map)

    def adjust_probs(self, x_softmax: torch.Tensor, scale_map: torch.Tensor):
        """
        Adjust the probabilities of the softmax output.
        params:
            x_softmax: torch.Tensor, shape (num_classes, height, width)
            scale_map: torch.Tensor, shape (height, width)
        returns:
            torch.Tensor, shape (num_classes, height, width)
        """

        scale_map = scale_map.unsqueeze(0).expand(x_softmax.shape, -1, -1)
        x_smoothing = self.smoothing(x_softmax, scale_map)
        x_sharpening = self.sharpening(x_smoothing, scale_map)

        return scale_map * x_sharpening + (1 - scale_map) * x_smoothing


class PseudoGTGeneratorBase:
    def __init__(self, pgt_config: PseudoGTConfig):
        self.topo_features = self.check_topofeatures(
            pgt_config.topo_features, pgt_config.num_classes
        )
        self.num_classes = pgt_config.num_classes

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

    def pseudo_gt(self, x_softmax: torch.Tensor):
        """
        Generate a pseudo ground truth for the given softmax output.
        params:
            x_softmax: torch.Tensor, shape (batch_size, num_classes, height, width)
        returns:
            torch.Tensor, shape (batch_size, num_classes, height, width)
        """

        raise NotImplementedError(
            "PseudoGTGeneratorBase is an abstract class and cannot be instantiated directly."
        )


class PGTSegGeneratorDim0(PseudoGTGeneratorBase):
    def __init__(self, pgt_config: PseudoGTConfig):
        super().__init__(pgt_config)

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.base_prob = 0.1

    def pseudo_gt(self, x_softmax: torch.Tensor, t: int, batch_idx: int):

        if x_softmax.device != self.device:
            print(f"moving x_softmax to device: {self.device}")
            x_softmax = x_softmax.to(self.device)

        prediction = torch.argmax(x_softmax, dim=1).unsqueeze(1)
        prediction = torch.zeros_like(x_softmax).scatter_(1, prediction, 1)

        binary_component_map = torch.zeros_like(prediction)

        for sample_idx in range(prediction.shape[0]):
            for class_idx in range(self.num_classes):
                binary_component_map[sample_idx, class_idx] = self.component_map(
                    prediction[sample_idx, class_idx], self.topo_features[class_idx][0]
                )

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
            torch.Tensor, shape (height, width)
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
        largest_component = -1
        largest_component_size = 0
        components = []

        while largest_value_in_map > 0:
            component = filtered_component_map == largest_value_in_map
            component_size = torch.sum(component)

            if component_size > largest_component_size:
                largest_component_size = component_size
                largest_component = largest_value_in_map

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


class PseudoGTGeneratorDim0_Comps(PseudoGTGeneratorBase):

    def __init__(self, pgt_config: PseudoGTDim0_CompsConfig):
        self.scaling_function = (
            lambda softmax, likelyhood: likelyhood_temperature_scaling(
                softmax, likelyhood, pgt_config.scaling_function.alpha
            )
        )
        self.analysis = pgt_config.analysis
        self.fixed_threshold = 0.4

        super().__init__(pgt_config)
        print(f"pgt gt object created with config: {pgt_config}")

    def pseudo_gt(self, x_softmax: torch.Tensor, no_scaling: bool = False):
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
                threshold, interval[0], base_prob=0.0, device=device
            )
            component_map = component_map.to(device=likelihood.device)
            likelihood = torch.max(likelihood, component_map)

        return likelihood
