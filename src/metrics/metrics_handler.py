import monai.metrics
import torch

import utils.metrics_wrapper
from utils.hydra_config import (
    LossGuidedMetricsHandlerConfig,
    MetricsConfig,
    MetricsHandlerConfig,
)


def generate_metrics_fn(config: MetricsConfig, num_classes: int):
    metric_fns = {}

    if config is None or config.metric_fns_config is None:
        return metric_fns

    for name, kwargs in config.metric_fns_config.items():
        if kwargs is None:
            kwargs = {}

        if hasattr(utils.metrics_wrapper, name):
            metric_fns[name] = getattr(utils.metrics_wrapper, name)(**kwargs)
        elif hasattr(torch.nn, name):
            metric_fns[name] = getattr(torch.nn, name)(**kwargs)
        elif hasattr(monai.metrics, name):
            metric_fns[name] = getattr(monai.metrics, name)(**kwargs)
        else:
            raise ValueError(f"Unknown score function {name}")

        # if hasattr(metric_fns[name], "num_classes"):
        #     metric_fns[name].num_classes = num_classes
        #     print(f"num_classes in {name}: {metric_fns[name].num_classes}")
    return metric_fns


def clean_nan_scores_and_avg(scores: torch.Tensor):
    nan_indices = torch.nonzero(~torch.isnan(scores), as_tuple=True)[0]
    return scores[nan_indices].mean().item()


class MetricsInput:
    seg_mask: torch.Tensor
    gt: torch.Tensor
    topo_inputs: dict[str, torch.Tensor]


class MetricsHandler:
    standard_metrics: dict
    topo_metrics: dict

    def __init__(self, config: MetricsHandlerConfig):
        self.standard_metrics = generate_metrics_fn(
            config.standard_metrics, config.num_classes
        )
        self.topo_metrics = generate_metrics_fn(config.topo_metrics, config.num_classes)

    def compute_metrics(self, inputs: MetricsInput, phase: str, logger):
        self.compute_standard_metrics(inputs.seg_mask, inputs.gt, phase, logger)
        self.compute_topo_metrics(inputs.seg_mask, inputs.topo_inputs, phase, logger)

    def compute_standard_metrics(
        self, seg_mask: torch.Tensor, gt: torch.Tensor, phase: str, logger
    ):
        for metric_name, metric_fn in self.standard_metrics.items():
            try:
                score = metric_fn(seg_mask, gt)

                if (
                    hasattr(metric_fn, "logging_names")
                    and metric_fn.logging_names is not None
                ):
                    if len(metric_fn.logging_names) >= 1:
                        assert len(metric_fn.logging_names) == len(score), (
                            "Number of logging names must match number of scores"
                        )
                        for i, name in enumerate(metric_fn.logging_names):
                            logger(
                                f"{phase}_{str(name)}",
                                clean_nan_scores_and_avg(score[i]),
                            )
                    else:
                        raise ValueError("Invalid number of logging names")
                else:
                    assert type(score) is torch.Tensor, "Score must be a tensor"
                    logger(
                        f"{phase}_{str(metric_name)}", clean_nan_scores_and_avg(score)
                    )

            except Exception as e:
                print(f"{metric_name} cannot be computed. Received Error: {str(e)}")

    def compute_topo_metrics(
        self,
        seg_mask: torch.Tensor,
        topo_inputs: dict[str, torch.Tensor],
        phase: str,
        logger,
    ):
        pass


class LossGuidedMetricsHandler(MetricsHandler):
    normal_metrics: dict
    topo_metrics: dict
    guided_step_metrics: dict

    def __init__(self, config: LossGuidedMetricsHandlerConfig):
        self.guided_step_metrics = generate_metrics_fn(
            config.guided_step_metrics, config.num_classes
        )
        super().__init__(config.normal_metrics)
