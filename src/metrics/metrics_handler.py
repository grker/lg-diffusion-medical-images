import monai.metrics
import torch

import metrics
import wandb
from utils.hydra_config import (
    MetricsConfig,
    MetricsHandlerConfig,
)


def generate_metrics_fn(config: MetricsConfig):
    metric_fns = {}

    if config is None or config.metric_fns_config is None:
        return metric_fns

    for name, dictionary in config.metric_fns_config.items():
        try:
            kwargs = {}

            if dictionary is not None:
                kwargs = dictionary.kwargs

            if hasattr(metrics, name):
                metric_fns[name] = getattr(metrics, name)(**kwargs)
            elif hasattr(torch.nn, name):
                metric_fns[name] = getattr(torch.nn, name)(**kwargs)
            elif hasattr(monai.metrics, name):
                metric_fns[name] = getattr(monai.metrics, name)(**kwargs)
            else:
                raise ValueError(f"Unknown score function {name}")

        except ValueError as e:
            print(f"Metric {name} cannot be initialized. Received Error: {str(e)}")

    return metric_fns


def generate_topo_metrics_fn(
    config: MetricsConfig, dataset_provided_topo_infos: list[str]
):
    metric_fns = {}

    if config is None or config.metric_fns_config is None:
        return metric_fns

    for name, dictionary in config.metric_fns_config.items():
        try:
            kwargs = {}

            if dictionary is not None:
                kwargs = dictionary.kwargs

            if hasattr(metrics, name):
                needed_inputs = getattr(metrics, name).needed_inputs

                for input_name in needed_inputs:
                    if input_name not in dataset_provided_topo_infos:
                        raise ValueError(
                            f"The metric {name} needs the topological input {input_name} which is not provided in the dataset"
                        )

                metric_fns[name] = getattr(metrics, name)(**kwargs)
            else:
                raise ValueError(f"Unknown topological metric function {name}")

        except ValueError as e:
            print(f"Metric {name} cannot be initialized. Received Error: {str(e)}")

    return metric_fns


def clean_nan_scores_and_avg(scores: torch.Tensor):
    nan_indices = torch.nonzero(~torch.isnan(scores), as_tuple=True)[0]
    return scores[nan_indices].mean().item()


class MetricsInput:
    seg_mask: torch.Tensor
    gt: torch.Tensor
    topo_inputs: dict[str, torch.Tensor]

    def __init__(
        self,
        seg_mask: torch.Tensor,
        gt: torch.Tensor,
        topo_inputs: dict[str, torch.Tensor],
    ):
        self.seg_mask = seg_mask
        self.gt = gt
        self.topo_inputs = topo_inputs


class MetricsHandler:
    standard_metrics: dict
    topo_metrics: dict

    def __init__(
        self, config: MetricsHandlerConfig, dataset_provided_topo_infos: list[str]
    ):
        self.standard_metrics = generate_metrics_fn(config.standard_metrics)
        self.topo_metrics = generate_topo_metrics_fn(
            config.topo_metrics, dataset_provided_topo_infos
        )

        self.guided_step_metrics = generate_metrics_fn(config.guided_step_metrics)
        self.guidance_metrics = GuidanceMetric(
            self.guided_step_metrics,
            None,
            config.starting_step,
        )

    def update_guidance_metrics(
        self, prediction: torch.Tensor, gt: torch.Tensor, timestep: int
    ):
        self.guidance_metrics.update(prediction, gt, timestep)

    def log_guidance_metrics(self):
        self.guidance_metrics.log_to_wandb()

    def add_losses(self, losses: list[str]):
        if self.guided_step_metrics is not None:
            self.guidance_metrics.add_losses(losses)

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
        for metric_name, metric_fn in self.topo_metrics.items():
            try:
                needed_inputs = metric_fn.get_needed_inputs()

                kwargs = {}

                for input_name in needed_inputs:
                    kwargs[input_name] = topo_inputs[input_name]

                score = metric_fn(seg_mask, **kwargs)

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


class GuidanceMetric:
    metrics_dict: dict
    losses: list[str]
    timesteps: list[int]
    metric_values_per_timestep: dict[str, list[float]]
    losses_per_timestep: dict[str, list[float]]
    total_samples: list[int]
    total_batches: list[int]

    def __init__(
        self,
        metrics_dict: dict,
        losses: list[str] | None,
        timesteps: int,
        initial_value: float = 0.0,
    ):
        self.metrics_dict = metrics_dict
        self.timesteps = [i for i in range(timesteps)]

        self.metric_values_per_timestep = {}
        for name in self.metrics_dict.keys():
            self.metric_values_per_timestep[name] = [initial_value] * timesteps

        self.losses_per_timestep = {}

        if losses is not None:
            for name in losses:
                self.losses_per_timestep[name] = [initial_value] * timesteps

        self.total_samples = [0] * timesteps
        self.total_batches = [0] * timesteps

    def add_losses(self, losses: list[str]):
        for name in losses:
            self.losses_per_timestep[name] = [0.0] * len(self.timesteps)

    def update(self, prediction: torch.Tensor, gt: torch.Tensor, timestep: int):
        num_samples = prediction.shape[0]

        for name, metric_fn in self.metrics_dict.items():
            metric_value = metric_fn(prediction, gt)
            self.metric_values_per_timestep[name][timestep] += torch.sum(
                metric_value
            ).item()

        self.total_samples[timestep] += num_samples

    def update_loss(self, losses: dict[str, float], timestep: int):
        for name, loss in losses.items():
            self.losses_per_timestep[name][timestep] += loss

            self.total_batches[timestep] += 1

    def compute(self):
        metric_values = {}
        for name in self.metrics_dict.keys():
            metric_values[name] = [
                (
                    self.metric_values_per_timestep[name][timestep]
                    / self.total_samples[timestep]
                    if self.total_samples[timestep] > 0
                    else 0
                )
                for timestep in self.timesteps
            ]

        for name in self.losses_per_timestep.keys():
            metric_values[name] = [
                (
                    self.losses_per_timestep[name][timestep]
                    / self.total_batches[timestep]
                    if self.total_batches[timestep] > 0
                    else 0
                )
                for timestep in self.timesteps
            ]

        return metric_values

    def log_to_wandb(self):
        computed_values = self.compute()
        for name in self.metrics_dict.keys():
            data = [[x, y] for (x, y) in zip(self.timesteps, computed_values[name])]
            table = wandb.Table(data=data, columns=["timestep", name])

            wandb.log({f"{name}_metric_per_timestep": table})

        for name in self.losses_per_timestep.keys():
            data = [[x, y] for (x, y) in zip(self.timesteps, computed_values[name])]
            table = wandb.Table(data=data, columns=["timestep", name])
            wandb.log({f"{name}_loss_per_timestep": table})
