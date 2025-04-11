import copy

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
    guided_fns = {}

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

            if dictionary.guidance:
                guided_fns[name] = copy.deepcopy(metric_fns[name])

        except ValueError as e:
            print(f"Metric {name} cannot be initialized. Received Error: {str(e)}")

    return metric_fns, guided_fns


def generate_topo_metrics_fn(
    config: MetricsConfig, dataset_provided_topo_infos: list[str]
):
    metric_fns = {}
    guided_fns = {}

    print(f"dataset_provided_topo_infos: {dataset_provided_topo_infos}")

    if config is None or config.metric_fns_config is None:
        return metric_fns, guided_fns

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

            if dictionary.guidance:
                guided_fns[name] = copy.deepcopy(metric_fns[name])

        except ValueError as e:
            print(f"Metric {name} cannot be initialized. Received Error: {str(e)}")

    return metric_fns, guided_fns


def clean_nan_scores_and_avg(scores: torch.Tensor):
    nan_indices = torch.nonzero(~torch.isnan(scores), as_tuple=True)[0]
    return scores[nan_indices].mean().item()


def clean_nan_scores_and_sum(scores: torch.Tensor):
    nan_indices = torch.nonzero(~torch.isnan(scores), as_tuple=True)[0]
    return scores[nan_indices].sum().item()


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
        topo_metrics_dict: dict,
        losses: list[str] | None,
        timesteps: int,
        initial_value: float = 0.0,
    ):
        self.metrics_dict = metrics_dict
        self.topo_metrics_dict = topo_metrics_dict
        self.timesteps = [i for i in range(timesteps)]

        self.metric_values_per_timestep = {}
        for name, metric_fn in self.metrics_dict.items():
            if (
                hasattr(metric_fn, "logging_names")
                and metric_fn.logging_names is not None
            ):
                for logging_name in metric_fn.logging_names:
                    self.metric_values_per_timestep[logging_name] = [
                        initial_value
                    ] * timesteps
            else:
                self.metric_values_per_timestep[name] = [initial_value] * timesteps

        self.topo_metric_values_per_timestep = {}
        for name in self.topo_metrics_dict.keys():
            self.topo_metric_values_per_timestep[name] = [initial_value] * timesteps

        self.losses_per_timestep = {}

        if losses is not None:
            for name in losses:
                self.losses_per_timestep[name] = [initial_value] * timesteps

        self.total_samples = [0] * timesteps
        self.total_batches = [0] * timesteps

    def add_losses(self, losses: list[str]):
        for name in losses:
            self.losses_per_timestep[name] = [0.0] * len(self.timesteps)

    def update(self, inputs: MetricsInput, timestep: int):
        prediction = inputs.seg_mask
        gt = inputs.gt

        num_samples = prediction.shape[0]

        for name, metric_fn in self.metrics_dict.items():
            try:
                metric_value = metric_fn(prediction, gt)

                if (
                    hasattr(metric_fn, "logging_names")
                    and metric_fn.logging_names is not None
                ):
                    assert len(metric_fn.logging_names) == len(metric_value), (
                        "Number of logging names must match number of scores"
                    )

                    for i, logging_name in enumerate(metric_fn.logging_names):
                        self.metric_values_per_timestep[logging_name][timestep] += (
                            clean_nan_scores_and_sum(metric_value[i])
                        )
                else:
                    self.metric_values_per_timestep[name][timestep] += (
                        clean_nan_scores_and_sum(metric_value)
                    )
            except Exception as e:
                print(f"{name} cannot be computed. Received Error: {str(e)}")

        for name, metric_fn in self.topo_metrics_dict.items():
            try:
                needed_inputs = metric_fn.get_needed_inputs()

                kwargs = {}

                for input_name in needed_inputs:
                    kwargs[input_name] = inputs.topo_inputs[input_name]

                    metric_value = metric_fn(prediction, **kwargs)
                    self.topo_metric_values_per_timestep[name][timestep] += torch.sum(
                        metric_value
                    ).item()
            except Exception as e:
                print(f"{name} cannot be computed. Received Error: {str(e)}")

        self.total_samples[timestep] += num_samples

    def update_loss(self, losses: dict[str, float], timestep: int):
        for name, loss in losses.items():
            self.losses_per_timestep[name][timestep] += loss

            self.total_batches[timestep] += 1

        print(f"losses per timestep: {self.losses_per_timestep}")

    def compute(self):
        metric_values = {}
        for name, metric_fn in self.metrics_dict.items():
            if (
                hasattr(metric_fn, "logging_names")
                and metric_fn.logging_names is not None
            ):
                for logging_name in metric_fn.logging_names:
                    metric_values[logging_name] = [
                        (
                            self.metric_values_per_timestep[logging_name][timestep]
                            / self.total_samples[timestep]
                            if self.total_samples[timestep] > 0
                            else 0
                        )
                        for timestep in self.timesteps
                    ]
            else:
                metric_values[name] = [
                    (
                        self.metric_values_per_timestep[name][timestep]
                        / self.total_samples[timestep]
                        if self.total_samples[timestep] > 0
                        else 0
                    )
                    for timestep in self.timesteps
                ]

        for name in self.topo_metric_values_per_timestep.keys():
            metric_values[name] = [
                (
                    self.topo_metric_values_per_timestep[name][timestep]
                    / self.total_samples[timestep]
                )
                if self.total_samples[timestep] > 0
                else 0
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
        for name, metric_fn in self.metrics_dict.items():
            if (
                hasattr(metric_fn, "logging_names")
                and metric_fn.logging_names is not None
            ):
                for logging_name in metric_fn.logging_names:
                    data = [
                        [x, y]
                        for (x, y) in zip(self.timesteps, computed_values[logging_name])
                    ]
                    table = wandb.Table(data=data, columns=["timestep", logging_name])
                    wandb.log({f"{logging_name}_metric_per_timestep": table})
            else:
                data = [[x, y] for (x, y) in zip(self.timesteps, computed_values[name])]
                table = wandb.Table(data=data, columns=["timestep", name])
                wandb.log({f"{name}_metric_per_timestep": table})

        for name in self.topo_metrics_dict.keys():
            data = [[x, y] for (x, y) in zip(self.timesteps, computed_values[name])]
            table = wandb.Table(data=data, columns=["timestep", name])
            wandb.log({f"{name}_metric_per_timestep": table})

        for name in self.losses_per_timestep.keys():
            data = [[x, y] for (x, y) in zip(self.timesteps, computed_values[name])]
            table = wandb.Table(data=data, columns=["timestep", name])
            wandb.log({f"{name}_loss_per_timestep": table})


class MetricsHandler:
    standard_metrics: dict
    topo_metrics: dict
    guidance_metrics: GuidanceMetric | None = None

    def __init__(
        self, config: MetricsHandlerConfig, dataset_provided_topo_infos: list[str]
    ):
        self.standard_metrics, self.guided_metrics = generate_metrics_fn(
            config.standard_metrics
        )

        print(f"guided metrics: {self.guided_metrics}")

        self.topo_metrics, self.guided_topo_metrics = generate_topo_metrics_fn(
            config.topo_metrics, dataset_provided_topo_infos
        )

        if (
            len(self.guided_metrics.keys()) > 0
            or len(self.guided_topo_metrics.keys()) > 0
        ):
            self.guidance_metrics = GuidanceMetric(
                self.guided_metrics,
                self.guided_topo_metrics,
                None,
                config.starting_step,
            )
            print("Guidance metrics initialized")

    def update_guidance_metrics(self, inputs: MetricsInput, timestep: int):
        if self.guidance_metrics is not None:
            self.guidance_metrics.update(inputs, timestep)

    def update_loss(self, losses: dict[str, float], timestep: int):
        if self.guidance_metrics is not None:
            self.guidance_metrics.update_loss(losses, timestep)

    def log_guidance_metrics(self):
        if self.guidance_metrics is not None:
            self.guidance_metrics.log_to_wandb()

    def add_losses(self, losses: list[str]):
        if self.guidance_metrics is not None:
            self.guidance_metrics.add_losses(losses)

    def compute_metrics(self, inputs: MetricsInput, phase: str, logger):
        self.compute_standard_metrics(inputs.seg_mask, inputs.gt, phase, logger)
        self.compute_topo_metrics(inputs.seg_mask, inputs.topo_inputs, phase, logger)

    def get_index_of_sub_metric(
        self, metric_name: str, metric_fn: torch.nn.Module, sub_name: str
    ):
        print(f"logging_names: {metric_fn.logging_names}")
        if (
            sub_name is not None
            and hasattr(metric_fn, "logging_names")
            and metric_fn.logging_names is not None
        ):
            try:
                return metric_fn.logging_names.index(sub_name)
            except ValueError:
                raise ValueError(
                    f"Metric {metric_name} does not have a logging name {sub_name}"
                )
        else:
            return None

    def find_bad_samples(
        self,
        inputs: MetricsInput,
        metric_name: str,
        sub_name: str = None,
        threshold: float = 0.0,
    ):
        score = None
        print(f"topo_metrics: {self.topo_metrics}")
        print(f"standard_metrics: {self.standard_metrics}")

        if metric_name in self.standard_metrics.keys():
            metric_fn = self.standard_metrics[metric_name]
            index = self.get_index_of_sub_metric(metric_name, metric_fn, sub_name)

            if index is None:
                score = metric_fn(inputs.seg_mask, inputs.gt)
            else:
                score = metric_fn(inputs.seg_mask, inputs.gt)[index]

        if metric_name in self.topo_metrics.keys():
            metric_fn = self.topo_metrics[metric_name]
            index = self.get_index_of_sub_metric(metric_name, metric_fn, sub_name)

            needed_inputs = metric_fn.get_needed_inputs()
            kwargs = {}
            for input_name in needed_inputs:
                kwargs[input_name] = inputs.topo_inputs[input_name]

            if index is None:
                score = metric_fn(inputs.seg_mask, **kwargs)
            else:
                score = metric_fn(inputs.seg_mask, **kwargs)[index]

            print(f"score: {score}")

        if score is None:
            raise ValueError(
                f"Metric {metric_name} that should be used to find bad samples not found"
            )

        indices = torch.nonzero(score > threshold, as_tuple=False)

        return indices, score

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
