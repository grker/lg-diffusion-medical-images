import torch
import monai.metrics
import numpy as np

import utils.metrics_wrapper
from utils.hydra_config import MetricsConfig




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
        
        if hasattr(metric_fns[name], 'num_classes'):
            metric_fns[name].num_classes = num_classes        
    return metric_fns


def compute_and_log_metrics(metric_fns: dict, logits: torch.Tensor,  gt: torch.Tensor, phase: str, logger) -> dict:
    scores = {}

    for metric_name, metric_fn in metric_fns.items():
        try: 
            score = metric_fn(logits, gt)
            print(f"{metric_name} score: {score}")
            nan_indices = torch.nonzero(~torch.isnan(score), as_tuple=True)[0]
            score = score[nan_indices]

            score = score.mean().item()
            logger(f"{phase}_{str(metric_name)}", score)
        except Exception as e:
            print(f"{metric_fn} cannot be computed. Received Error: {str(e)}")

    return scores
