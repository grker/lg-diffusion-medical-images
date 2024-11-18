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
    print(f"allocated memory before metric computation: {torch.cuda.memory_allocated()}")
    for metric_name, metric_fn in metric_fns.items():
        try: 
            score = metric_fn(logits, gt)

            if hasattr(metric_fn, "logging_names") and metric_fn.logging_names is not None:
                if len(metric_fn.logging_names) > 1:
                    assert len(metric_fn.logging_names) == len(score), "Number of logging names must match number of scores"
                    for i, name in enumerate(metric_fn.logging_names):
                        logger(f"{phase}_{str(name)}", clean_nan_scores_and_avg(score[i]))
                elif len(metric_fn.logging_names) == 1:
                    assert type(score) == torch.Tensor, "Score must be a tensor"
                    logger(f"{phase}_{str(metric_fn.logging_names[0])}", clean_nan_scores_and_avg(score))
                else:
                    raise ValueError("Invalid number of logging names")
            else:
                assert type(score) == torch.Tensor, "Score must be a tensor"
                logger(f"{phase}_{str(metric_name)}", clean_nan_scores_and_avg(score))
                
        except Exception as e:
            print(f"{metric_fn} cannot be computed. Received Error: {str(e)}")

    print(f"allocated memory after metric computation: {torch.cuda.memory_allocated()}")
    return scores


def clean_nan_scores_and_avg(scores: torch.Tensor):
    nan_indices = torch.nonzero(~torch.isnan(scores), as_tuple=True)[0]
    return scores[nan_indices].mean().item()