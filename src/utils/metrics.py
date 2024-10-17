import torch
import monai.metrics

from utils.hydra_config import MetricsConfig


def generate_metrics_fn(config: MetricsConfig):
    metric_fns = {}

    if config is None or config.metric_fns_config is None:
        return metric_fns

    for name, kwargs in config.metric_fns_config.items():
        if kwargs is None:
            kwargs = {}
        
        print(f"building metric with arguments: {kwargs}")
        if hasattr(torch.nn, name):
            metric_fns[name] = getattr(torch.nn, name)(**kwargs)
        elif hasattr(monai.metrics, name):
            metric_fns[name] = getattr(monai.metrics, name)(**kwargs)
        else:
            raise ValueError(f"Unknown score function {name}")
        
    return metric_fns


def compute_and_log_metrics(metric_fns: dict, pred: torch.Tensor, gt: torch.Tensor, phase: str, logger) -> dict:
    scores = {}
    
    for metric_name, metric_fn in metric_fns.items():
        try: 
            score = metric_fn(pred, gt).mean().item()
            logger(f"{phase}_{str(metric_name)}", score)
        except Exception as e:
            print(f"{metric_fn} cannot be computed. Received Error: {str(e)}")

    return scores
