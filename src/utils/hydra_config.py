from dataclasses import dataclass
from typing import Union


class ModelConfig:
    name: str
    pass

class ResNetConfig(ModelConfig):
    layers: list[int]
    starting_channels: int

# class UNetConfig(ModelConfig):
#     sample_layers: list[int]
#     bottleneck_layers: list[int]
#     starting_channels: int
#     time_dim: int

class TransformConfig:
    apply_to_mask: bool
    args: dict

class PreprocessorConfig:
    transform_config: dict[str, TransformConfig]

class MaskTransformerConfig(ModelConfig):
    mask_type: str
    dataset_mapping: dict
    train_mapping: dict
    prediction_type: str
    train_switch: bool
    threshold: float
    ensemble_mode: str
    multiclass: bool


class DiffusionConfig:
    noise_steps: int
    beta_start: float
    beta_end: float
    scheduler_type: str
    device: str
    var_learned: bool
    repetitions: int
    repetitions_test: int
    threshold: float
    prediction_type: str
    num_inference_steps: int
    clip_range: int
    

class LDSegConfig:
    resnet: ResNetConfig
    diffusion: DiffusionConfig
    has_decoder: bool

class DatasetConfig:
    name: str
    data_path: str
    image_size: tuple[int,int]
    normalize: bool
    mode: str
    mask_transformer: MaskTransformerConfig


class DataloaderConfig:
    batch_size: int
    val_batch_size: int
    shuffle: bool
    train_ratio: float
    validation_ratio: float

class TrainerConfig:
    max_epochs: int
    enable_progress_bar: bool
    accelerator: str
    argmax_metric: str
    argmax_mode: str # either max or min

class UNetConfig(ModelConfig):
    image_size: tuple[int,int]
    in_channels: int
    model_channels: int
    out_channels: int
    num_res_blocks: int
    attention_resolutions: list[int]
    dropout: float
    channel_mult: list[int]
    conv_resample: bool
    dims: int
    num_classes: int
    use_checkpoint: bool
    use_fp16: bool
    num_heads: int
    num_head_channels: int
    num_heads_upsample: int
    use_scale_shift_norm: bool
    resblock_updown: bool
    use_new_attention_order: bool

class BasicUNetConfig(ModelConfig):
    spatial_dims: int
    in_channels: int
    out_channels: int
    features: list[int]
    dropout: float
    emb_channels: int
    time_start: int

class Loss:
    scale: float
    args: dict

class LossConfig:
    loss_fns_config: dict[str, Loss]
    log_loss_parts: bool

class MetricsConfig:
    metric_fns_config: dict

class SchedulerConfig:
    name: str
    args: dict

class OptimizerConfig:
    lr: float
    weight_decay: float
    scheduler: SchedulerConfig

class SegmentationConfig:
    project_name: str
    dataset: DatasetConfig
    dataloader: DataloaderConfig
    model: ModelConfig
    diffusion: DiffusionConfig
    trainer: TrainerConfig
    metrics: MetricsConfig
    loss: LossConfig
    optimizer: OptimizerConfig
    train: bool
    model_path: str
    wandb_tags: list[str]
    seed: int 
    validation_period: int

class TestConfig:
    wandb_username: str
    wandb_project: str
    run_id: str
    repetitions: list[int]
    seed: int
    wandb_tags: list[str]


