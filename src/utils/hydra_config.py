from dataclasses import dataclass
from typing import Union


class ModelConfig:
    pass

class ResNetConfig(ModelConfig):
    layers: list[int]
    starting_channels: int

# class UNetConfig(ModelConfig):
#     sample_layers: list[int]
#     bottleneck_layers: list[int]
#     starting_channels: int
#     time_dim: int

class DiffusionConfig:
    noise_steps: int
    beta_start: float
    beta_end: float
    scheduler_type: str
    device: str
    var_learned: bool
    repetitions: int
    threshold: float

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
    multiclass: bool
    switch: int


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


class MetricsConfig:
    metric_fns_config: dict

class OptimizerConfig:
    lr: float
    weight_decay: float


class SegmentationConfig:
    project_name: str
    dataset: DatasetConfig
    dataloader: DataloaderConfig
    model: ModelConfig
    diffusion: DiffusionConfig
    trainer: TrainerConfig
    metrics: MetricsConfig
    optimizer: OptimizerConfig
    train: bool
    model_path: str
    wandb_tags: list[str]
    seed: int

