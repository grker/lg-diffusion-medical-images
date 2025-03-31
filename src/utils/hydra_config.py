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


class PersistanceHomologyConfig:
    num_classes: int
    topo_features: dict
    min_persistance: float
    train_switch: bool


class TransformConfig:
    apply_to_mask: bool
    args: dict | None


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
    diffusion_type: str
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
    image_size: tuple[int, int]
    normalize: bool
    mode: str
    mask_transformer: MaskTransformerConfig | None


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
    argmax_mode: str  # either max or min


class UNetConfig(ModelConfig):
    image_size: tuple[int, int]
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


class RegLossConfig(LossConfig):
    mode_for_reference_mask: str


class Metric:
    kwargs: dict


class MetricsConfig:
    metric_fns_config: dict[str, Metric]


class SchedulerConfig:
    name: str
    args: dict


class EMAConfig:
    alpha: float


class OptimizerConfig:
    name: str
    lr: float
    weight_decay: float
    scheduler: SchedulerConfig
    ema: EMAConfig | None


# Configs for the loss guidance
class ScalingFunctionConfig:
    pass


class LikelihoodTempScalingConfig(ScalingFunctionConfig):
    alpha: float


class AnalysisConfig:
    name: str


class PolynomialAnalysisConfig(AnalysisConfig):
    num_bins: int
    poly_degree: int
    minimal_threshold: float


class FixedThresholdAnalysisConfig(AnalysisConfig):
    fixed_threshold: float


class PseudoGTConfig:
    name: str
    topo_features: dict
    num_classes: int
    base_prob: float
    scaling_function: ScalingFunctionConfig
    analysis: AnalysisConfig


class PseudoGTDim0_CompsConfig(PseudoGTConfig):
    scaling_function: LikelihoodTempScalingConfig
    analysis: AnalysisConfig


class GuiderConfig:
    name: str
    num_classes: int
    loss: LossConfig


class BettiGuiderConfig(GuiderConfig):
    topo_features: dict | None
    fixed_betti_numbers: bool


class BettiSegmentationGuiderConfig(BettiGuiderConfig):
    base_prob: float


class BettiPersHomologyGuiderConfig(BettiGuiderConfig):
    pass


class BettiBirthDeathGuiderConfig(BettiPersHomologyGuiderConfig):
    downsampling: bool
    downsampling_factor: tuple[float, float]
    downsampling_mode: str
    modifier: str


class Dim0_CompsScalerGuiderConfig(BettiPersHomologyGuiderConfig):
    with_softmax: bool
    scaling: bool
    analysis: AnalysisConfig
    scaling_function: LikelihoodTempScalingConfig


class RegularizerConfig:
    reg_loss: LossConfig
    weighting: (
        float  # defines the weighting of the regularization loss and the guidance loss
    )
    repeated: bool
    average_ensemble: str
    mode_for_reference_mask: str


class LossGuidanceConfig:
    gamma: float
    starting_step: int
    stop_step: int
    visualize_gradients: bool
    guidance_metrics: MetricsConfig
    guider: GuiderConfig
    regularizer: RegularizerConfig | None


class LossGuidance3StepConfig(LossGuidanceConfig):
    mode: str
    last_step_unguided: bool


class LossGuidedDiffusionConfig(DiffusionConfig):
    loss_guidance: LossGuidanceConfig
    # regularized_loss: RegularizerConfig | None


class MetricsHandlerConfig:
    standard_metrics: MetricsConfig
    topo_metrics: MetricsConfig
    guided_step_metrics: MetricsConfig | None
    starting_step: int


class LossGuidedMetricsHandlerConfig(MetricsHandlerConfig):
    guided_step_metrics: MetricsConfig


# Main Configs (configs passed to main functions)
# ---------------------------------------------


# Segmentation Config (trains and tests a segmentation model):
class SegmentationConfig:
    project_name: str
    dataset: DatasetConfig
    dataloader: DataloaderConfig
    model: ModelConfig
    diffusion: DiffusionConfig
    trainer: TrainerConfig
    metrics: MetricsHandlerConfig
    loss: LossConfig
    optimizer: OptimizerConfig
    train: bool
    model_path: str
    wandb_tags: list[str] | None
    seed: int
    validation_period: int


# Config for the loss guidance inference with a trained model
class LossGuidanceInferenceConfig:
    wandb_username: str
    wandb_project: str
    run_id: str
    seed: int
    wandb_tags: list[str]
    loss_guidance: LossGuidanceConfig
    test_batch_size: int
    repetitions: list[int]
    metrics: MetricsConfig | None = None


# Ensemble Config (tests an ensemble of a trained model):
class EnsembleConfig:
    wandb_username: str
    wandb_project: str
    run_id: str
    repetitions: list[int]
    seed: int
    wandb_tags: list[str]


class ReproduceConfig:
    wandb_username: str
    wandb_project: str
    run_id: str
    seed: int
    wandb_tags: list[str]
    metrics: MetricsConfig
