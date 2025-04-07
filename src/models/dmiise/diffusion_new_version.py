import torch.nn as nn
from diffusion import DDPM_DPS_Regularized

from utils.hydra_config import LossGuidedDiffusionConfig, OptimizerConfig
from utils.mask_transformer import BaseMaskMapping


class MultiGuidanceDiffusion(DDPM_DPS_Regularized):
    def __init__(
        self,
        model: nn.Module,
        diffusion_config: LossGuidedDiffusionConfig,
        optimizer_config: OptimizerConfig,
        metrics: dict,
        mask_transformer: BaseMaskMapping,
        loss: nn.Module,
    ):
        super().__init__(
            model, diffusion_config, optimizer_config, metrics, mask_transformer, loss
        )

        if "guidance_reps" in diffusion_config.loss_guidance.keys():
            self.guidance_reps = diffusion_config.loss_guidance.guidance_reps
        else:
            self.guidance_reps = 1

        if "guidance_target" in diffusion_config.loss_guidance.keys():
            self.guidance_target = diffusion_config.loss_guidance.guidance_target
        else:
            self.guidance_target = "mask"

    def guided_step(
        self,
        noisy_mask,
        images,
        topo_inputs,
        t,
        batch_idx,
        gamma,
        return_gradients=False,
    ):
        pass
