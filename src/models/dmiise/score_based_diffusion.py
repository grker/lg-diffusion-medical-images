import torch
from diffusers.schedulers import EulerDiscreteScheduler
from torch import nn

from metrics import MetricsHandler
from models.dmiise.diffusion import DDPM
from utils.helper import unpack_batch
from utils.hydra_config import DiffusionConfig, OptimizerConfig
from utils.mask_transformer import BaseMaskMapping


class ScoreBasedDiffusion(DDPM):
    scheduler: EulerDiscreteScheduler
    model: nn.Module
    optimizer_config: OptimizerConfig
    metrics: MetricsHandler
    mask_transformer: BaseMaskMapping
    num_classes: int

    def __init__(
        self,
        model: nn.Module,
        diffusion_config: DiffusionConfig,
        optimizer_config: OptimizerConfig,
        metrics: MetricsHandler,
        mask_transformer: BaseMaskMapping,
        loss: torch.nn.Module,
    ):
        super().__init__(
            model, diffusion_config, optimizer_config, metrics, mask_transformer, loss
        )

        if diffusion_config.num_inference_steps is None:
            self.inference_timesteps = diffusion_config.noise_steps

            print(f"inference steps: {self.inference_timesteps}")
        else:
            self.inference_timesteps = diffusion_config.num_inference_steps

            print(f"inference steps not null: {self.inference_timesteps}")

    def create_scheduler(self, diffusion_config: DiffusionConfig):
        return EulerDiscreteScheduler(
            num_train_timesteps=diffusion_config.noise_steps,
            beta_start=diffusion_config.beta_start,
            beta_end=diffusion_config.beta_end,
            beta_schedule=diffusion_config.scheduler_type,
            prediction_type=diffusion_config.prediction_type,
        )

    def get_sqrt_one_minus_alpha_cumprod(
        self, timesteps: torch.Tensor, dimensions: int
    ):
        shape = [-1] + [1] * (dimensions - 1)
        return torch.sqrt(
            1.0 - self.scheduler.alphas_cumprod.to(timesteps.device)[timesteps]
        ).view(*shape)

    def training_step(self, batch, batch_idx):
        self.model.train()
        self.log("learning rate", self.trainer.optimizers[0].param_groups[0]["lr"])
        images, gt_masks, gt_train_masks = unpack_batch(batch, "train")

        noise = torch.randn_like(gt_train_masks, device=gt_train_masks.device)
        timesteps = torch.full(
            (gt_masks.shape[0],), self.scheduler.config.num_train_timesteps - 1
        ).to(gt_masks.device)

        noisy_batch = self.scheduler.add_noise(gt_train_masks, noise, timesteps)
        model_output = self.model(torch.cat((noisy_batch, images), dim=1), timesteps)

        if self.prediction_type == "sample":
            raise NotImplementedError(
                "Prediction type sampling not implemented for score-based diffusion"
            )

        loss = 0.0
        if self.prediction_type == "epsilon":
            variance = self.get_sqrt_one_minus_alpha_cumprod(
                timesteps, len(noise.shape)
            )
            loss = self.loss_fn(model_output, -noise / variance)

        self.log("train_loss", loss)

        return loss

    def get_model_output(
        self, noisy_mask: torch.Tensor, images: torch.Tensor, timestep: int
    ):
        num_samples = images.shape[0]

        noisy_mask = self.scheduler.scale_model_input(noisy_mask, timestep)

        model_output = self.model(
            torch.cat((noisy_mask, images), dim=1),
            torch.full((num_samples,), timestep, device=images.device),
        )

        return model_output

    def val_test_step(self, batch, batch_idx, phase):
        self.scheduler.set_timesteps(self.inference_timesteps)

        super().val_test_step(batch, batch_idx, phase)
        # self.model.eval()

        # generator = torch.Generator(device=self.device)
        # seed = batch_idx * self.current_epoch
        # generator.manual_seed(seed)

        # images, gt_masks, gt_train_masks, topo_inputs = self.unpack_batch(batch, "test")
        # num_samples = images.shape[0]

        # reps = self.repetitions_test if phase == "test" else self.repetitions

        # with torch.no_grad():
        #     ensemble_mask = []
        #     for rep in range(reps):
        #         noisy_mask = torch.rand_like(gt_train_masks, device=images.device)
        #         for t in tqdm(self.scheduler.timesteps):
        #             model_output = self.model(
        #                 torch.cat((noisy_mask, images), dim=1),
        #                 torch.full((num_samples,), t, device=images.device),
        #             )

        #             noisy_mask = self.scheduler.step(
        #                 model_output=model_output,
        #                 timestep=t,
        #                 sample=noisy_mask,
        #                 generator=generator,
        #             )
