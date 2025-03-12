import random

# import time
import pytorch_lightning as pl
import torch
import torch.nn as nn
from diffusers.schedulers import DDPMScheduler
from torch.utils.data import Dataset
from tqdm import tqdm

import wandb
from guidance.loss_guider import LossGuider
from metrics import MetricsHandler, MetricsInput
from utils.helper import unpack_batch
from utils.hydra_config import (
    DiffusionConfig,
    GuiderConfig,
    LossGuidance3StepConfig,
    LossGuidedDiffusionConfig,
    OptimizerConfig,
)
from utils.mask_transformer import BaseMaskMapping
from utils.visualize import (
    create_wandb_image,
    gif_over_timesteps,
    visualize_gradients,
    visualize_mean_variance,
    visualize_segmentation,
)


def scheduler_factory(
    scheduler_type: str, beta_start: float, beta_end: float, noise_steps: int
):
    if scheduler_type == "linear":
        return torch.linspace(beta_start, beta_end, noise_steps)
    else:
        raise NotImplementedError(f"Scheduler {scheduler_type} not implemented")


class DDPM(pl.LightningModule):
    scheduler: DDPMScheduler
    model: nn.Module
    optimizer_config: OptimizerConfig
    metrics: MetricsHandler
    mask_transformer: BaseMaskMapping
    num_classes: int  # always includes the background

    def __init__(
        self,
        model: nn.Module,
        diffusion_config: DiffusionConfig,
        optimizer_config: OptimizerConfig,
        metrics: MetricsHandler,
        mask_transformer: BaseMaskMapping,
        loss: torch.nn.Module,
    ):
        super().__init__()

        print("Creating DDPM model")

        self.create_gif_over_timesteps = False
        self.scheduler = DDPMScheduler(
            num_train_timesteps=diffusion_config.noise_steps,
            beta_start=diffusion_config.beta_start,
            beta_end=diffusion_config.beta_end,
            beta_schedule=diffusion_config.scheduler_type,
            prediction_type=diffusion_config.prediction_type,
            clip_sample_range=diffusion_config.clip_range,
        )

        if (
            diffusion_config.num_inference_steps is None
            or diffusion_config.num_inference_steps > diffusion_config.noise_steps
        ):
            self.scheduler.set_timesteps(diffusion_config.noise_steps)
        else:
            self.scheduler.set_timesteps(diffusion_config.num_inference_steps)

        self.model = model
        self.optimizer_config = optimizer_config
        self.metric_handler = metrics
        self.repetitions = diffusion_config.repetitions
        self.threshold = diffusion_config.threshold
        self.prediction_type = diffusion_config.prediction_type
        self.mask_transformer = mask_transformer
        self.repetitions_test = diffusion_config.repetitions_test
        self.num_classes = mask_transformer.get_num_classes()

        self.loss_fn = loss

    def noise_tester(self, dataset: Dataset, batch_size: int, device="cuda"):
        samples = len(dataset)
        batch_start_idx = 0

        flatt_values = torch.empty(0).to(device)

        while batch_start_idx < samples:
            end_idx = min(batch_start_idx + batch_size, samples)
            _, masks = dataset[batch_start_idx:end_idx]
            batch = masks.to(device)

            noise = torch.randn_like(batch, device=batch.device)
            timesteps = torch.full(
                (noise.shape[0],), self.scheduler.config.num_train_timesteps - 1
            ).to(batch.device)
            noisy_batch = self.scheduler.add_noise(batch, noise, timesteps)

            flatt_values = torch.cat((flatt_values, torch.flatten(noisy_batch)), dim=0)
            batch_start_idx = batch_start_idx + batch_size

        values = flatt_values.shape[0]
        print(f"number of pixels in dataset: {values}")
        print(f"mean: {torch.mean(flatt_values)}")
        print(f"std: {torch.std(flatt_values)}")

    def unpack_batch(self, batch):
        images, gt_masks, gt_train_masks = None, None, None
        if len(batch) == 2:
            images, gt_train_masks = batch
            gt_masks = gt_train_masks
        elif len(batch) == 3:
            images, gt_masks, gt_train_masks = batch
        else:
            raise ValueError(
                f"Batch has {len(batch)} to unpack. A batch can only have 2 or 3 values to unpack."
            )

        assert (
            images.shape[0] == gt_masks.shape[0]
            and gt_masks.shape[0] == gt_train_masks.shape[0]
        ), (
            "Assertion Error: images, gt_masks and gt_train_masks need to have the same number of samples"
        )

        return images, gt_masks, gt_train_masks

    def training_step(self, batch, batch_idx):
        self.model.train()
        self.log("learning rate", self.trainer.optimizers[0].param_groups[0]["lr"])
        images, gt_masks, gt_train_masks = unpack_batch(batch, "train")
        num_samples = images.shape[0]

        noise = torch.randn_like(gt_train_masks, device=gt_train_masks.device)
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (num_samples,),
            device=gt_train_masks.device,
            dtype=torch.int64,
        )
        noisy_image = self.scheduler.add_noise(gt_train_masks, noise, timesteps)

        prediction = self.model(torch.cat((noisy_image, images), dim=1), timesteps)

        loss = 0.0
        if self.prediction_type == "epsilon":
            loss = self.loss_fn(prediction, noise)
        elif self.prediction_type == "sample":
            loss = self.loss_fn(prediction, gt_train_masks)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        return self.val_test_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.val_test_step(batch, batch_idx, "test")

    def val_test_step(self, batch, batch_idx, phase):
        self.model.eval()

        images, gt_masks, gt_train_masks, topo_inputs = unpack_batch(batch, "test")
        num_samples = images.shape[0]
        reps = self.repetitions_test if phase == "test" else self.repetitions

        if self.create_gif_over_timesteps:
            prediction_for_gif = torch.zeros(
                (
                    self.scheduler.config.num_train_timesteps,
                    reps,
                    *gt_train_masks.shape[1:],
                )
            )
            random_idx = random.randint(0, num_samples - 1)

        with torch.no_grad():
            ensemble_mask = []
            for rep in range(reps):
                noisy_mask = torch.rand_like(gt_train_masks, device=images.device)
                for t in tqdm(self.scheduler.timesteps):
                    model_output = self.model(
                        torch.cat((noisy_mask, images), dim=1),
                        torch.full((num_samples,), t, device=images.device),
                    )

                    noisy_mask = self.scheduler.step(
                        model_output=model_output, timestep=t, sample=noisy_mask
                    ).prev_sample

                    if self.create_gif_over_timesteps:
                        prediction_for_gif[t, rep] = noisy_mask[random_idx]
                ensemble_mask.append(noisy_mask.detach().cpu())

            if self.create_gif_over_timesteps:
                gif_over_timesteps(
                    torch.flip(prediction_for_gif, dims=[0]),
                    gt_train_masks[random_idx],
                    f"./results/gif_over_timesteps_output_{random_idx}.gif",
                    mode="probs",
                    time=500,
                )

            logits = self.mask_transformer.get_logits(torch.stack(ensemble_mask, dim=0))
            seg_mask = self.mask_transformer.get_segmentation(logits)
            gt_masks = gt_masks.to(device=seg_mask.device)

            # Metrics computation
            metrics_input = MetricsInput(seg_mask, gt_masks, topo_inputs)
            self.metric_handler.compute_metrics(metrics_input, phase, self.log)

            # Visualization
            index = random.randint(0, num_samples - 1)
            visualize_segmentation(
                images,
                gt_masks,
                seg_mask,
                None,
                phase,
                self.mask_transformer.gt_mapping_for_visualization(),
                batch_idx,
                self.num_classes,
                [index],
            )
            if reps > 1:
                visualize_mean_variance(
                    ensemble_mask, phase, batch_idx, index_list=[index]
                )

        return 0

    def test_variance(
        self,
        image: torch.Tensor,
        gt_mask: torch.Tensor,
        gt_train_mask: torch.Tensor,
        reps: int,
        batch_idx: int,
    ):
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        if len(gt_mask.shape) == 3:
            gt_mask = gt_mask.unsqueeze(0)
        if len(gt_train_mask.shape) == 3:
            gt_train_mask = gt_train_mask.unsqueeze(0)

        num_samples = image.shape[0]
        ensemble_shape = (reps, *gt_train_mask.shape)
        ensemble_mask = torch.zeros(ensemble_shape, device=image.device)

        for r in range(reps):
            noisy_mask = torch.rand_like(gt_train_mask, device=image.device)

            for t in tqdm(self.scheduler.timesteps):
                model_output = self.model(
                    torch.cat((noisy_mask, image), dim=1),
                    torch.full((num_samples,), t, device=image.device),
                )
                noisy_mask = self.scheduler.step(
                    model_output=model_output, timestep=t, sample=noisy_mask
                ).prev_sample

            ensemble_mask[r] = noisy_mask

        mean_mask = torch.mean(ensemble_mask, dim=0)
        std_mask = torch.std(ensemble_mask, dim=0)

        min_per_sample_mean = torch.min(
            torch.flatten(mean_mask, start_dim=1), dim=1
        ).values
        max_per_sample_mean = torch.max(
            torch.flatten(mean_mask, start_dim=1), dim=1
        ).values

        min_per_sample_std = torch.min(
            torch.flatten(std_mask, start_dim=1), dim=1
        ).values
        max_per_sample_std = torch.max(
            torch.flatten(std_mask, start_dim=1), dim=1
        ).values

        mean_mask = (mean_mask - min_per_sample_mean) / (
            max_per_sample_mean - min_per_sample_mean
        )
        std_mask = (std_mask - min_per_sample_std) / (
            max_per_sample_std - min_per_sample_std
        )

        for idx in range(num_samples):
            wandb.log(
                {
                    "mean_mask": create_wandb_image(
                        mean_mask[idx],
                        caption=f"Testing VarianceBIdx_{batch_idx}_Idx_{idx}",
                    )
                }
            )
            wandb.log(
                {
                    "std_mask": create_wandb_image(
                        std_mask[idx],
                        caption=f"Testing VarianceBIdx_{batch_idx}_Idx_{idx}",
                    )
                }
            )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.optimizer_config.lr
        )

        if hasattr(torch.optim.lr_scheduler, self.optimizer_config.scheduler.name):
            args = (
                self.optimizer_config.scheduler.args
                if self.optimizer_config.scheduler.args is not None
                else {}
            )
            scheduler = getattr(
                torch.optim.lr_scheduler, self.optimizer_config.scheduler.name
            )(optimizer, **args)
        else:
            raise ValueError(
                f"Scheduler {self.optimizer_config.scheduler.name} not found!"
            )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class DDPM_DPS(DDPM):
    loss_guider: LossGuider
    starting_step: int
    gamma: float
    metrics_fn_dict: dict
    topology_loss: nn.Module

    def __init__(
        self,
        model: nn.Module,
        diffusion_config: LossGuidedDiffusionConfig,
        optimizer_config: OptimizerConfig,
        metrics: dict,
        mask_transformer: BaseMaskMapping,
        loss: torch.nn.Module,
    ):
        super().__init__(
            model, diffusion_config, optimizer_config, metrics, mask_transformer, loss
        )

        self.loss_guider = self.initialize_guider(diffusion_config.loss_guidance.guider)
        self.starting_step = diffusion_config.loss_guidance.starting_step
        self.gamma = diffusion_config.loss_guidance.gamma

        losses = [self.loss_guider.loss_name]
        self.metric_handler.add_losses(losses)

        self.visualize_gradients = diffusion_config.loss_guidance.visualize_gradients

    def initialize_guider(self, guider_config: GuiderConfig):
        import guidance

        if hasattr(guidance, guider_config.name):
            return getattr(guidance, guider_config.name)(guider_config)
        else:
            raise ValueError(f"PseudoGTGenerator {guider_config.name} not found!")

    @torch.inference_mode(False)
    def val_test_step(self, batch, batch_idx, phase):
        self.model.eval()

        images, gt_masks, gt_train_masks, topo_inputs = unpack_batch(batch, "test")
        num_samples = images.shape[0]
        reps = self.repetitions_test if phase == "test" else self.repetitions

        ensemble_mask = []
        for rep in range(reps):
            noisy_mask = torch.rand_like(gt_train_masks, device=images.device)

            # unguided diffusion steps
            for t in tqdm(self.scheduler.timesteps[: -self.starting_step]):
                noisy_mask = self.unguided_step(noisy_mask, images, t)

            # activate the gradient for the model for the guided steps
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    param.requires_grad_(True)

            noisy_mask = noisy_mask.requires_grad_(True)

            # guided diffusion steps
            for t in tqdm(self.scheduler.timesteps[-self.starting_step : -1]):
                noisy_mask = self.guided_step(noisy_mask, images, t, batch_idx)

                self.metric_handler.update(
                    self.mask_transformer.get_segmentation(
                        self.mask_transformer.get_logits(noisy_mask.unsqueeze(0))
                    ),
                    gt_masks,
                    t,
                )

            # deactivate the gradient for the model
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.requires_grad_(False)

            # last step is always unguided
            noisy_mask = noisy_mask.requires_grad_(False)
            noisy_mask = self.unguided_step(noisy_mask, images, 0)
            self.guidance_metrics.update(
                self.mask_transformer.get_segmentation(
                    self.mask_transformer.get_logits(noisy_mask.unsqueeze(0))
                ),
                gt_masks,
                0,
            )

            self.loss_guider.guidance_loss(
                noisy_mask,
                0,
                batch_idx,
            )

            ensemble_mask.append(noisy_mask.detach().cpu())
            self.metric_handler.log_guidance_metrics()

        # Prediction Segmentation mask
        logits = self.mask_transformer.get_logits(
            torch.stack(
                ensemble_mask,
                dim=0,
            )
        )
        seg_mask = self.mask_transformer.get_segmentation(logits)
        gt_masks = gt_masks.to(device=seg_mask.device)

        # Metrics computation
        metrics_input = MetricsInput(seg_mask, gt_masks, topo_inputs)
        self.metric_handler.compute_metrics(metrics_input, phase, self.log)

        # Visualization
        index = random.randint(0, num_samples - 1)
        visualize_segmentation(
            images,
            gt_masks,
            seg_mask,
            None,
            phase,
            self.mask_transformer.gt_mapping_for_visualization(),
            batch_idx,
            self.num_classes,
            [index],
        )
        if reps > 1:
            visualize_mean_variance(ensemble_mask, phase, batch_idx, index_list=[index])

        # log_cuda_memory(f"after val_test_step")
        return 0

    @torch.inference_mode(False)
    def guided_step(
        self,
        noisy_mask: torch.Tensor,
        images: torch.Tensor,
        t: int,
        batch_idx: int,
        gamma: float | None = None,
        return_gradients: bool = False,
    ):
        num_samples = images.shape[0]
        noisy_mask = noisy_mask.requires_grad_(True)
        model_output = self.model(
            torch.cat((noisy_mask, images), dim=1),
            torch.full((num_samples,), t, device=images.device),
        )

        loss = self.loss_guider.guidance_loss(
            model_output,
            t,
            batch_idx,
        )
        print(f"loss at timestep {t}: {loss}")

        self.metric_handler.update_loss(
            {self.loss_guider.loss_name: loss.item()},
            t,
        )

        loss.backward()

        # for name, param in self.model.named_parameters():
        #     if param.grad is not None:
        #         grad_sum = param.grad.sum().item()  # Sum of gradients
        #         # grad_size = param.grad.shape  # Size of gradient tensor
        #         print(f"Parameter: {name}, Gradient Sum: {grad_sum}, Mean: {param.grad.mean().item()}")
        #     else:
        #         print(f"Parameter: {name} has no gradient and enabled grad: {param.requires_grad}")

        with torch.no_grad():
            noisy_mask_grads = noisy_mask.grad

            new_noisy_mask = (
                self.scheduler.step(
                    model_output=model_output, timestep=t, sample=noisy_mask
                ).prev_sample
                - (self.gamma if gamma is None else gamma) * noisy_mask_grads
            )

            random_idx = int(num_samples / 2)
            if self.visualize_gradients:
                visualize_gradients(
                    noisy_mask_grads[random_idx],
                    t,
                    batch_idx,
                    random_idx,
                    torch.clamp(model_output[random_idx], -1, 1),
                    commit=(t == 1),
                )

        if return_gradients:
            return new_noisy_mask, noisy_mask_grads

        model_output = model_output.detach().cpu()
        noisy_mask = noisy_mask.detach().cpu()

        return new_noisy_mask

    @torch.inference_mode(False)
    def unguided_step(
        self,
        noisy_mask: torch.Tensor,
        images: torch.Tensor,
        t: int,
    ):
        with torch.no_grad():
            num_samples = images.shape[0]
            model_output = self.model(
                torch.cat((noisy_mask, images), dim=1),
                torch.full((num_samples,), t, device=images.device),
            )

            # if t == 0:
            #     return model_output

            noisy_mask = self.scheduler.step(
                model_output=model_output,
                timestep=t,
                sample=noisy_mask,
            ).prev_sample

        model_output = model_output.detach().cpu()
        return noisy_mask

    def normalize_noisy_mask(
        self,
        noisy_mask: torch.Tensor,
        class_wise: bool = True,
    ):
        if class_wise:
            dim = (2, 3)
        else:
            dim = (1, 2, 3)

        min_vals = torch.amin(noisy_mask, dim=dim, keepdim=True)
        max_vals = torch.amax(noisy_mask, dim=dim, keepdim=True)

        return 2 * (noisy_mask - min_vals) / (max_vals - min_vals + 1e-8) - 1


class DDPM_DPS_3Steps(DDPM_DPS):
    def __init__(
        self,
        model: nn.Module,
        diffusion_config: LossGuidedDiffusionConfig,
        optimizer_config: OptimizerConfig,
        metrics: dict,
        mask_transformer: BaseMaskMapping,
        loss: torch.nn.Module,
    ):
        super().__init__(
            model, diffusion_config, optimizer_config, metrics, mask_transformer, loss
        )

        self.set_mode(diffusion_config.loss_guidance)

    def set_mode(self, loss_guidance_config: LossGuidance3StepConfig):
        self.mode = loss_guidance_config.mode
        self.last_step_unguided = loss_guidance_config.last_step_unguided

    @torch.inference_mode(False)
    def val_test_step(self, batch, batch_idx, phase):
        self.model.eval()

        images, gt_masks, gt_train_masks, topo_inputs = unpack_batch(batch, "test")
        num_samples = images.shape[0]
        reps = self.repetitions_test if phase == "test" else self.repetitions

        ensemble_mask = []
        for rep in range(reps):
            noisy_mask = torch.rand_like(gt_train_masks, device=images.device)

            # unguided diffusion steps
            for t in tqdm(self.scheduler.timesteps[: -self.starting_step]):
                noisy_mask = self.unguided_step(noisy_mask, images, t)

            # activate the gradient for the model for the guided steps
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    param.requires_grad_(True)

            noisy_mask = noisy_mask.requires_grad_(True)

            if self.mode == "only_guided":
                noisy_mask = self.only_guidance(
                    noisy_mask,
                    t - 1,
                    batch_idx,
                    gt_masks,
                    last_step_unguided=self.last_step_unguided,
                )
            elif self.mode == "dps_guidance":
                noisy_mask = self.dps_guidance(
                    noisy_mask,
                    images,
                    t - 1,
                    batch_idx,
                    gt_masks,
                    last_step_unguided=self.last_step_unguided,
                )
            else:
                raise ValueError(f"Mode {self.mode} not supported!")

            # deactivate the gradient for the model
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.requires_grad_(False)

            if self.last_step_unguided:
                noisy_mask = noisy_mask.requires_grad_(False)
                noisy_mask = self.unguided_step(noisy_mask, images, 0)
                self.guidance_metrics.update(
                    self.mask_transformer.get_segmentation(
                        self.mask_transformer.get_logits(noisy_mask.unsqueeze(0))
                    ),
                    gt_masks,
                    0,
                )

            # final loss:
            loss = self.loss_guider.guidance_loss(
                noisy_mask,
                0,
                batch_idx,
            )
            self.metric_handler.update_loss(
                {self.loss_guider.loss_name: loss.item()},
                0,
            )
            print(f"final loss: {loss}")

            ensemble_mask.append(noisy_mask.detach().cpu())
            self.metric_handler.log_guidance_metrics()

        # Prediction Segmentation mask
        logits = self.mask_transformer.get_logits(
            torch.stack(
                ensemble_mask,
                dim=0,
            )
        )
        seg_mask = self.mask_transformer.get_segmentation(logits)
        gt_masks = gt_masks.to(device=seg_mask.device)

        # Metrics computation
        metrics_input = MetricsInput(seg_mask, gt_masks, topo_inputs)
        self.metric_handler.compute_metrics(metrics_input, phase, self.log)

        # Visualization
        index = random.randint(0, num_samples - 1)
        visualize_segmentation(
            images,
            gt_masks,
            seg_mask,
            None,
            phase,
            self.mask_transformer.gt_mapping_for_visualization(),
            batch_idx,
            self.num_classes,
            [index],
        )
        if reps > 1:
            visualize_mean_variance(ensemble_mask, phase, batch_idx, index_list=[index])

        return 0

    @torch.inference_mode(False)
    def dps_guidance(
        self,
        noisy_mask: torch.Tensor,
        images: torch.Tensor,
        current_t: int,
        batch_idx: int,
        gt_masks: torch.Tensor,
        last_step_unguided: bool,
    ):
        end_t = 1 if last_step_unguided else 0
        for t in range(current_t, end_t - 1, -1):
            noisy_mask = self.guided_step(noisy_mask, images, t, batch_idx)

            self.metric_handler.update(
                self.mask_transformer.get_segmentation(
                    self.mask_transformer.get_logits(noisy_mask.unsqueeze(0))
                ),
                gt_masks,
                t,
            )

        return noisy_mask

    @torch.inference_mode(False)
    def only_guidance(
        self,
        noisy_mask: torch.Tensor,
        current_t: int,
        batch_idx: int,
        gt_masks: torch.Tensor,
        last_step_unguided: bool,
    ):
        end_t = 1 if last_step_unguided else 0
        for t in range(current_t, end_t - 1, -1):
            noisy_mask = self.only_guided_step(
                noisy_mask, t, batch_idx, last_step_unguided
            )

            self.metric_handler.update(
                self.mask_transformer.get_segmentation(
                    self.mask_transformer.get_logits(noisy_mask.unsqueeze(0))
                ),
                gt_masks,
                t,
            )

        return noisy_mask

    @torch.inference_mode(False)
    def only_guided_step(
        self,
        noisy_mask: torch.Tensor,
        t: int,
        batch_idx: int,
        gamma: float | None = None,
        return_gradients: bool = False,
    ):
        noisy_mask = noisy_mask.requires_grad_(True)

        loss = self.loss_guider.guidance_loss(
            noisy_mask,
            t,
            batch_idx,
        )
        print(f"loss at timestep {t}: {loss}")

        self.metric_handler.update_loss(
            {self.loss_guider.loss_name: loss.item()},
            t,
        )

        loss.backward()

        with torch.no_grad():
            noisy_mask_grads = noisy_mask.grad
            new_noisy_mask = (
                noisy_mask - (self.gamma if gamma is None else gamma) * noisy_mask_grads
            )

        if return_gradients:
            return new_noisy_mask, noisy_mask_grads
        else:
            return new_noisy_mask
