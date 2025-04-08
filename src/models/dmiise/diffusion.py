import random

# import time
import pytorch_lightning as pl
import torch
import torch.nn as nn
from diffusers.schedulers import DDIMScheduler, DDPMScheduler
from torch.utils.data import Dataset
from tqdm import tqdm

import wandb
from guidance import LossGuider
from loss import CustomLoss
from metrics import MetricsHandler, MetricsInput
from utils.helper import EMA, unpack_batch
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
        self.scheduler = self.create_scheduler(diffusion_config)

        self.sampling_scheduler = self.create_sampling_scheduler(diffusion_config)

        if (
            diffusion_config.num_inference_steps is None
            or diffusion_config.num_inference_steps > diffusion_config.noise_steps
        ):
            self.sampling_scheduler.set_timesteps(diffusion_config.noise_steps)
        else:
            self.sampling_scheduler.set_timesteps(diffusion_config.num_inference_steps)

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

    def create_scheduler(self, diffusion_config: DiffusionConfig):
        return DDIMScheduler(
            num_train_timesteps=diffusion_config.noise_steps,
            beta_start=diffusion_config.beta_start,
            beta_end=diffusion_config.beta_end,
            beta_schedule=diffusion_config.scheduler_type,
            prediction_type=diffusion_config.prediction_type,
        )

    def create_sampling_scheduler(self, diffusion_config: DiffusionConfig):
        return DDPMScheduler(
            num_train_timesteps=diffusion_config.noise_steps,
            beta_start=diffusion_config.beta_start,
            beta_end=diffusion_config.beta_end,
            beta_schedule=diffusion_config.scheduler_type,
            prediction_type=diffusion_config.prediction_type,
        )

    def create_ema(self, optimizer_config: OptimizerConfig):
        if "ema" in optimizer_config.keys() and optimizer_config.ema is not None:
            self.ema = EMA(**optimizer_config.ema)
        else:
            self.ema = EMA(1)

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
            alphas_cumprod = self.scheduler.alphas_cumprod.to(gt_train_masks.device)
            sqrt_alpha_t = torch.sqrt(alphas_cumprod[timesteps])[:, None, None, None]
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alphas_cumprod[timesteps])[
                :, None, None, None
            ]

            # Reconstruct noise target from noisy image and clean image
            reconstructed_noise = (
                noisy_image - sqrt_alpha_t * gt_train_masks
            ) / sqrt_one_minus_alpha_t

            print(
                f"mse loss noise: {torch.nn.functional.mse_loss(noise, reconstructed_noise)}"
            )
            loss = self.loss_fn(prediction, reconstructed_noise)
        elif self.prediction_type == "sample":
            loss = self.loss_fn(prediction, gt_train_masks)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        return self.val_test_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.val_test_step(batch, batch_idx, "test")

    def get_model_output(
        self, noisy_mask: torch.Tensor, images: torch.Tensor, timestep: int
    ):
        num_samples = images.shape[0]

        model_output = self.model(
            torch.cat((noisy_mask, images), dim=1),
            torch.full((num_samples,), timestep, device=images.device),
        )

        return model_output

    def val_test_step(self, batch, batch_idx, phase):
        self.model.eval()

        print(f"prediction type scheduler: {self.scheduler.prediction_type}")

        generator = torch.Generator(device=self.device)
        seed = batch_idx * self.current_epoch
        generator.manual_seed(seed)

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
                    model_output = self.get_model_output(noisy_mask, images, t)

                    noisy_mask = self.sampling_scheduler.step(
                        model_output=model_output,
                        timestep=t,
                        sample=noisy_mask,
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
        if "name" in self.optimizer_config.keys():
            if self.optimizer_config.name == "sgd":
                optimizer = torch.optim.SGD(
                    self.model.parameters(),
                    lr=self.optimizer_config.lr,
                    weight_decay=self.optimizer_config.weight_decay,
                    momentum=0.0,
                )
                print("using the sgd optimizer")
            elif self.optimizer_config.name == "adamw":
                optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.optimizer_config.lr,
                    weight_decay=self.optimizer_config.weight_decay,
                )
                print("using the adamw optimizer")
            else:
                optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=self.optimizer_config.lr,
                    weight_decay=self.optimizer_config.weight_decay,
                )
                print("using the adam optimizer")
        else:
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.optimizer_config.lr,
                weight_decay=self.optimizer_config.weight_decay,
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


class DDPM_DPS_Regularized(DDPM):
    loss_guider: LossGuider
    starting_step: int
    stop_step: int
    gamma: float

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

        # set up the loss guider
        self.loss_guider = self.initialize_guider(diffusion_config.loss_guidance.guider)
        losses = [self.loss_guider.loss_name]

        self.set_mode(diffusion_config.loss_guidance)

        # set up the regularizer if exists
        if "regularizer" in diffusion_config.loss_guidance.keys():
            self.regularizer = True
            self.regularized_loss = CustomLoss(
                diffusion_config.loss_guidance.regularizer.reg_loss
            )
            self.weighting = diffusion_config.loss_guidance.regularizer.weighting
            self.regularized_loss_name = "Regularized Loss"
            self.repeated = diffusion_config.loss_guidance.regularizer.repeated
            self.average_ensemble = (
                diffusion_config.loss_guidance.regularizer.average_ensemble
            )
            self.mode_for_reference_mask = (
                diffusion_config.loss_guidance.regularizer.mode_for_reference_mask
            )

            losses.append(self.regularized_loss_name)
        else:
            self.regularizer = False

        self.metric_handler.add_losses(losses)

    def initialize_guider(self, guider_config: GuiderConfig):
        import guidance

        if hasattr(guidance, guider_config.name):
            print(f"guider config: {guider_config}")
            return getattr(guidance, guider_config.name)(guider_config)
        else:
            raise ValueError(f"PseudoGTGenerator {guider_config.name} not found!")

    def set_mode(self, loss_guidance_config: LossGuidance3StepConfig):
        self.mode = loss_guidance_config.mode
        self.gamma = loss_guidance_config.gamma
        self.visualize_gradients = loss_guidance_config.visualize_gradients

        self.starting_step = loss_guidance_config.starting_step
        self.stop_step = loss_guidance_config.stop_step

        assert (
            self.stop_step <= self.starting_step
            and self.stop_step >= 0
            and self.starting_step <= len(self.sampling_scheduler.timesteps) - 1
        ), "Starting step must be larger than or equal to stop step"

        if self.stop_step == 0:
            self.stop_step = 1
            self.no_switch = True
        else:
            self.no_switch = False

    def compute_regularized_loss(
        self, model_output: torch.Tensor, referenced_mask: torch.Tensor
    ):
        return self.regularized_loss(model_output, referenced_mask)

    def get_softmax_prediction(self, output: torch.Tensor, clamping: bool = False):
        if clamping:
            output = torch.clamp(output, -1, 1)
        return torch.softmax(output, dim=1)

    def prepare_reference_mask(
        self, reference_mask: torch.Tensor, mode: str = "segmentation"
    ):
        if mode in ["segmentation", "softmax"]:
            reference_mask = self.get_softmax_prediction(reference_mask, clamping=False)

            if mode == "segmentation":
                reference_mask_argmax = torch.argmax(
                    reference_mask, dim=1, keepdim=True
                )
                reference_mask = torch.zeros_like(reference_mask).scatter_(
                    1, reference_mask_argmax, 1
                )

        return reference_mask

    @torch.inference_mode(False)
    def val_test_step(self, batch, batch_idx, phase):
        self.model.eval()

        images, gt_masks, gt_train_masks, topo_inputs = unpack_batch(batch, "test")
        num_samples = images.shape[0]
        reps = self.repetitions_test if phase == "test" else self.repetitions

        # Run the whole backward pass once to get the reference mask, all steps are unguided
        if self.regularizer:
            reference_mask = torch.rand_like(gt_train_masks, device=images.device)

            for t in tqdm(self.sampling_scheduler.timesteps):
                reference_mask = self.unguided_step(reference_mask, gt_masks, images, t)

            # add the reference mask to the topo_inputs
            topo_inputs["reference_mask"] = self.prepare_reference_mask(
                reference_mask, self.mode_for_reference_mask
            )
        else:
            print("no regularizer!")

        ensemble_mask = []
        # run the actual guided backward pass
        for rep in range(reps):
            noisy_mask = torch.rand_like(gt_train_masks, device=images.device)

            # unguided diffusion steps
            for t in tqdm(self.sampling_scheduler.timesteps[: -self.starting_step]):
                noisy_mask = self.unguided_step(noisy_mask, gt_masks, images, t)

            # activate the gradient for the model for the guided steps
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    param.requires_grad_(True)

            noisy_mask = noisy_mask.requires_grad_(True)

            # guided steps
            for t in self.sampling_scheduler.timesteps[
                -self.starting_step : -self.stop_step
            ]:
                noisy_mask = self.guided_step(
                    noisy_mask, images, topo_inputs, t, batch_idx
                )
                self.guidance_metrics(noisy_mask, gt_masks, topo_inputs, t)

            if self.no_switch:
                noisy_mask = self.guided_step(
                    noisy_mask, images, topo_inputs, 0, batch_idx
                )
                self.guidance_metrics(noisy_mask, gt_masks, topo_inputs, 0)

                # deactivate the gradient for the model
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        param.requires_grad_(False)

            else:
                # again unguided steps
                for t in self.sampling_scheduler.timesteps[-self.stop_step :]:
                    noisy_mask = noisy_mask.requires_grad_(False)
                    noisy_mask = self.unguided_step(noisy_mask, gt_masks, images, t)

                    self.guidance_metrics(noisy_mask, gt_masks, topo_inputs, t)

                # final loss:
                # loss = self.loss_guider.guidance_loss(
                #     noisy_mask,
                #     0,
                #     batch_idx,
                #     **topo_inputs,
                # )
                # loss = loss.view(1)
                # print(f"final guidance loss: {loss}")
                # loss_update = {self.loss_guider.loss_name: loss.item()}

                # if self.regularizer:
                #     reg_loss = self.compute_regularized_loss(noisy_mask, topo_inputs["reference_mask"])
                #     print(f"final regularized loss: {reg_loss}")
                #     loss += reg_loss * self.scalar_reg_loss
                #     loss_update[self.regularized_loss_name] = reg_loss.item()

                # print(f"final total loss: {loss}")

                # self.metric_handler.update_loss(
                #     {self.loss_guider.loss_name: loss.item()},
                #     0,
                # )

            ensemble_mask.append(noisy_mask.detach().cpu())

            if self.regularizer and self.repeated:
                topo_inputs["reference_mask"] = self.prepare_reference_mask(
                    noisy_mask, self.mode_for_reference_mask
                )

            self.metric_handler.log_guidance_metrics()

        ensemble_mask = torch.stack(ensemble_mask, dim=0)
        if self.regularizer and self.repeated and self.final_mask == "last":
            ensemble_mask = ensemble_mask[-1].unsqueeze(0)

        logits = self.mask_transformer.get_logits(ensemble_mask)

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

    def guidance_metrics(
        self,
        noisy_mask: torch.Tensor,
        gt_masks: torch.Tensor,
        topo_inputs: dict[str, torch.Tensor],
        t: int,
    ):
        inputs = MetricsInput(
            self.mask_transformer.get_segmentation(
                self.mask_transformer.get_logits(noisy_mask.unsqueeze(0))
            ),
            gt_masks,
            topo_inputs,
        )

        self.metric_handler.update_guidance_metrics(inputs, t)

    @torch.inference_mode(False)
    def unguided_step(
        self,
        noisy_mask: torch.Tensor,
        gt_masks: torch.Tensor,
        images: torch.Tensor,
        t: int,
        generator: torch.Generator | None = None,
    ):
        with torch.no_grad():
            num_samples = images.shape[0]
            model_output = self.model(
                torch.cat((noisy_mask, images), dim=1),
                torch.full((num_samples,), t, device=images.device),
            )

            noisy_mask = self.sampling_scheduler.step(
                model_output=model_output,
                timestep=t,
                sample=noisy_mask,
                generator=generator,
            ).prev_sample

        segmentation = self.mask_transformer.get_segmentation(model_output.unsqueeze(0))
        metrics_input = MetricsInput(segmentation, gt_masks, {})

        self.metric_handler.update_guidance_metrics(metrics_input, t)

        model_output = model_output.detach().cpu()

        return noisy_mask

    @torch.inference_mode(False)
    def guided_step(
        self,
        noisy_mask: torch.Tensor,
        images: torch.Tensor,
        topo_inputs: dict[str, torch.Tensor],
        t: int,
        batch_idx: int,
        gamma: float | None = None,
        return_gradients: bool = False,
    ):
        noisy_mask = noisy_mask.requires_grad_(True)
        prediction = None

        if self.mode == "only_guided":
            prediction = noisy_mask
        elif self.mode == "dps_guidance" or self.mode == "dps_only_reg":
            prediction = self.model(
                torch.cat((noisy_mask, images), dim=1),
                torch.full((images.shape[0],), t, device=images.device),
            )
        else:
            raise ValueError(f"Mode {self.mode} not supported!")

        prediction.retain_grad()

        loss_update = {}
        loss = 0.0
        if self.mode != "dps_only_reg":
            guidance_loss = self.loss_guider.guidance_loss(
                prediction,
                t,
                batch_idx,
                **topo_inputs,
            )
            guidance_loss = guidance_loss.view(1)
            print(f"\nguidance loss at timestep {t}: {guidance_loss.item()}")

            loss += guidance_loss
            loss_update = {self.loss_guider.loss_name: guidance_loss.item()}

        if self.regularizer:
            reg_loss = self.compute_regularized_loss(
                prediction, topo_inputs["reference_mask"]
            )
            print(f"regularized loss at timestep {t}: {reg_loss.item()}")

            loss = self.weighting * loss + (1 - self.weighting) * reg_loss
            loss_update[self.regularized_loss_name] = reg_loss.item()

        print(f"total loss at timestep {t}: {loss.item()}")

        # update the loss for the metric handler
        self.metric_handler.update_loss(
            loss_update,
            t,
        )

        # compute the gradients and update the noisy mask
        loss.backward()

        with torch.no_grad():
            noisy_mask_grads = (
                self.gamma if gamma is None else gamma
            ) * noisy_mask.grad

            print(f"noisy_mask_grads max: {noisy_mask_grads.max()}")
            print(f"noisy_mask_grads min: {noisy_mask_grads.min()}")
            if self.mode == "only_guided":
                new_noisy_mask = noisy_mask - noisy_mask_grads
            elif self.mode == "dps_guidance" or self.mode == "dps_only_reg":
                noisy_mask = self.sampling_scheduler.step(
                    model_output=prediction, timestep=t, sample=noisy_mask
                ).prev_sample
                new_noisy_mask = noisy_mask - noisy_mask_grads
            else:
                raise ValueError(f"Mode {self.mode} not supported!")

            if self.visualize_gradients:
                random_idx = int(images.shape[0] / 2)
                visualize_gradients(
                    noisy_mask_grads[random_idx],
                    t,
                    batch_idx,
                    random_idx,
                    prediction[random_idx],
                    commit=(t == 1),
                )

        prediction = prediction.detach().cpu()
        noisy_mask = noisy_mask.detach().cpu()

        if return_gradients:
            return new_noisy_mask, noisy_mask_grads
        else:
            return new_noisy_mask


def max_min_normalization(model_output: torch.Tensor):
    """
    params:
        model_output: torch.Tensor, shape (batch_size, num_classes, height, width)
    returns:
        torch.Tensor, shape (batch_size, num_classes, height, width)
    """
    sample_min = torch.amin(model_output, dim=(1, 2, 3), keepdim=True)
    sample_max = torch.amax(model_output, dim=(1, 2, 3), keepdim=True)
    return (model_output - sample_min) / (sample_max - sample_min)


class DDPM_DPS_Regularized_Repeated(DDPM_DPS_Regularized):
    def __init__(
        self,
        model: nn.Module,
        diffusion_config: LossGuidedDiffusionConfig,
        optimizer_config: OptimizerConfig,
        metrics: dict,
        mask_transformer: BaseMaskMapping,
        loss: torch.nn.Module,
    ):
        if "reps_per_guided_step" in diffusion_config.loss_guidance.keys():
            self.reps_per_guided_step = (
                diffusion_config.loss_guidance.reps_per_guided_step
            )
        else:
            self.reps_per_guided_step = 1

        super().__init__(
            model, diffusion_config, optimizer_config, metrics, mask_transformer, loss
        )

    def val_test_step(self, batch, batch_idx, phase):
        return super().val_test_step(batch, batch_idx, phase)

    @torch.inference_mode(False)
    def guided_step(
        self,
        noisy_mask: torch.Tensor,
        images: torch.Tensor,
        topo_inputs: dict[str, torch.Tensor],
        t: int,
        batch_idx: int,
        gamma: float | None = None,
        return_gradients: bool = False,
    ):
        prediction = None

        noisy_mask = noisy_mask.requires_grad_(True)

        if self.mode == "only_guided":
            prediction = noisy_mask
        elif self.mode == "dps_guidance" or self.mode == "dps_only_reg":
            prediction = self.model(
                torch.cat((noisy_mask, images), dim=1),
                torch.full((images.shape[0],), t, device=images.device),
            )
        else:
            raise ValueError(f"Mode {self.mode} not supported!")

        # for name, param in self.model.named_parameters():
        #     if param.grad is not None:
        #         print(name, param.grad.shape)

        print(f"prediction has grad: {prediction.requires_grad}")

        for step_rep in range(self.reps_per_guided_step):
            print(f"iterating over reps {step_rep}")
            loss_update = {}
            loss = 0.0

            if self.mode != "dps_only_reg":
                guidance_loss = self.loss_guider.guidance_loss(
                    prediction,
                    t,
                    batch_idx,
                    **topo_inputs,
                )
                guidance_loss = guidance_loss.view(1)
                print(
                    f"\nguidance loss at timestep {t} and rep {step_rep}: {guidance_loss.item()}"
                )

                loss += guidance_loss
                loss_update = {self.loss_guider.loss_name: guidance_loss.item()}

            if self.regularizer:
                reg_loss = self.compute_regularized_loss(
                    prediction, topo_inputs["reference_mask"]
                )
                print(
                    f"regularized loss at timestep {t} and rep {step_rep}: {reg_loss.item()}"
                )

                loss = self.weighting * loss + (1 - self.weighting) * reg_loss
                loss_update[self.regularized_loss_name] = reg_loss.item()

            print(f"total loss at timestep {t} and rep {step_rep}: {loss.item()}")

            self.metric_handler.update_loss(
                loss_update,
                t,
            )

            prediction.retain_grad()

            loss.backward()

            with torch.no_grad():
                print(f"prediciton grads: {prediction.grad}")
                prediction_grads = (
                    self.gamma if gamma is None else gamma
                ) * prediction.grad

                print(f"prediction_grads max: {prediction_grads.max()}")
                print(f"prediction_grads min: {prediction_grads.min()}")

                prediction = prediction - prediction_grads.detach()
                prediction = prediction.detach()

                print(f"prediction has grad after: {prediction.requires_grad}")
                prediction = prediction.requires_grad_(True)

                print(f"prediction has grad after: {prediction.requires_grad}")

            print(f"prediction has grad: {prediction.requires_grad}")
            prediction = prediction.requires_grad_(True)

            print(f"prediction has grad after: {prediction.requires_grad}")

        if self.mode != "only_guided":
            noisy_mask = self.sampling_scheduler.step(
                model_output=prediction, timestep=t, sample=noisy_mask
            ).prev_sample

        if return_gradients:
            return noisy_mask, prediction_grads
        else:
            return noisy_mask
