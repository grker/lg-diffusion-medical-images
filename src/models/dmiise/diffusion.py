import pytorch_lightning as pl
import torch
import math
import random
import wandb

import torch.nn as nn
from torch.utils.data import Dataset

from tqdm import tqdm
from diffusers.schedulers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

from utils.hydra_config import DiffusionConfig, OptimizerConfig
from utils.visualize import visualize_segmentation, create_wandb_image, visualize_mean_variance
from utils.metrics import compute_and_log_metrics
from utils.mask_transformer import BaseMaskMapping
from utils.helper import unpack_batch


def scheduler_factory(scheduler_type: str, beta_start: float, beta_end: float, noise_steps: int):
    if scheduler_type == "linear":
        return torch.linspace(beta_start, beta_end, noise_steps)
    else:
        raise NotImplementedError(f"Scheduler {scheduler_type} not implemented")


        
class DDPM(pl.LightningModule):
    scheduler: DDPMScheduler
    model: nn.Module
    optimizer_config: OptimizerConfig
    metrics: dict
    mask_transformer: BaseMaskMapping
    num_classes: int # always includes the background
    
    def __init__(self, model: nn.Module, diffusion_config: DiffusionConfig, optimizer_config: OptimizerConfig, metrics: dict, mask_transformer: BaseMaskMapping, loss: torch.nn.Module):
        super().__init__()
        
        self.scheduler = DDPMScheduler(num_train_timesteps=diffusion_config.noise_steps, beta_start=diffusion_config.beta_start, beta_end=diffusion_config.beta_end, beta_schedule=diffusion_config.scheduler_type, prediction_type=diffusion_config.prediction_type, clip_sample_range=diffusion_config.clip_range)
        
        if diffusion_config.num_inference_steps is None or diffusion_config.num_inference_steps>diffusion_config.noise_steps:
            self.scheduler.set_timesteps(diffusion_config.noise_steps)
        else:
            self.scheduler.set_timesteps(diffusion_config.num_inference_steps)

        self.model = model
        self.optimizer_config = optimizer_config
        self.metrics = metrics
        self.repetitions = diffusion_config.repetitions
        self.threshold = diffusion_config.threshold
        self.prediction_type = diffusion_config.prediction_type
        self.mask_transformer = mask_transformer
        self.num_classes = mask_transformer.get_num_classes()

        # if hasattr(self.mask_transformer, "set_threshold_func"):
        #     getattr(self.mask_transformer, "set_threshold_func")(self.threshold)

        self.loss_fn = loss

    def noise_tester(self, dataset: Dataset, batch_size: int, device="cuda"):
        samples = len(dataset)
        batch_start_idx = 0

        flatt_values = torch.empty(0).to(device)
        loss_fn = torch.nn.MSELoss(reduction='mean')
        
        while batch_start_idx < samples:
            end_idx = min(batch_start_idx + batch_size, samples)
            images, masks = dataset[batch_start_idx:end_idx]
            batch = masks.to(device)

            noise = torch.randn_like(batch, device=batch.device)
            timesteps = torch.full((noise.shape[0],), self.scheduler.config.num_train_timesteps-1).to(batch.device)
            noisy_batch = self.scheduler.add_noise(batch, noise, timesteps)

            flatt_values = torch.cat((flatt_values,torch.flatten(noisy_batch)), dim=0)
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
            raise ValueError(f"Batch has {len(batch)} to unpack. A batch can only have 2 or 3 values to unpack.")
        
        assert(images.shape[0] == gt_masks.shape[0] and  gt_masks.shape[0] == gt_train_masks.shape[0], "Assertion Error: images, gt_masks and gt_train_masks need to have the same number of samples")

        return images, gt_masks, gt_train_masks
    
        
    def training_step(self, batch, batch_idx):
        self.log("learning rate", self.trainer.optimizers[0].param_groups[0]["lr"])
        images, gt_masks, gt_train_masks = unpack_batch(batch)
        num_samples = images.shape[0]

        noise = torch.randn_like(gt_train_masks, device=gt_train_masks.device)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (num_samples,), device=gt_train_masks.device, dtype=torch.int64)
        noisy_image = self.scheduler.add_noise(gt_train_masks, noise, timesteps)
        
        prediction = self.model(torch.cat((noisy_image, images), dim=1), timesteps)

        loss = 0.0
        if self.prediction_type == "epsilon":
            loss = self.loss_fn(prediction, noise)
        elif self.prediction_type == "sample":
            loss = self.loss_fn(prediction, gt_train_masks)
        
        self.log('train_loss', loss)

        return loss
    

    def validation_step(self, batch, batch_idx):
        return self.val_test_step(batch,batch_idx,"val")
    

    def test_step(self,batch,batch_idx):
        return self.val_test_step(batch,batch_idx,"test")
    
    
    def val_test_step(self, batch, batch_idx, phase):
        images, gt_masks, gt_train_masks = unpack_batch(batch)
        num_samples = images.shape[0]
        ensemble_shape = (self.repetitions, *gt_train_masks.shape)
        ensemble_mask = torch.zeros(ensemble_shape, device=images.device)

        self.model.eval()
        with torch.no_grad():
            for reps in range(self.repetitions):
                noisy_mask = torch.rand_like(gt_train_masks, device=images.device)
                for t in tqdm(self.scheduler.timesteps):
                    model_output = self.model(torch.cat((noisy_mask, images), dim=1), torch.full((num_samples,), t, device=images.device)).detach()
                    noisy_mask = self.scheduler.step(model_output=model_output, timestep=t, sample=noisy_mask).prev_sample
                    del model_output  # Free model_output explicitly
                    torch.cuda.empty_cache()  # Clear memory cache if needed
                ensemble_mask[reps] = noisy_mask.detach()  # Detach to prevent gradient history

        self.model.train()

        logits = self.mask_transformer.get_logits(ensemble_mask)
        seg_mask, one_hot_seg_mask = self.mask_transformer.get_segmentation(logits)

        # Detach before passing for metric computation
        compute_and_log_metrics(self.metrics, seg_mask.detach(), gt_masks.detach(), phase, self.log)

        # Visualization
        index = random.randint(0, num_samples - 1)
        visualize_segmentation(
            images, gt_masks, seg_mask, ensemble_mask, phase, 
            self.mask_transformer.gt_mapping_for_visualization(), batch_idx, self.num_classes, [index]
        )
        if self.repetitions > 1:
            visualize_mean_variance(ensemble_mask, phase, batch_idx, index_list=[index])

        # Clear ensemble_mask if not needed further
        del ensemble_mask
        del logits
        del seg_mask
        del one_hot_seg_mask
        torch.cuda.empty_cache()
    
        return 0

    

    # def val_test_step(self, batch, batch_idx, phase):
    #     images, gt_masks, gt_train_masks = unpack_batch(batch)
    #     num_samples = images.shape[0]
    #     ensemble_mask = torch.zeros_like(images, device=images.device)

    #     self.model.eval()

    #     ensemble_shape = (self.repetitions, *gt_train_masks.shape)
    #     print(f"ensemble shape: {ensemble_shape}")
    #     ensemble_mask = torch.zeros(ensemble_shape, device=images.device)
    #     with torch.no_grad():
    #         for reps in range(self.repetitions):
    #             noisy_mask = torch.rand_like(gt_train_masks, device=images.device)

    #             for t in tqdm(self.scheduler.timesteps):
    #                 model_output = self.model(torch.cat((noisy_mask, images), dim=1), torch.full((num_samples,), t, device=images.device))
    #                 noisy_mask = self.scheduler.step(model_output=model_output, timestep=t, sample=noisy_mask).prev_sample

    #             ensemble_mask[reps] = noisy_mask

    #     self.model.train()
    #     # ensemble_mask = ensemble_mask/self.repetitions

    #     logits = self.mask_transformer.get_logits(ensemble_mask)
    #     seg_mask, one_hot_seg_mask = self.mask_transformer.get_segmentation(logits)
        
    #     print(f"histogram of seg_mask ranging from {torch.min(seg_mask)} to {torch.max(seg_mask)}: {torch.histc(seg_mask[0], bins=10)}")
    #     print(f"histogram of gt_mask ranging from {torch.min(gt_masks)} to {torch.max(gt_masks)}: {torch.histc(gt_masks[0], bins=10)}")

    #     compute_and_log_metrics(self.metrics, seg_mask, gt_masks, phase, self.log)

    #     # visualize the segmentation
    #     index = random.randint(0, num_samples-1)        
    #     visualize_segmentation(images, gt_masks, seg_mask, ensemble_mask, phase, self.mask_transformer.gt_mapping_for_visualization(), batch_idx, self.num_classes, [index])
    #     if self.repetitions > 1:
    #         # need multiple samples per sample in order to compute the std
    #         visualize_mean_variance(ensemble_mask, phase, batch_idx, index_list=[index])

    #     return 0
    

    def test_variance(self, image: torch.Tensor, gt_mask: torch.Tensor, gt_train_mask: torch.Tensor, reps: int, batch_idx: int):
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
                model_output = self.model(torch.cat((noisy_mask, image), dim=1), torch.full((num_samples,), t, device=image.device))
                noisy_mask = self.scheduler.step(model_output=model_output, timestep=t, sample=noisy_mask).prev_sample

            ensemble_mask[r] = noisy_mask

        
        mean_mask = torch.mean(ensemble_mask, dim=0)
        std_mask = torch.std(ensemble_mask, dim=0)

        min_per_sample_mean = torch.min(torch.flatten(mean_mask, start_dim=1), dim=1).values
        max_per_sample_mean = torch.max(torch.flatten(mean_mask, start_dim=1), dim=1).values

        min_per_sample_std = torch.min(torch.flatten(std_mask, start_dim=1), dim=1).values
        max_per_sample_std = torch.max(torch.flatten(std_mask, start_dim=1), dim=1).values
        
        mean_mask = (mean_mask - min_per_sample_mean) / (max_per_sample_mean - min_per_sample_mean)
        std_mask = (std_mask - min_per_sample_std) / (max_per_sample_std - min_per_sample_std)

        for idx in range(num_samples):
            wandb.log({"mean_mask": create_wandb_image(mean_mask[idx], caption=f"Testing VarianceBIdx_{batch_idx}_Idx_{idx}")})
            wandb.log({"std_mask": create_wandb_image(std_mask[idx], caption=f"Testing VarianceBIdx_{batch_idx}_Idx_{idx}")})

    

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.optimizer_config.lr)

        if hasattr(torch.optim.lr_scheduler, self.optimizer_config.scheduler.name):
            args = self.optimizer_config.scheduler.args if self.optimizer_config.scheduler.args is not None else {}
            scheduler = getattr(torch.optim.lr_scheduler, self.optimizer_config.scheduler.name)(optimizer, **args)
        else:
            raise ValueError(f"Scheduler {self.optimizer_config.scheduler.name} not found!")

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    