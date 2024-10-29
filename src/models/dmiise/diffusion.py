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
from utils.visualize import visualize_sampling_res, load_res_to_wandb, create_wandb_image, visualize_segmentation
from monai.losses import DiceLoss, DiceCELoss
from utils.metrics import compute_and_log_metrics
from utils.mask_transformer import BaseMaskMapping


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
        self.mask_transformer.set_threshold_func(self.threshold)

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
        images, gt_masks, gt_train_masks = self.unpack_batch(batch)
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
        images, gt_masks, gt_train_masks = self.unpack_batch(batch)
        num_samples = images.shape[0]
        ensemble_mask = torch.zeros_like(images, device=images.device)

        self.model.eval()

        with torch.no_grad():
            for reps in range(self.repetitions):
                noisy_mask = torch.rand_like(images, device=images.device)

                for t in tqdm(self.scheduler.timesteps):
                    model_output = self.model(torch.cat((noisy_mask, images), dim=1), torch.full((num_samples,), t, device=images.device))
                    noisy_mask = self.scheduler.step(model_output=model_output, timestep=t, sample=noisy_mask).prev_sample

                ensemble_mask = torch.add(ensemble_mask, noisy_mask)

        self.model.train()
        ensemble_mask = ensemble_mask/self.repetitions

        pred_masks = self.mask_transformer.create_gt_mask_from_pred(ensemble_mask)
        
        print(f"histogram of pred_mask: {torch.histc(pred_masks[0], bins=10, min=0, max=1)}")
        print(f"histogram of gt_mask: {torch.histc(gt_masks[0], bins=10, min=0, max=1)}")

        compute_and_log_metrics(self.metrics, pred_masks, gt_masks, phase, self.log)        
        visualize_segmentation(images, gt_masks, pred_masks, ensemble_mask, phase, self.mask_transformer.gt_mapping_for_visualization(), batch_idx)
        
        return 0
    


    # def validation_step(self, batch, batch_idx):
    #     images, gt_masks, gt_masks_train = batch
        
    #     num_images = images.shape[0]
    #     noisy_mask = torch.rand_like(images, device=images.device)

    #     index = random.randint(0, num_images-1)

    #     self.model.eval()
    #     ensemble_mask = torch.zeros_like(images, device=images.device)
    #     with torch.no_grad():
    #         for reps in range(self.repetitions):
    #             noisy_mask = torch.rand_like(images, device=images.device)

    #             for t in tqdm(self.scheduler.timesteps):
    #                 model_output = self.model(torch.cat((noisy_mask, images), dim=1), torch.full((num_images,), t, device=images.device))

    #                 noisy_mask = self.scheduler.step(model_output=model_output, timestep=t, sample=noisy_mask).prev_sample

    #                 # if t < 200 and t % 20 == 0:
    #                 #     mask_step = load_res_to_wandb(images[index], pred_mask=noisy_mask[index] > 0.5, gt_mask=None, caption=f"{batch_idx}_{index} at step {t}")
    #                 #     wandb.log({"step_images": mask_step})

    #             print(f"max bevor clamping: {torch.max(noisy_mask)}")
    #             print(f"min bevor clamping: {torch.min(noisy_mask)}")
    #             ensemble_mask = torch.add(ensemble_mask, noisy_mask)
    #     self.model.train()

    #     ensemble_mask = ensemble_mask/self.repetitions
        
        
    #     pred_masks = torch.where(ensemble_mask > self.threshold, 1.0, 0.0)
    #     val_images = load_res_to_wandb(images[index], gt_masks[index], pred_masks[index], caption=f"BIdx_{batch_idx}_Idx_{index}")

    #     wandb.log({"val_examples": val_images})
    #     mask_image = create_wandb_image(pred_masks[index])
    #     gt_image = create_wandb_image(gt_masks[index])

    #     wandb.log({"pred_masks": mask_image})
    #     wandb.log({"gt_masks": gt_image})

    #     compute_and_log_metrics(self.metrics, pred_masks, gt_masks, "val", self.log)

    #     # pred_masks = torch.where(noisy_mask.clamp(-1, 1) > 0.0, 1.0, 0.0)
    #     # print(f"distribution of pred mask: {torch.histc(pred_masks[index], bins=10, min=0, max=1)}")
    #     # print(f"how many ones: {(pred_masks[index] == 1.0).sum()}")
    #     # val_images = load_res_to_wandb(images[index], gt_masks[index] < 0, pred_masks[index], caption=f"BIdx_{batch_idx}_Idx_{index}")
    #     # wandb.log({"val_examples": val_images})
        
    #     print(f"max bevor clamping ens: {torch.max(ensemble_mask[index])}")
    #     print(f"min bevor clamping ens: {torch.min(ensemble_mask[index])}")

    #     # final_pic = ((ensemble_mask + 1) / 2).clamp(0,1)
    #     final_pic = (ensemble_mask).clamp(0,1)


    #     final_img = create_wandb_image(final_pic[index], caption=f"{batch_idx}_{index} at step {t}")
    #     wandb.log({"final pic": final_img})

    #     print(f"histogram of final pic: {torch.histc(final_pic[index], bins=10, min=0, max=1)}")
    #     print(f"max after clamping: {torch.max(final_pic[index])}")
    #     print(f"min after clamping: {torch.min(final_pic[index])}")

    #     # compute_and_log_metrics(self.metrics, pred_masks, gt_masks, "val", self.log)

    #     return 0

    # def validation_step(self, batch, batch_idx):
    #     images, gt_masks = batch
        
    #     num_images = images.shape[0]
    #     noisy_mask = torch.rand_like(images, device=images.device)

    #     index = random.randint(0, num_images-1)

    #     self.model.eval()
    #     ensemble_mask = torch.zeros_like(images, device=images.device)
    #     with torch.no_grad():
    #         for reps in range(self.repetitions):
    #             noisy_mask = torch.rand_like(images, device=images.device)

    #             for t in tqdm(self.scheduler.timesteps):
    #                 model_output = self.model(torch.cat((noisy_mask, images), dim=1), torch.full((num_images,), t, device=images.device))

    #                 noisy_mask = self.scheduler.step(model_output=model_output, timestep=t, sample=noisy_mask).prev_sample

    #                 # if t < 200 and t % 20 == 0:
    #                 #     mask_step = load_res_to_wandb(images[index], pred_mask=noisy_mask[index] > 0.5, gt_mask=None, caption=f"{batch_idx}_{index} at step {t}")
    #                 #     wandb.log({"step_images": mask_step})

    #             print(f"max bevor clamping: {torch.max(noisy_mask[index])}")
    #             print(f"min bevor clamping: {torch.min(noisy_mask[index])}")
    #             ensemble_mask = torch.add(ensemble_mask, noisy_mask)
    #     self.model.train()

    #     ensemble_mask = ensemble_mask/self.repetitions
        
        
    #     pred_masks = torch.where((ensemble_mask).clamp(0, 1) > self.threshold, 1.0, 0.0)
    #     val_images = load_res_to_wandb(images[index], gt_masks[index], pred_masks[index], caption=f"BIdx_{batch_idx}_Idx_{index}")

    #     wandb.log({"val_examples": val_images})
    #     mask_image = create_wandb_image(pred_masks[index])
    #     gt_image = create_wandb_image(gt_masks[index])

    #     wandb.log({"pred_masks": mask_image})
    #     wandb.log({"gt_masks": gt_image})

    #     compute_and_log_metrics(self.metrics, pred_masks, gt_masks, "val", self.log)

    #     # pred_masks = torch.where(noisy_mask.clamp(-1, 1) > 0.0, 1.0, 0.0)
    #     # print(f"distribution of pred mask: {torch.histc(pred_masks[index], bins=10, min=0, max=1)}")
    #     # print(f"how many ones: {(pred_masks[index] == 1.0).sum()}")
    #     # val_images = load_res_to_wandb(images[index], gt_masks[index] < 0, pred_masks[index], caption=f"BIdx_{batch_idx}_Idx_{index}")
    #     # wandb.log({"val_examples": val_images})
        
    #     print(f"max bevor clamping ens: {torch.max(ensemble_mask[index])}")
    #     print(f"min bevor clamping ens: {torch.min(ensemble_mask[index])}")

    #     final_pic = ensemble_mask.clamp(0, 1)
    #     final_img = create_wandb_image(final_pic[index], caption=f"{batch_idx}_{index} at step {t}")
    #     wandb.log({"final pic": final_img})

    #     print(f"histogram of final pic: {torch.histc(final_pic[index], bins=10, min=0, max=1)}")
    #     print(f"max after clamping: {torch.max(final_pic[index])}")
    #     print(f"min after clamping: {torch.min(final_pic[index])}")

    #     # compute_and_log_metrics(self.metrics, pred_masks, gt_masks, "val", self.log)

    #     return 0


      ### Old Training Step Function
    # def training_step(self, batch, batch_idx):
    #     images, gt_masks, gt_masks_train = batch
    #     num_images = gt_masks_train.shape[0]
    #     noise = torch.randn_like(gt_masks_train, device=gt_masks.device)

    #     timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (num_images,), device=gt_masks.device, dtype=torch.int64)
    #     noisy_image = self.scheduler.add_noise(gt_masks_train, noise, timesteps)

    #     pred_noise = self.model(torch.cat((noisy_image, images), dim=1), timesteps)

    #     loss = self.loss_fn(pred_noise, noise)
    #     self.log('train_loss', loss)

    #     return loss
    

    # def test_step(self, batch, batch_idx):
    #     images, gt_masks, gt_masks_train = batch
    #     num_images = images.shape[0]
    #     noisy_mask = torch.rand_like(images, device=images.device)

    #     index = random.randint(0, num_images-1)

    #     self.model.eval()
    #     with torch.no_grad():
    #         for t in tqdm(self.scheduler.timesteps):
    #             model_output = self.model(torch.cat((noisy_mask, images), dim=1), torch.ones(num_images, device=images.device) * t)

    #             noisy_mask = self.scheduler.step(model_output=model_output, timestep=t, sample=noisy_mask).prev_sample


    #     self.model.train()

    #     pred_masks = torch.where(noisy_mask > self.threshold, 1.0, 0.0)
    #     val_images = load_res_to_wandb(images[index], gt_masks[index], pred_masks[index], caption=f"BIdx_{batch_idx}_Idx_{index}")

    #     wandb.log({"test_examples": val_images})
    #     compute_and_log_metrics(self.metrics, pred_masks, gt_masks, "test", self.log)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.optimizer_config.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    