import pytorch_lightning as pl
import torch
import math
import random
import wandb

import torch.nn as nn

from tqdm import tqdm
from diffusers.schedulers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

from utils.hydra_config import DiffusionConfig, OptimizerConfig
from utils.visualize import visualize_sampling_res, load_res_to_wandb
from monai.losses import DiceLoss, DiceCELoss
from utils.metrics import compute_and_log_metrics


def scheduler_factory(scheduler_type: str, beta_start: float, beta_end: float, noise_steps: int):
    if scheduler_type == "linear":
        return torch.linspace(beta_start, beta_end, noise_steps)
    else:
        raise NotImplementedError(f"Scheduler {scheduler_type} not implemented")


class Diffusion(nn.Module):
    noise_steps: int
    beta_start: float
    beta_end: float
    scheduler_type: str
    device: str
    val_learned: bool

    def __init__(self, config: DiffusionConfig, device: str="cpu"):
        super().__init__()
        self.noise_steps = config.noise_steps
        self.beta_start = config.beta_start
        self.beta_end = config.beta_end
        self.scheduler_type = config.scheduler_type
        self.device = device
        self.val_learned = config.var_learned

        self.beta = scheduler_factory(self.scheduler_type, self.beta_start, self.beta_end, self.noise_steps)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.alpha_hat_sqrt = torch.sqrt(self.alpha_hat)
        self.one_minus_alpha_hat_sqrt = torch.sqrt(1 - self.alpha_hat)
        self.alpha_hat_prev = torch.cat((torch.tensor([1.0]), self.alpha_hat[:-1]))

        self.loss_fn = torch.nn.MSELoss()



    def extract(self, input: torch.Tensor, index: torch.Tensor, shape):
        out     = torch.gather(input, 0, index.to(input.device))
        reshape = [shape[0]] + [1] * (len(shape) - 1)
        out     = out.reshape(*reshape).to(input.device)
        return out

    def sample_timesteps(self, n: int, device: str="cpu"):
        return torch.randint(low=0, high=self.noise_steps-1, size=(n,)).to(device)
    
    def q_samples(self, x: torch.Tensor, timesteps: torch.Tensor):
        device = x.device
        noise = torch.randn_like(x).to(device)
        return (self.extract(self.alpha_hat_sqrt, timesteps, x.shape) * x + self.extract(self.one_minus_alpha_hat_sqrt, timesteps, x.shape) * noise), noise
    
    def compute_train_loss(self, x: torch.Tensor, model: nn.Module):
        images, gt_masks = x
        device = images.device
        num_images = gt_masks.shape[0]
        timesteps = self.sample_timesteps(num_images, device)
        noisy_x, noise = self.q_samples(gt_masks, timesteps)

        model_output = model(torch.cat((noisy_x, images), dim=1), timesteps)
        """
        if self.val_learned:
            model_output, model_variance = torch.split(model_output, 1, dim=1)

        """
        return  self.loss_fn(model_output, noise)
    

    def inference(self, x: torch.Tensor, model: nn.Module):
        device = x.device
        noisy_mask = torch.randn_like(x).to(device)
        index = random.randint(0, x.shape[0] - 1)

        model.eval()
        with torch.no_grad():
            for timestep in tqdm(reversed(range(self.noise_steps))):
                timesteps = torch.ones(x.shape[0]).to(device) * timestep
                scaling = 1.0 / torch.sqrt(self.alpha[timestep])
                mean_scaling = (1 - self.alpha[timestep]) / (self.one_minus_alpha_hat_sqrt[timestep])
                variance_scaling = math.sqrt((1 - self.alpha_hat_prev[timestep]) / (1 - self.alpha_hat[timestep]) * self.beta[timestep])

                model_output = model(torch.cat((noisy_mask, x), dim=1), timesteps)

                
                # visualize_sampling_res(x[index], torch.where((torch.clamp(noisy_mask[index], -1, 1) + 1) / 2 > 0.5, 1.0, 0.0), x[index], batch_idx=timestep)

                noise = None
                if timestep == 0:
                    noise = torch.zeros_like(x).to(device)
                else: 
                    noise = torch.randn_like(x).to(device)

                noisy_mask = scaling * (noisy_mask - mean_scaling * model_output) + variance_scaling * noise

                if timestep % 100 == 0:
                    mask_step = load_res_to_wandb(x[25], pred_mask=noisy_mask[25] > 0.5, gt_mask=None, caption=str(timestep))
                    wandb.log({"step_images": mask_step})

        model.train()
        print(f"noisy mask: {noisy_mask[2]}")
        # noisy_mask = (torch.clamp(noisy_mask, -1, 1) + 1) / 2

        return noisy_mask
    

    def to(self, device: str):
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha_hat = self.alpha_hat.to(device)
        self.alpha_hat_sqrt = self.alpha_hat_sqrt.to(device)
        self.one_minus_alpha_hat_sqrt = self.one_minus_alpha_hat_sqrt.to(device)
        self.alpha_hat_prev = self.alpha_hat_prev.to(device)
        self.loss_fn = self.loss_fn.to(device)
    


class DDP(pl.LightningModule):
    def __init__(self, model: nn.Module, diffusion: Diffusion, metrics: dict, optim_args: dict):
        super().__init__()
        self.model = model
        self.diffusion = diffusion
        self.metrics = metrics
        self.optim_args = optim_args
        self.val_loss_fn = DiceLoss(include_background=False)
    

    def training_step(self, batch, batch_idx):
        loss = self.diffusion.compute_train_loss(batch, self.model)
        self.log("train_loss", loss)
        print(f"Training loss: {loss}")
        return loss

    def validation_step(self, batch, batch_idx):
        images, gt_masks = batch
        pred_masks = self.sample(images)

        mins = (torch.min(pred_masks.view(pred_masks.shape[0], -1), dim=1).values).view(-1, 1, 1).expand(pred_masks.size(1), pred_masks.size(2))
        maxs = torch.max(pred_masks.view(pred_masks.shape[0], -1), dim=1).values.view(-1, 1, 1).expand(pred_masks.size(1), pred_masks.size(2))

        pred_masks = ((pred_masks - mins) / (maxs-mins)) > 0.5
        
        loss = self.val_loss_fn(gt_masks, pred_masks).item()
        self.log("validation_dice_loss", loss)

        index = random.randint(0, pred_masks.shape[0] - 1)
        index = 25
        val_images = load_res_to_wandb(images[index], gt_masks[index], pred_masks[index], caption=f"BIdx_{batch_idx}_Idx_{index}")

        wandb.log({"val_examples": val_images})
        compute_and_log_metrics(self.metrics, pred_masks, gt_masks, "val", self.log)

        return loss
    
    def test_step(self, batch, batch_idx):
        images, gt_masks = batch
        pred_masks = self.sample(images)

        loss = DiceLoss(pred_masks, gt_masks)
        self.log("test_dice_loss", 1-loss)

        index = random.randint(0, pred_masks.shape[0] - 1)
        index = 25
        load_res_to_wandb(images[index], gt_masks[index], pred_masks[index] > 0.5, caption=f"Test_BIdx_{batch_idx}_Idx_{index}")
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), **self.optim_args)
        print(f"current state of optimizer: {optimizer.state_dict()}")
        return optimizer

    def sample(self, images: torch.Tensor):
        return self.diffusion.inference(images, self.model)
    
    def to(self, device: str):
        super().to(device)
        print(f"to function was called with device: {device}")
        self.diffusion.to(device)
        self.model.to(device)


        
class DDPM(pl.LightningModule):
    scheduler: DDPMScheduler
    model: nn.Module
    optimizer_config: OptimizerConfig
    metrics: dict
    
    def __init__(self, model: nn.Module, diffusion_config: DiffusionConfig, optimizer_config: OptimizerConfig, metrics: dict):
        super().__init__()
        
        self.scheduler = DDPMScheduler(num_train_timesteps=diffusion_config.noise_steps, beta_start=diffusion_config.beta_start, beta_end=diffusion_config.beta_end, beta_schedule=diffusion_config.scheduler_type)
        self.scheduler.set_timesteps(diffusion_config.noise_steps)
        self.model = model
        self.optimizer_config = optimizer_config
        self.metrics = metrics

        self.loss_fn = torch.nn.MSELoss()


    def training_step(self, batch, batch_idx):
        images, gt_masks = batch
        num_images = gt_masks.shape[0]
        noise = torch.randn_like(gt_masks, device=gt_masks.device)

        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (num_images,), device=gt_masks.device, dtype=torch.int64)
        noisy_image = self.scheduler.add_noise(gt_masks, noise, timesteps)

        pred_noise = self.model(torch.cat((noisy_image, images), dim=1), timesteps)

        loss = self.loss_fn(pred_noise, noise)
        self.log('train_loss', loss)

        return loss
    

    def validation_step(self, batch, batch_idx):
        images, gt_masks = batch
        num_images = images.shape[0]
        noisy_mask = torch.rand_like(images, device=images.device)

        index = random.randint(0, num_images-1)

        self.model.eval()
        with torch.no_grad():
            for t in tqdm(self.scheduler.timesteps):
                model_output = self.model(torch.cat((noisy_mask, images), dim=1), torch.ones(num_images, device=images.device) * t)

                noisy_mask = self.scheduler.step(model_output=model_output, timestep=t, sample=noisy_mask).prev_sample

                if t % 100 == 0:
                    mask_step = load_res_to_wandb(images[index], pred_mask=noisy_mask[index] > 0.5, gt_mask=None, caption=f"{batch_idx}_{index} at step {t}")
                    wandb.log({"step_images": mask_step})

        self.model.train()

        pred_masks = torch.where(noisy_mask.clamp(0, 1) > 0.5, 1.0, 0.0)
        val_images = load_res_to_wandb(images[index], gt_masks[index], pred_masks[index], caption=f"BIdx_{batch_idx}_Idx_{index}")

        wandb.log({"val_examples": val_images})
        compute_and_log_metrics(self.metrics, pred_masks, gt_masks, "val", self.log)

        return 0

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.optimizer_config.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    