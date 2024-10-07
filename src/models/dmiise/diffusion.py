import pytorch_lightning as pl
import torch
import math
import random

import torch.nn as nn

from tqdm import tqdm
from diffusers.schedulers import DDPMScheduler
from utils.hydra_config import DiffusionConfig
from utils.metrics import dice_loss
from utils.visualize import visualize_sampling_res


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
    img_size: tuple[int,int]
    device: str
    val_learned: bool

    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.noise_steps = config.noise_steps
        self.beta_start = config.beta_start
        self.beta_end = config.beta_end
        self.scheduler_type = config.scheduler_type
        self.img_size = config.img_size
        self.device = config.device
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
        out     = out.reshape(*reshape).to(self.device)
        return out

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    
    def q_samples(self, x: torch.Tensor, timesteps: torch.Tensor):
        noise = torch.randn_like(x)
        return (self.extract(self.alpha_hat_sqrt, timesteps, x.shape) * x + self.extract(self.one_minus_alpha_hat_sqrt, timesteps, x.shape) * noise), noise
    
    def compute_train_loss(self, x: torch.Tensor, model: nn.Module):
        images, gt_masks = x
        num_images = gt_masks.shape[0]
        timesteps = self.sample_timesteps(num_images)
        noisy_x, noise = self.q_samples(gt_masks, timesteps)

        model_output = model(torch.cat((noisy_x, images), dim=1), timesteps)
        """
        if self.val_learned:
            model_output, model_variance = torch.split(model_output, 1, dim=1)

        """
        return  self.loss_fn(model_output, noise)
    

    def inference(self, x: torch.Tensor, model: nn.Module):
        noisy_mask = torch.randn_like(x)

        model.eval()
        with torch.no_grad():
            for timestep in tqdm(reversed(range(self.noise_steps))):
                timesteps = torch.ones(x.shape[0]) * timestep
                scaling = 1.0 / torch.sqrt(self.alpha[timestep])
                mean_scaling = (1 - self.alpha[timestep]) / (self.one_minus_alpha_hat_sqrt[timestep])
                variance_scaling = math.sqrt((1 - self.alpha_hat_prev[timestep]) / (1 - self.alpha_hat[timestep]) * self.beta[timestep])

                model_output = model(torch.cat((noisy_mask, x), dim=1), timesteps)

                if timestep %  100 == 0:
                    print(f"scale: {scaling}")
                    print(f"mean scaling: {mean_scaling}")
                    print(f"variance scaling: {variance_scaling}")
                    print(f"model output max: {torch.max(model_output)}")
                    print(f"model output min: {torch.min(model_output)}")
                    print(f"model output has nan: {torch.isnan(model_output).any()}")
                    visualize_sampling_res(x[0], torch.where((torch.clamp(noisy_mask[0], -1, 1) + 1) / 2 > 0.5, 1.0, 0.0), x[0], batch_idx=timestep)

                noise = None
                if timestep == 0:
                    noise = torch.zeros_like(x)
                else: 
                    noise = torch.randn_like(x)

                noisy_mask = scaling * (noisy_mask - mean_scaling * model_output) + variance_scaling * noise
        
        model.train()

        print(f"Noisy mask max before clamp: {torch.max(noisy_mask)}")
        print(f"Noisy mask min before clamp: {torch.min(noisy_mask)}")

        noisy_mask = (torch.clamp(noisy_mask, -1, 1) + 1) / 2

        print(f"Noisy mask max after clamp: {torch.max(noisy_mask)}")
        print(f"Noisy mask min after clamp: {torch.min(noisy_mask)}")

        return noisy_mask
    


class DDP(pl.LightningModule):
    def __init__(self, model: nn.Module, diffusion: Diffusion):
        super().__init__()
        self.model = model
        self.diffusion = diffusion
    

    def training_step(self, batch, batch_idx):
        images, gt_masks = batch
        print(f"Executing training step with batch of size {images.shape}")
        loss = self.diffusion.compute_train_loss(batch, self.model)
        self.log("train_loss", loss)
        print(f"Training loss: {loss}")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.diffusion.compute_train_loss(batch, self.model)
        self.log("val_loss", loss)

        pred_masks = self.sample(batch)
        index = random.randint(0, pred_masks.shape[0] - 1)
        visualize_sampling_res(batch[0][index], pred_masks[index], batch[1][index], batch_idx=batch_idx)
    
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self.diffusion.compute_train_loss(batch, self.model)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)

    def sample(self, batch: torch.Tensor):
        images, gt_masks = batch
        pred_masks = self.diffusion.inference(images, self.model)
        
        pred_masks = torch.where(pred_masks > 0.5, 1, 0)
        d_loss = dice_loss(pred_masks, gt_masks)
        print(f"Dice loss: {d_loss}")

        return pred_masks
    
    
    


        






"""

class DiffusionModel(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, config: DiffusionConfig):
        super().__init__()
        self.model = model

        self.noise_steps = config.noise_steps
        self.beta_start = config.beta_start
        self.beta_end = config.beta_end
        self.img_size = config.img_size
        self.device = config.device

        self.beta = self.linear_scheduler().to(self.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.loss_fn = torch.nn.MSELoss()

    def linear_scheduler(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def add_noise_to_images(self, images: torch.Tensor, t: int):
        noise = torch.randn_like(images)
        return torch.sqrt(self.alpha_hat[t]).item() * images + noise * torch.sqrt(1 - self.alpha_hat[t]).item(), noise

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def training_step(self, batch: torch.Tensor):
        
        Args:
            batch: 4D tensor of shape (B, C, H, W)
            PS: Currently the model expects C=1
        

        assert(batch.shape[1] == 1, "Channel dimension should be 1")

        num_images = batch.shape[0]
        segmentation_shape = batch.shape[2:] # height and width of the image
        noise = torch.randn((num_images, *segmentation_shape), device=self.device)

        timesteps = self.sample_timesteps(num_images)
        noisy_images, noise = self.add_noise_to_images(batch, timesteps)

        concatentated_images = torch.cat((noisy_images, batch), dim=1)
        predicted_noise = self.model(concatentated_images, timesteps)
        loss = self.loss_fn(predicted_noise, noise)

        return loss
    

    def validation_step(self,batch,batch_idx):
        pass
        
    
    def p_mean_variance(self, x, t, clip_denoised)

    def sample(self, batch: torch.Tensor):
        
        Args:
            batch: 4D tensor of shape (B, C, H, W)
            PS: Currently the model expects C=1
        

        assert(batch.shape[1] == 1, "Channel dimension should be 1")

        self.model.eval()
        num_images = batch.shape[0]
        segmentation_shape = batch.shape[2:]

        with torch.no_grad():

            sampled_images = torch.randn((num_images, *segmentation_shape), device=self.device)
            
            for t in range(self.noise_steps - 1, 0):
                timesteps = (torch.ones(num_images) * t).long().to(self.device)
                concatenated_images = torch.cat((sampled_images, batch), dim=1)
                predicted_noise = self.model(concatenated_images, timesteps)

                if t > 1:
                    noise = torch.randn_like(sampled_images)
                else:
                    noise = torch.zeros_like(sampled_images)

                alpha_t_sqrt = math.sqrt(self.alpha_hat[t].item())
                beta_t = self.beta[t].item()
                alpha_head_t = math.sqrt(1 - self.alpha_hat[t].item())

                images = 1.0 / alpha_t_sqrt (sampled_images - (beta_t / alpha_head_t) * predicted_noise) + math.sqrt(beta_t) * noise

        self.model.train()

        images = torch.clamp(images, -1, 1) + 1 / 2
        images = (images * 255).type(torch.uint8)
 
        return images

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


"""