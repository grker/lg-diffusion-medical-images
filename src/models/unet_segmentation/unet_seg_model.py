
import pytorch_lightning as pl
import random
import torch
import torch.nn as nn
import wandb

from monai.losses import DiceLoss, DiceCELoss
from models.base_segmentation import BaseSegmentation
from utils.hydra_config import SegmentationConfig, UNetConfig
# from utils.metrics import dice_loss
from utils.visualize import visualize_sampling_res, load_res_to_wandb
from monai.losses import DiceLoss
from utils.metrics import compute_and_log_metrics

class UnetSegmentation(BaseSegmentation):
    def __init__(self, config: SegmentationConfig):
        super().__init__(config)

    def create_segmentation_model(self):
        model = self.create_model()
        metrics = self.create_metrics_fn()
        return UnetSegmentationModel(model, metrics)
 
    def create_model(self):
        from monai.networks.nets import BasicUNet
        
        return BasicUNet(spatial_dims=2, features=[32, 64, 128, 256, 512, 32], dropout=0.2, in_channels=1, out_channels=1)
    
    def initialize(self):
        dataset, dataloader = self.create_dataset_dataloader()
        train_loader, val_loader, test_loader = dataloader
        
        return self.create_segmentation_model(), train_loader, val_loader, test_loader



class UnetSegmentationModel(pl.LightningModule):
    def __init__(self, model: nn.Module, metrics: dict):
        super(UnetSegmentationModel, self).__init__()
        self.model = model
        self.metrics = metrics
        self.loss_fn = DiceLoss(include_background=False)
        

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        images, masks = batch
        pred_masks = self.model(images)

        loss = self.loss_fn(pred_masks, masks)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, gt_masks = batch
        pred_masks = self.model(images)
        pred_masks = pred_masks > 0.5

        loss = self.loss_fn(pred_masks, gt_masks)
        self.log('val_loss', loss)

        index = random.randint(0, pred_masks.shape[0] - 1)
        # val_images = load_res_to_wandb(images[index], gt_masks[index], pred_masks[index], caption=f"BIdx_{batch_idx}_Idx_{index}")
        # wandb.log({"val_examples": val_images})

        compute_and_log_metrics(self.metrics, pred_masks, gt_masks, "val", self.log)
        return loss

    def test_step(self, batch, batch_idx):
        images, gt_masks = batch
        pred_masks = self.model(images)
        pred_masks = pred_masks > 0.5

        loss = self.loss_fn(pred_masks, gt_masks)
        self.log('test_loss', loss)

        index = random.randint(0, pred_masks.shape[0] - 1)
        # val_images = load_res_to_wandb(images[index], gt_masks[index], pred_masks[index], caption=f"BIdx_{batch_idx}_Idx_{index}")
        # wandb.log({"test_examples": val_images})

        compute_and_log_metrics(self.metrics, pred_masks, gt_masks, "test", self.log)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)
    