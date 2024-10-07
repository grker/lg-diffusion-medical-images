
import pytorch_lightning as pl
import random
import torch
import torch.nn as nn

from monai.losses import DiceLoss, DiceCELoss
from models.base_segmentation import BaseSegmentation
from utils.hydra_config import SegmentationConfig, UNetConfig
from utils.metrics import dice_loss
from utils.visualize import visualize_sampling_res

class UnetSegmentation(BaseSegmentation):
    def __init__(self):
        super().__init__()

    def create_segmentation_model(self, config: SegmentationConfig):
        return UnetSegmentationModel(self.create_model(config.model))
 
    def create_model(self, config: UNetConfig):
        from monai.networks.nets import BasicUNet
        
        return BasicUNet(spatial_dims=2, features=[32, 64, 128, 256, 512, 32], dropout=0.2, in_channels=1, out_channels=1)


class UnetSegmentationModel(pl.LightningModule):
    def __init__(self, model: nn.Module):
        super(UnetSegmentationModel, self).__init__()
        self.model = model
        self.loss_fn = DiceLoss(sigmoid=True)
        self.val_step = 0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        images, masks = batch
        pred_masks = self.model(images)
        
        loss = self.loss_fn(pred_masks, masks)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch):
        images, masks = batch
        pred_masks = self.model(images)
        loss = self.loss_fn(pred_masks, masks)
        self.log('val_loss', loss)

        index = random.randint(0, pred_masks.shape[0] - 1)
        visualize_sampling_res(images[index], pred_masks[index] > 0.5, masks[index], name='unet_seg', batch_idx=self.val_step)
        self.val_step += 1
        return loss

    def test_step(self, batch):
        images, masks = batch
        pred_masks = self.model(images)
        loss = self.loss_fn(pred_masks, masks)
        self.log('val_loss', loss)

        index = random.randint(0, pred_masks.shape[0] - 1)
        visualize_sampling_res(images[index], pred_masks[index], masks[index], name='unet_seg', batch_idx=self.val_step)
        self.val_step += 1
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)
    