
import pytorch_lightning as pl
import torch
import torch.nn as nn

from models.base_segmentation import BaseSegmentation
from utils.hydra_config import SegmentationConfig, UNetConfig
from utils.visualize import visualize_segmentation
from utils.metrics import compute_and_log_metrics
from utils.helper import unpack_batch
from omegaconf import open_dict
from utils.mask_transformer import BaseMaskMapping



class UnetSegmentation(BaseSegmentation):
    def __init__(self, config: SegmentationConfig):
        super().__init__(config)

    def create_segmentation_model(self, mask_transformer: BaseMaskMapping, num_classes: int) -> pl.LightningModule:
        model = self.create_model()
        metrics = self.create_metrics_fn(num_classes)
        loss = self.create_loss()
        return UnetSegmentationModel(model, metrics, mask_transformer, loss)
 
    def create_model(self):
        from monai.networks.nets import BasicUNet
        
        return BasicUNet(**self.config.model)
    
    def initialize(self):
        dataset, dataloader = self.create_dataset_dataloader()
        train_loader, val_loader, test_loader = dataloader

        if hasattr(dataset, 'mask_transformer'):
            mask_transformer = getattr(dataset, 'mask_transformer')
            output_channels = mask_transformer.get_num_train_channels()
            num_classes = mask_transformer.get_num_classes()
        else:
            raise AttributeError("No Mask Transformer is specified!")
        
        with open_dict(self.config.model):
            self.config.model.out_channels = output_channels
        
        return self.create_segmentation_model(mask_transformer, num_classes), train_loader, val_loader, test_loader



class UnetSegmentationModel(pl.LightningModule):
    model: nn.Module
    metrics: dict
    loss_fn: torch.nn.Module
    mask_transformer: BaseMaskMapping
    num_classes: int

    def __init__(self, model: nn.Module, metrics: dict, mask_transformer: BaseMaskMapping, loss_fn: torch.nn.Module):
        super(UnetSegmentationModel, self).__init__()
        self.model = model
        self.metrics = metrics
        self.loss_fn = loss_fn
        self.mask_transformer = mask_transformer
        self.num_classes = mask_transformer.get_num_classes()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks, training_mask = unpack_batch(batch)
        pred_masks = self.model(images)

        print(f"pred_masks: {pred_masks.shape}")
        print(f"masks: {masks.shape}")

        loss = self.loss_fn(pred_masks, training_mask)
        self.log('train_loss', loss)
        return loss
    
    def val_test_step(self, batch, batch_idx, phase):
        images, gt_masks, _ = unpack_batch(batch)
        pred_masks = self.model(images)

        logits = self.mask_transformer.get_logits(pred_masks)
        seg_mask, one_hot_seg_mask = self.mask_transformer.get_segmentation(logits)

        compute_and_log_metrics(self.metrics, seg_mask, gt_masks, phase, self.log)
        visualize_segmentation(images, gt_masks, seg_mask, pred_masks, phase, self.mask_transformer.gt_mapping_for_visualization(), batch_idx, self.num_classes)

        return 0

    def validation_step(self, batch, batch_idx):
        return self.val_test_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.val_test_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)
    