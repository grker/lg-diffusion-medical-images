from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torch.nn as nn
import pytorch_lightning as pl
from utils.hydra_config import DatasetConfig, DataloaderConfig, SegmentationConfig, UNetConfig, DiffusionConfig
from models.base_segmentation import BaseSegmentation
from omegaconf import open_dict
from utils.mask_transformer import BaseMaskMapping


class DmiiseSegmentation(BaseSegmentation):
    def __init__(self, config: SegmentationConfig):
        super().__init__(config)

    def create_segmentation_model(self, mask_transformer: BaseMaskMapping, num_classes: int) -> pl.LightningModule:
        from modules.unet import UNetModel
        from models.dmiise.diffusion import DDPM

        model = UNetModel(self.config.model)
        metrics = self.create_metrics_fn(num_classes)
        loss = self.create_loss()
        return DDPM(model, self.config.diffusion, self.config.optimizer, metrics, mask_transformer, loss)
    
    
    def initialize(self) -> pl.LightningModule:
        dataset, dataloader = self.create_dataset_dataloader()
        train_loader, val_loader, test_loader = dataloader

        image_height, image_width = dataset.get_image_height(), dataset.get_image_width()
        with open_dict(self.config.model):
            self.config.model.image_size = (image_height, image_width)
        

        if hasattr(dataset, 'mask_transformer'):
            mask_transformer = getattr(dataset, 'mask_transformer')
            output_channels = mask_transformer.get_num_train_channels()
            num_classes = mask_transformer.get_num_classes()
        else:
            raise AttributeError("No Mask Transformer is specified!")
        
        with open_dict(self.config.model):
            self.config.model.out_channels = output_channels
            self.config.model.in_channels = output_channels+1

            print(f"model in_channels: {self.config.model.in_channels}")
            print(f"model out_channels: {self.config.model.out_channels}")
        
        return self.create_segmentation_model(mask_transformer, num_classes), train_loader, val_loader, test_loader
    