from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torch.nn as nn
import pytorch_lightning as pl
from utils.hydra_config import DatasetConfig, DataloaderConfig, SegmentationConfig, UNetConfig, DiffusionConfig
from models.base_segmentation import BaseSegmentation
from omegaconf import open_dict


class DmiiseSegmentation(BaseSegmentation):
    def __init__(self, config: SegmentationConfig):
        super().__init__(config)

    def create_segmentation_model(self) -> pl.LightningModule:
        model = self.create_model()
        return self.create_diffusion(model, self.device)

    def create_model(self):
        from modules.unet import UNetModel
        return UNetModel(self.config.model)
    
    def create_diffusion(self, model: nn.Module, device: str="cpu"):
        from models.dmiise.diffusion import DDPM
        metrics = self.create_metrics_fn()
        return DDPM(model, self.config.diffusion, self.config.optimizer, metrics)
    
    def initialize(self) -> pl.LightningModule:
        dataset, dataloader = self.create_dataset_dataloader()
        train_loader, val_loader, test_loader = dataloader

        image_height, image_width = dataset.get_image_height(), dataset.get_image_width()
        with open_dict(self.config.model):
            self.config.model.image_size = (image_height, image_width)

        return self.create_segmentation_model(), train_loader, val_loader, test_loader
    
    
    
    
            