from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torch.nn as nn
from utils.hydra_config import DatasetConfig, DataloaderConfig, SegmentationConfig, UNetConfig, DiffusionConfig
from models.base_segmentation import BaseSegmentation


class DmiiseSegmentation(BaseSegmentation):
    def __init__(self):
        super().__init__()

    def create_segmentation_model(self, config: SegmentationConfig):
        model = self.create_model(config.model)
        return self.create_diffusion(config.diffusion, model)

    def create_model(self, config: UNetConfig):
        from modules.unet import UNetModel
        
        return UNetModel(config)
    
    def create_diffusion(self, config: DiffusionConfig, model: nn.Module):
        from models.dmiise.diffusion import Diffusion, DDP
        
        diffusion = Diffusion(config)
        return DDP(model, diffusion)
    
    
    
    
            