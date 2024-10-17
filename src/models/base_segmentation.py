
import pytorch_lightning as pl

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from utils.metrics import generate_metrics_fn
import torch
import torch.nn as nn
from utils.hydra_config import DatasetConfig, DataloaderConfig, UNetConfig, DiffusionConfig, SegmentationConfig, MetricsConfig


def create_segmentor(config: SegmentationConfig):
    project_name = config.project_name
    if project_name == 'dmiise':
        from models.dmiise.dmiise_model import DmiiseSegmentation
        return DmiiseSegmentation(config)
    elif project_name == 'unet_seg':
        from models.unet_segmentation.unet_seg_model import UnetSegmentation
        return UnetSegmentation(config)
    else:
        raise NotImplementedError(f"Segmentation model {project_name} not implemented")


class BaseSegmentation:
    def __init__(self, config: SegmentationConfig):
        super().__init__()
        self.config = config
        self.device = self.set_device()

    def set_device(self) -> str:
        if self.config.trainer.accelerator is None:
            return "cpu"
        if self.config.trainer.accelerator == "gpu":
            if torch.cuda.is_available():
                return f"cuda:{torch.cuda.current_device()}"
            else:
                return "cpu"

        return self.config.trainer.accelerator
        

    def create_dataset(self, config: DatasetConfig) -> Dataset:
        print(f"Creating dataset {config.name}")
        if config.name == 'acdc':
            from dataset.acdc_dataset import ACDCDataset
            return ACDCDataset(config)
        elif config.name == 'm2nist':
            from dataset.mnist import M2NISTDataset
            return M2NISTDataset(config)
        else:
            raise NotImplementedError(f'Dataset {config.name} not implemented')
        

    def create_dataloaders(self, config: DataloaderConfig, dataset: Dataset) -> DataLoader:
        print(f"Creating dataloaders")
        train_size = int(config.train_ratio * len(dataset))
        val_size = int(config.validation_ratio * len(dataset))
        test_size = len(dataset) - train_size - val_size

        assert train_size + val_size + test_size == len(dataset)

        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=config.shuffle)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    

    def create_dataset_dataloader(self) -> tuple[Dataset, DataLoader]:
        dataset = self.create_dataset(self.config.dataset)
        return dataset, self.create_dataloaders(self.config.dataloader, dataset)
    
    def create_metrics_fn(self):
        metrics = generate_metrics_fn(self.config.metrics)
        return metrics
    
    # def create_segmentation_model(self) -> pl.LightningModule:
    #     raise NotImplementedError("Model creation method not implemented")

    # def create_model(self, config: UNetConfig):
    #     raise NotImplementedError("Model creation method not implemented")
    
    # def create_diffusion(self, config: DiffusionConfig, model: nn.Module):
    #     raise NotImplementedError("Diffusion creation method not implemented")
    
    def initialize(self) -> pl.LightningModule:
        raise NotImplementedError("Initialize methode not implemented")


    
    
    
    def train(self):
        raise NotImplementedError("Train method not implemented")
    
    
    def sample(self):
        raise NotImplementedError("Sample method not implemented")
        
    

    

            