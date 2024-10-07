
import pytorch_lightning as pl

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torch.nn as nn
from utils.hydra_config import DatasetConfig, DataloaderConfig, UNetConfig, DiffusionConfig, SegmentationConfig


def create_segmentor(project_name: str):
    if project_name == 'dmisse':
        from models.dmiise.dmiise_model import DmiiseSegmentation
        return DmiiseSegmentation()
    elif project_name == 'unet_seg':
        from models.unet_segmentation.unet_seg_model import UnetSegmentation
        return UnetSegmentation()
    else:
        raise NotImplementedError(f"Segmentation model {project_name} not implemented")


class BaseSegmentation:
    def __init__(self):
        super().__init__()

    def create_dataset(self, config: DatasetConfig) -> Dataset:
        print(f"Creating dataset {config.name}")
        if config.name == 'acdc':
            from dataset.acdc_dataset import ACDCDataset
            return ACDCDataset(folder_path=config.data_path, image_size=config.image_size)
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
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=config.shuffle)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=config.shuffle)
        
        return train_loader, val_loader, test_loader
    

    def create_dataset_dataloader(self, dataset_config: DatasetConfig, dataloader_config: DataloaderConfig) -> tuple[Dataset, DataLoader]:
        dataset = self.create_dataset(dataset_config)
        return dataset, self.create_dataloaders(dataloader_config, dataset)
    
    def create_segmentation_model(self, config: SegmentationConfig) -> pl.LightningModule:
        raise NotImplementedError("Model creation method not implemented")

    def create_model(self, config: UNetConfig):
        raise NotImplementedError("Model creation method not implemented")
    
    def create_diffusion(self, config: DiffusionConfig, model: nn.Module):
        raise NotImplementedError("Diffusion creation method not implemented")
    
    
    def train(self):
        raise NotImplementedError("Train method not implemented")
    
    
    def sample(self):
        raise NotImplementedError("Sample method not implemented")
        
    

    

            