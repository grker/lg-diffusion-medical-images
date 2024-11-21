
from typing import Callable, Union
import pytorch_lightning as pl
import torch
import functools

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from utils.metrics import generate_metrics_fn
from utils.hydra_config import DatasetConfig, DataloaderConfig, UNetConfig, DiffusionConfig, SegmentationConfig, MetricsConfig
from utils.loss import CustomLoss


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
        """
        Dataset factory method.
        Creates a dataset from a config. 
        Throws an error if the dataset has not been implemented yet.
        
        :param config: DatasetConfig
        :return: Dataset
        """
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
        """
        Creates a dataloader from a config and a dataset.
        
        :param config: DataloaderConfig
        :param dataset: Dataset
        :return: DataLoader
        """
        
        print(f"Creating dataloaders")
        train_size = int(config.train_ratio * len(dataset))
        val_size = int(config.validation_ratio * len(dataset))
        test_size = len(dataset) - train_size - val_size

        assert train_size + val_size + test_size == len(dataset)

        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=config.shuffle)
        val_loader = DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.val_batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    

    def create_dataset_dataloader(self) -> tuple[Dataset, DataLoader]:
        """
        Creates a dataset and a dataloader from a config.
        
        :return: tuple[Dataset, DataLoader]
        """
        dataset = self.create_dataset(self.config.dataset)
        return dataset, self.create_dataloaders(self.config.dataloader, dataset)
    
    def create_metrics_fn(self, num_classes: int=2) -> dict[str, Callable]:
        """
        Creates a metrics function from a config.
        
        :param num_classes: int
        :return: dict[str, Callable]
        """
        metrics = generate_metrics_fn(self.config.metrics, num_classes)
        return metrics
    
    def lightning_module(self) -> pl.LightningModule:
        """
        Returns the class of the underlying (pytorch-lightning) segmentation model.
        
        :return: pl.LightningModule
        """
        raise NotImplementedError("lightning_module method not implemented")
    
    def create_loss(self) -> CustomLoss:
        """
        Creates a loss function from a config.
        
        :return: CustomLoss (inherits from torch.nn.Module)
        """
        return CustomLoss(self.config.loss)
    
    def initialize(self, test: bool=False) -> Union[tuple[pl.LightningModule, DataLoader, DataLoader, DataLoader], tuple[functools.partial, DataLoader, dict]]:
        """
        Sets up the whole Segmentation. Creates the dataset, dataloaders, loss and metrics function as well as the LightningModule.
        If test is False, the function returns an instance of the underlying LightningModule, a train_loader, a val_loader and a test_loader.
        If test is True, the function returns a partial function that can be used to initialize the LightningModule, a test_loader and a dict of model arguments.
        The model arguments are the arguments expected by the LightningModule's __init__ method to correctly create an instance of the LightningModule.
        :param test: bool
        :return: Depends on test:
            test=False: tuple[pl.LightningModule, DataLoader, DataLoader, DataLoader]
            test=True: tuple[functools.partial, DataLoader, dict]
        """
        raise NotImplementedError("Initialize methode not implemented")

    
    def train(self):
        raise NotImplementedError("Train method not implemented")
    
    
    def sample(self):
        raise NotImplementedError("Sample method not implemented")
        
    

    

            