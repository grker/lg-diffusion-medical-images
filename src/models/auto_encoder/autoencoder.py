import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import open_dict

from metrics import MetricsHandler  # , MetricsInput
from models.base_segmentation import BaseSegmentation
from utils.helper import unpack_batch
from utils.hydra_config import SegmentationConfig
from utils.mask_transformer import BaseMaskMapping

# from utils.visualize import visualize_segmentation


class AutoEncoder(BaseSegmentation):
    def __init__(self, config: SegmentationConfig):
        super().__init__(config)

    def initialize(self, test: bool = False) -> pl.LightningModule:
        dataset, dataloader = self.create_dataset_dataloader()
        train_loader, val_loader, test_loader = dataloader

        if hasattr(dataset, "mask_transformer"):
            mask_transformer = getattr(dataset, "mask_transformer")
            channels = mask_transformer.get_num_train_channels()
            num_classes = mask_transformer.get_num_classes()
        else:
            raise AttributeError("No Mask Transformer is specified!")

        with open_dict(self.config.model):
            self.config.model.in_channels = channels
            self.config.model.out_channels = channels

        model_args = self.create_seg_model_args(mask_transformer, num_classes)

        if test:
            return self.lightning_module(), test_loader, model_args
        else:
            return (
                self.lightning_module()(**model_args),
                train_loader,
                val_loader,
                test_loader,
            )

    def create_seg_model_args(
        self, mask_transformer: BaseMaskMapping, num_classes: int
    ) -> dict:
        return {
            "mask_transformer": mask_transformer,
            "model": self.create_model(),
            "metrics": self.create_metric_handler(),
            "loss": self.create_loss(),
        }

    def create_model(self):
        return EncoderDecoderModel(self.config.model.in_channels)

    def lightning_module(self):
        return AutoEncoderModel


class AutoEncoderModel(pl.LightningModule):
    model: nn.Module
    metrics: MetricsHandler
    loss_fn: torch.nn.Module
    mask_transformer: BaseMaskMapping

    def __init__(
        self,
        model: nn.Module,
        metrics: MetricsHandler,
        loss: torch.nn.Module,
        mask_transformer: BaseMaskMapping,
    ):
        super().__init__()
        self.model = model
        self.metric_handler = metrics
        self.loss_fn = loss
        self.mask_transformer = mask_transformer
        self.num_classes = mask_transformer.get_num_classes()

    def training_step(self, batch, batch_idx):
        images, masks, training_mask = unpack_batch(batch)
        pred_masks = self.model(training_mask)

        loss = self.loss_fn(pred_masks, training_mask)
        self.log("train_loss", loss)
        return loss

    def val_test_step(self, batch, batch_idx, phase):
        images, masks, training_mask = unpack_batch(batch)
        pred_masks = self.model(training_mask)

        loss = self.loss_fn(pred_masks, training_mask)
        self.log(f"{phase}_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.val_test_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.val_test_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.00001)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class EncoderDecoderModel(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            ConvBlock(channels, 32),
            ConvBlock(32, 64),
            ConvBlock(
                64, channels
            ),  # Output shape: [B, channels, H, W] with real values
        )

        # Decoder
        self.decoder = nn.Sequential(
            ConvBlock(channels, 64),
            ConvBlock(64, 32),
            nn.Conv2d(32, channels, kernel_size=3, padding=1),
            nn.Sigmoid(),  # Reconstruct binary mask (range 0-1)
        )

    def say_hello(self):
        print("hello from model")

    def forward(self, x):
        latent = self.encoder(x)  # [B, C, H, W] real-valued
        reconstructed = self.decoder(latent)  # [B, C, H, W] in [0, 1]
        return reconstructed

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)
