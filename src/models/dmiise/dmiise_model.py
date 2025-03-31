import pytorch_lightning as pl
from omegaconf import open_dict

from models.base_segmentation import BaseSegmentation
from models.dmiise.diffusion import (
    DDPM,
    DDPM_DPS_Regularized,
)
from models.dmiise.score_based_diffusion import ScoreBasedDiffusion
from utils.hydra_config import SegmentationConfig
from utils.mask_transformer import BaseMaskMapping


class DmiiseSegmentation(BaseSegmentation):
    def __init__(self, config: SegmentationConfig, loss_guided: bool = False):
        super().__init__(config)
        self.loss_guided = loss_guided

    def create_segmentation_model(
        self, mask_transformer: BaseMaskMapping, num_classes: int
    ) -> pl.LightningModule:
        from modules.timestep_basic_unet import TimestepsBasicUNet
        from modules.unet import UNetModel

        if self.config.model.name == "unet_dmisse":
            model = UNetModel(self.config.model)
        elif self.config.model.name == "basic_unet":
            model = TimestepsBasicUNet(self.config.model)
        else:
            raise ValueError(f"Model {self.config.model.name} not found!")

        metrics = self.create_metrics_fn(num_classes)
        loss = self.create_loss()
        if self.loss_guided:
            if hasattr(self.config.diffusion, "loss_guidance"):
                return DDPM_DPS_Regularized(
                    model,
                    self.config.diffusion,
                    self.config.optimizer,
                    metrics,
                    mask_transformer,
                    loss,
                )
            else:
                raise ValueError("No loss_guidance config has been specified!")

        return DDPM(
            model,
            self.config.diffusion,
            self.config.optimizer,
            metrics,
            mask_transformer,
            loss,
        )

    def create_seg_model_args(
        self, mask_transformer: BaseMaskMapping, num_classes: int
    ):
        from modules.timestep_basic_unet import TimestepsBasicUNet
        from modules.unet import UNetModel

        model_args = {
            "mask_transformer": mask_transformer,
        }

        if self.config.model.name == "unet_dmisse":
            model = UNetModel(self.config.model)
        elif self.config.model.name == "basic_unet":
            model = TimestepsBasicUNet(self.config.model)
        else:
            raise ValueError(f"Model {self.config.model.name} not found!")

        model_args["model"] = model
        model_args["diffusion_config"] = self.config.diffusion
        model_args["optimizer_config"] = self.config.optimizer
        model_args["metrics"] = self.create_metric_handler()
        model_args["loss"] = self.create_loss()

        return model_args

    def initialize(self, test: bool = False) -> pl.LightningModule:
        dataset, dataloader = self.create_dataset_dataloader()
        train_loader, val_loader, test_loader = dataloader

        image_height, image_width = (
            dataset.get_image_height(),
            dataset.get_image_width(),
        )
        with open_dict(self.config.model):
            self.config.model.image_size = (image_height, image_width)

        if hasattr(dataset, "mask_transformer"):
            mask_transformer = getattr(dataset, "mask_transformer")
            output_channels = mask_transformer.get_num_train_channels()
            num_classes = mask_transformer.get_num_classes()
        else:
            raise AttributeError("No Mask Transformer is specified!")

        with open_dict(self.config.model):
            self.config.model.out_channels = output_channels
            self.config.model.in_channels = output_channels + 1

            print(f"model in_channels: {self.config.model.in_channels}")
            print(f"model out_channels: {self.config.model.out_channels}")

        model_args = self.create_seg_model_args(mask_transformer, num_classes)

        if test:
            return (
                self.lightning_module(),
                test_loader,
                model_args if model_args is not None else {},
            )
        else:
            return (
                self.lightning_module()(**model_args),
                train_loader,
                val_loader,
                test_loader,
            )

    def lightning_module(self):
        if self.loss_guided:
            return DDPM_DPS_Regularized
        diffusion_config = self.config.diffusion

        if "diffusion_type" in diffusion_config.keys():
            if diffusion_config["diffusion_type"] == "score_based":
                return ScoreBasedDiffusion
            elif diffusion_config["diffusion_type"] == "ddpm":
                return DDPM
        else:
            return DDPM  # this is to support older configs which do not have the diffusion_type attribute
