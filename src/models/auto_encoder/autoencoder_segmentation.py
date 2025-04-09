import os

import pytorch_lightning as pl
from omegaconf import OmegaConf, open_dict

import wandb
from models.auto_encoder.autoencoder import AutoEncoder
from models.dmiise.diffusion import DDPM_Autoencoder
from models.dmiise.dmiise_model import DmiiseSegmentation
from utils.hydra_config import SegmentationConfig
from utils.mask_transformer import BaseMaskMapping
from wandb import Api


class AutoencoderSegmentation(DmiiseSegmentation):
    def __init__(self, config: SegmentationConfig, loss_guided: bool = False):
        super().__init__(config, loss_guided)

    def initialize(self, test: bool = False) -> pl.LightningModule:
        diffusion_config = self.config.diffusion

        autoencoder = None
        if "autoencoder" in diffusion_config.keys():
            autoencoder = self.load_autoencoder(diffusion_config.autoencoder)
            autoencoder.eval()
        else:
            raise ValueError("autoencoder_run_id not found in diffusion config")

        mask_transformer_ae = autoencoder.mask_transformer

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
            self.check_compatibility(mask_transformer_ae, mask_transformer)
            output_channels = mask_transformer.get_num_train_channels()
            num_classes = mask_transformer.get_num_classes()
        else:
            raise AttributeError("No Mask Transformer is specified!")

        with open_dict(self.config.model):
            self.config.model.out_channels = output_channels
            self.config.model.in_channels = output_channels + 1

            print(f"model in_channels: {self.config.model.in_channels}")
            print(f"model out_channels: {self.config.model.out_channels}")

        model_args = self.create_seg_model_args(mask_transformer_ae, num_classes)
        autoencoder.model.say_hello()

        model_args["autoencoder"] = autoencoder.model

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

    def check_compatibility(
        self, mask_transformer_ae: BaseMaskMapping, mask_transformer: BaseMaskMapping
    ):
        assert (
            mask_transformer_ae.get_num_classes() == mask_transformer.get_num_classes()
        ), "Autoencoder and segmentation model have different number of classes"

        assert (
            mask_transformer_ae.get_num_train_channels()
            == mask_transformer.get_num_train_channels()
        ), "Autoencoder and segmentation model have different number of train channels"

        assert mask_transformer_ae.train_switch == mask_transformer.train_switch, (
            "Autoencoder and segmentation model have different train switches"
        )

    def load_autoencoder(self, autoencoder_config: dict):
        print(f"autoencoder_config: {autoencoder_config}")
        api = Api()
        autoencoder_run = api.run(
            f"{autoencoder_config.wandb_username}/{autoencoder_config.wandb_project}/{autoencoder_config.run_id}"
        )
        old_config = OmegaConf.create(autoencoder_run.config)

        autoencoder_model_class, _, model_args = AutoEncoder(old_config).initialize(
            test=True
        )

        model_artifact = wandb.use_artifact(
            f"model-{autoencoder_config.run_id}:best", type="model"
        )
        model_artifact_dir = model_artifact.download()
        checkpoint_path = os.path.join(model_artifact_dir, "model.ckpt")
        autoencoder_model = autoencoder_model_class.load_from_checkpoint(
            checkpoint_path=checkpoint_path, **model_args
        )

        return autoencoder_model

    def lightning_module(self):
        return DDPM_Autoencoder
