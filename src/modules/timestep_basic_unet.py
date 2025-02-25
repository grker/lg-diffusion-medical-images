"""
A wrapper around MONAI's BasicUNet that accepts both image and timestep inputs.
This modified architecture extends the standard BasicUNet by incorporating
temporal information through timestep embeddings, making it suitable for diffusion models
"""

import logging

import torch
import torch.nn as nn
from monai.networks.nets import BasicUNet

from utils.hydra_config import BasicUNetConfig

from .utils import linear, timestep_embedding

logger = logging.getLogger(__name__)


class TimestepsBasicUNet(BasicUNet):
    spatial_dims: int
    in_channels: int
    out_channels: int
    features: list[int]
    dropout: float
    emb_channels: int
    time_start: int
    use_scale_shift_norm: bool

    def __init__(self, config: BasicUNetConfig):
        self.layers = 4  # number of downsampling layers and upsampling layers --> set to 4 as BasicUNet of MONAI has 4 layers (cannot be changed)

        if config.features is not None and len(config.features) == self.layers + 2:
            self.features = config.features
        else:
            logger.warning(
                "Features not provided or incorrect length. Using default features."
            )
            self.features = [32, 32, 64, 128, 256, 32]

        self.spatial_dims = config.spatial_dims
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.dropout = config.dropout
        self.emb_channels = config.emb_channels
        self.time_start = config.time_start

        bUnet_config = {
            "spatial_dims": config.spatial_dims,
            "in_channels": config.in_channels,
            "out_channels": config.out_channels,
            "features": config.features,
            "dropout": config.dropout,
        }

        super().__init__(**bUnet_config)  # builds the underlying BasicUNet

        # Timestep embedding
        self.time_embed = nn.Sequential(
            linear(self.time_start, self.emb_channels),
            nn.SiLU(),
            linear(self.emb_channels, self.emb_channels),
        )

        # Timestep layers for downsampling
        self.time_layers_down = nn.ModuleList()
        for i in range(self.layers):
            self.time_layers_down.append(
                nn.Sequential(
                    nn.SiLU(),
                    linear(self.emb_channels, self.features[i + 1]),
                )
            )

        # Timestep layers for upsampling
        self.time_layers_up = nn.ModuleList()
        for i in range(self.layers - 1):
            self.time_layers_up.append(
                nn.Sequential(
                    nn.SiLU(),
                    linear(self.emb_channels, self.features[-(i + 3)]),
                )
            )
        self.time_layers_up.append(
            nn.Sequential(
                nn.SiLU(),
                linear(self.emb_channels, self.features[-1]),
            )
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor):
        """
        Forward pass of the TimestepsBasicUNet.

        :param x: (B, C, H, W) tensor of input images
        :param timesteps: (B,) tensor of timestep values
        :return: (B, C, H, W) tensor of output images
        """
        time_emb = timestep_embedding(timesteps, self.time_start)
        time_emb = self.time_embed(time_emb)
        x = self.conv_0(x)
        res = [x]

        # Downsampling
        for i in range(self.layers):
            x = getattr(self, f"down_{i + 1}")(x)
            emb_out = self.time_layers_down[i](time_emb).type(x.dtype)
            while len(emb_out.shape) < len(x.shape):
                emb_out = emb_out[..., None]
            x = x + emb_out

            if i != self.layers - 1:
                res.append(x)

        assert len(res) == self.layers

        # Upsampling
        for i in range(self.layers):
            skip_element = res[-(i + 1)]
            x = getattr(self, f"upcat_{self.layers - i}")(x, skip_element)
            emb_out = self.time_layers_up[i](time_emb).type(x.dtype)
            while len(emb_out.shape) < len(x.shape):
                emb_out = emb_out[..., None]
            x = x + emb_out

        logits = self.final_conv(x)
        return logits
