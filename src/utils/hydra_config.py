from dataclasses import dataclass
from typing import Union



class ResNetConfig:
    layers: list[int]
    starting_channels: int