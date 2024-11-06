import monai.transforms
import torch

from utils.hydra_config import PreprocessorConfig


def generate_preprocessor(config: PreprocessorConfig):
    preprocessor = {}
    apply_to_mask = {}

    for transform_name, kwargs in config.transform_config.items():
        apply_to_mask = kwargs.apply_to_mask if kwargs.apply_to_mask is not None else False
        args = kwargs.args if kwargs.args is not None else {}

        if hasattr(monai.transforms, transform_name):
            preprocessor[transform_name] = getattr(monai.transforms, transform_name)(**kwargs)
            apply_to_mask[transform_name] = apply_to_mask
        else:
            raise ValueError(f"Unknown transform {transform_name}. The preprocessor so far only supports transforms from monai.transforms")

    return preprocessor, apply_to_mask


def apply_preprocessor(preprocessor: dict, apply_to_mask: dict, images: torch.Tensor, masks: torch.Tensor):
    for transform_name, transform in preprocessor.items():
        if apply_to_mask[transform_name]:
            masks = transform(masks)
        images = transform(images)

    return images, masks




