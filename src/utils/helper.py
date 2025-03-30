from copy import deepcopy

import omegaconf
import torch
from omegaconf import OmegaConf


class EMA:
    def __init__(self, alpha: float = 0.9):
        self.alpha = alpha
        self.model_state_dict = None

    def update(self, model: torch.nn.Module):
        new_state_dict = deepcopy(model.state_dict())
        if self.model_state_dict is None:
            self.model_state_dict = new_state_dict
        else:
            for key in self.model_state_dict.keys():
                self.model_state_dict[key] = (
                    self.model_state_dict[key] * (1 - self.alpha)
                    + new_state_dict[key] * self.alpha
                )

            model.load_state_dict(self.model_state_dict)


def create_wandb_tags(config: omegaconf.DictConfig):
    config = OmegaConf.create(config)
    wandb_tags = config.wandb_tags if config.wandb_tags is not None else []

    base_tags = [
        config.project_name,
        config.dataset.name,
        str(config.dataset.data_path).split("/")[-1],
        "multiclass" if config.dataset.mask_transformer.multiclass else "binary",
    ]

    wandb_tags = base_tags + wandb_tags

    if (
        "loss_guidance" in config.diffusion
        and config.diffusion.loss_guidance is not None
    ):
        wandb_tags.append(config.diffusion.loss_guidance.guider.name)
        wandb_tags.append(config.diffusion.loss_guidance.mode)

    return wandb_tags


def unpack_batch(batch, phase: str = "train"):
    images, gt_masks, gt_train_masks, guidance_gt = None, None, None, {}

    if len(batch) == 3:
        images, gt_masks, gt_train_masks = batch
    elif len(batch) == 4:
        images, gt_masks, gt_train_masks, guidance_gt = batch
    else:
        raise ValueError(
            f"Batch has {len(batch)} to unpack. A batch can only have 3 or 4 values to unpack."
        )

    assert (
        images.shape[0] == gt_masks.shape[0]
        and gt_masks.shape[0] == gt_train_masks.shape[0]
    ), (
        "Assertion Error: images, gt_masks and gt_train_masks need to have the same number of samples"
    )

    if phase == "train":
        return images, gt_masks, gt_train_masks
    elif phase == "test":
        return images, gt_masks, gt_train_masks, guidance_gt
    else:
        raise ValueError(f"Invalid phase: {phase}. Choose between 'train' and 'test'.")


def log_cuda_memory(stage: str, flush: bool = False) -> None:
    """
    Logs the current cuda memory usage.
    """
    free_mem_bytes, total_mem_bytes = torch.cuda.mem_get_info()
    free_mem = free_mem_bytes / 1_000_000_000
    total_mem = total_mem_bytes / 1_000_000_000
    alloc_mem = torch.cuda.memory_allocated() / 1_000_000_000
    reserved_mem = torch.cuda.memory_reserved() / 1_000_000_000
    max_mem_reserved = torch.cuda.max_memory_reserved() / 1_000_000_000
    max_mem_allocated = torch.cuda.max_memory_allocated() / 1_000_000_000

    print(f"\n>>>>{stage}>>>>")
    print(
        f"\tCurrent, max memory allocated: {alloc_mem:.2f}/{total_mem:.2f}GiB, "
        f"{max_mem_allocated:.2f}/{total_mem:.2f}GiB"
    )
    print(
        f"\tCurrent, max memory reserved: {reserved_mem:.2f}/{total_mem:.2f}GiB, "
        f"{max_mem_reserved:.2f}/{total_mem:.2f}GiB"
    )
    print(f"\tCurrent memory free: {free_mem:.2f}/{total_mem:.2f} GiB")
    print(f"<<<<{stage}<<<<")


def check_topofeatures(topo_features: dict, num_classes: int):
    """
    Check if the topo_features are valid.
    """
    if topo_features is None:
        return None

    if len(topo_features) != num_classes:
        raise ValueError(
            f"Expected {num_classes} topo_features definitions, but got {len(topo_features)}"
        )

    idx_list = [i for i in range(num_classes)]

    for class_idx, topo_feature in topo_features.items():
        if not isinstance(topo_feature, omegaconf.dictconfig.DictConfig):
            raise ValueError(f"Topo feature for class {class_idx} is not a dictionary")
        if class_idx in idx_list:
            idx_list.remove(class_idx)
        else:
            raise ValueError(
                f"Topo feature for class {class_idx} is not in the idx list of the classes"
            )

        if (
            0 in topo_feature.keys()
            and type(topo_feature[0]) is int
            and topo_feature[0] >= 0
        ):
            if (
                1 in topo_feature.keys()
                and type(topo_feature[1]) is int
                and topo_feature[1] >= 0
            ):
                continue
            else:
                raise ValueError(
                    f"Topo feature for class {class_idx} does not contain homology dimension for class 1"
                )
        else:
            raise ValueError(
                f"Topo feature for class {class_idx} does not contain homology dimension for class 0"
            )

    return dict(topo_features)


def get_fixed_betti_numbers(topo_feature: dict, num_classes: int):
    if topo_feature is None:
        return None, None

    if len(topo_feature) != num_classes:
        raise ValueError(
            f"Expected {num_classes} topo_feature definitions, but got {len(topo_feature)}"
        )

    betti_0 = torch.tensor([topo_feature[i][0] for i in range(num_classes)])
    betti_1 = torch.tensor([topo_feature[i][1] for i in range(num_classes)])

    print(f"betti_0 shape: {betti_0.shape}, betti_1 shape: {betti_1.shape}")
    print(f"betti_0: {betti_0}")
    print(f"betti_1: {betti_1}")

    return betti_0, betti_1
