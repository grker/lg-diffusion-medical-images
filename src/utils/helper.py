import omegaconf
import torch


def unpack_batch(batch):
    images, gt_masks, gt_train_masks = None, None, None
    if len(batch) == 2:
        images, gt_train_masks = batch
        gt_masks = gt_train_masks
    elif len(batch) == 3:
        images, gt_masks, gt_train_masks = batch
    else:
        raise ValueError(
            f"Batch has {len(batch)} to unpack. A batch can only have 2 or 3 values to unpack."
        )

    assert (
        images.shape[0] == gt_masks.shape[0]
        and gt_masks.shape[0] == gt_train_masks.shape[0]
    ), (
        "Assertion Error: images, gt_masks and gt_train_masks need to have the same number of samples"
    )

    return images, gt_masks, gt_train_masks


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
