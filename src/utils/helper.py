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
