
def unpack_batch(batch):
    images, gt_masks, gt_train_masks = None, None, None
    if len(batch) == 2:
        images, gt_train_masks = batch
        gt_masks = gt_train_masks
    elif len(batch) == 3:
        images, gt_masks, gt_train_masks = batch
    else:
        raise ValueError(f"Batch has {len(batch)} to unpack. A batch can only have 2 or 3 values to unpack.")
    
    assert(images.shape[0] == gt_masks.shape[0] and  gt_masks.shape[0] == gt_train_masks.shape[0], "Assertion Error: images, gt_masks and gt_train_masks need to have the same number of samples")

    return images, gt_masks, gt_train_masks