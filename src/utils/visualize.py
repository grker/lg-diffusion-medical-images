import matplotlib.pyplot as plt
import torch
import wandb
import random
import time
import numpy as np
import torchvision

from PIL import Image


def visualize_nii(data, start=1, end=2):
    assert start < end
    elements = min(end - start, 10)

    if len(data.shape) == 4:
        raise ValueError(
            "Data has 4 dimensions, expected 3. For 4d use visualize_nii_4d"
        )

    fig, axes = plt.subplots(1, elements, figsize=(15, 5))

    for i, ax in enumerate(axes):
        ax.imshow(data[:, :, i])
        ax.axis("off")

    plt.show()


def visualize_nii_4d(data, save_location, slice=0):
    assert len(data.shape) == 4 and data.shape[2] > slice
    list_data = [
        Image.fromarray(data[:, :, slice, i], mode="L") for i in range(data.shape[3])
    ]

    list_data[0].save(
        save_location, save_all=True, append_images=list_data[1:], duration=20, loop=0
    )


def visualize_sampling_res(
    image: torch.Tensor,
    pred_mask: torch.Tensor,
    true_mask: torch.Tensor,
    name: str = None,
    batch_idx: int = None,
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    def to_numpy(tensor):
        if tensor.is_cuda:
            tensor = tensor.cpu()
        if tensor.requires_grad:
            tensor = tensor.detach()
        while len(tensor.shape) > 2:
            tensor = tensor.squeeze(0)
        return tensor.numpy()

    axes[0].imshow(to_numpy(image), cmap="gray")
    axes[0].axis("off")
    axes[0].set_title("Image")

    axes[1].imshow(to_numpy(pred_mask), cmap="gray")
    axes[1].axis("off")
    axes[1].set_title("Predicted Mask")

    axes[2].imshow(to_numpy(true_mask), cmap="gray")
    axes[2].axis("off")
    axes[2].set_title("True Mask")

    name = ("_" + str(name)) if name else ""
    name = name + (("_" + str(batch_idx)) if batch_idx else "")
    plt.savefig(f"../results/sampling{name}.png")


def torch_to_2d_numpy(tensor: torch.Tensor):
    tensor = tensor.cpu().detach()
    while len(tensor.shape) > 2:
        tensor = tensor.squeeze()

    return tensor.numpy()


def load_res_to_wandb(
    image: torch.Tensor,
    gt_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    mapping: dict,
    caption: str = "",
):
    image = torch_to_2d_numpy(image)
    if gt_mask is not None:
        gt_mask = torch_to_2d_numpy(gt_mask) + 2

    pred_mask = torch_to_2d_numpy(pred_mask)
    dens, edges = np.histogram(pred_mask)

    if gt_mask is not None:
        return wandb.Image(
            image,
            masks={
                "predictions": {
                    "mask_data": pred_mask,
                    "class_labels": {0: "background", 1: "foreground"},
                },
                "ground_truth": {
                    "mask_data": gt_mask,
                    "class_labels": {2: "background", 3: "foreground"},
                },
            },
            caption=caption,
        )
    else:
        return wandb.Image(
            image,
            masks={
                "predictions": {
                    "mask_data": pred_mask,
                    "class_labels": {0: "background", 1: "foreground"},
                }
            },
            caption=caption,
        )


def create_wandb_image(image: torch.Tensor, caption: str = ""):
    image = torch_to_2d_numpy(image)

    return wandb.Image(image, caption=caption)


def create_wandb_mask_visualization(
    image: torch.Tensor,
    gt_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    mapping: dict,
    caption: str = "",
):
    offset = mapping["offset"]

    image = torch_to_2d_numpy(image)
    if gt_mask is not None:
        gt_mask = torch_to_2d_numpy(gt_mask)

    pred_mask = torch_to_2d_numpy(pred_mask) + offset

    if gt_mask is not None:
        return wandb.Image(
            image,
            masks={
                "predictions": {
                    "mask_data": pred_mask,
                    "class_labels": mapping["class_labels_pred"],
                },
                "ground_truth": {
                    "mask_data": gt_mask,
                    "class_labels": mapping["class_labels"],
                },
            },
            caption=caption,
        )
    else:
        return wandb.Image(
            image,
            masks={
                "predictions": {
                    "mask_data": pred_mask,
                    "class_labels": mapping["class_labels_pred"],
                }
            },
            caption=caption,
        )


def normalize(tensor: torch.Tensor):
    # ensures that the tensor is in the range [0, 1]
    return (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor))


def visualize_segmentation(
    image: torch.Tensor,
    gt_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    before_threshold: torch.Tensor,
    phase: str,
    mapping: dict,
    batch_idx: int,
    num_classes: int,
    img_index_list: list[int] = None,
):
    if img_index_list is None:
        img_index_list = [random.randint(0, image.shape[0] - 1)]

    for img_index in img_index_list:
        caption = f"BatchIdx_{batch_idx}_ImageIdx_{img_index}"
        mask_vis = create_wandb_mask_visualization(
            image[img_index], gt_mask[img_index], pred_mask[img_index], mapping, caption
        )
        pred_mask_wandb = create_wandb_image(normalize(pred_mask[img_index]), caption)
        gt_mask_wandb = create_wandb_image(normalize(gt_mask[img_index]), caption)
        # if num_classes == 2:
        #     before_threshold = create_wandb_image(normalize(before_threshold[img_index]), caption)
        #     wandb.log({f"{phase}_before_threshold": before_threshold})

        wandb.log({f"{phase}_mask_comparison": mask_vis})
        wandb.log({f"{phase}_pred_mask": pred_mask_wandb})
        wandb.log({f"{phase}_gt_mask": gt_mask_wandb})


def normalize_to_0_1(tensor: torch.Tensor, max=None, min=None):
    if max is None:
        max_per_sample = torch.max(torch.flatten(tensor, start_dim=1), dim=1).values
    else:
        max_per_sample = torch.fill(tensor.shape[0], max, device=tensor.device)
    if min is None:
        min_per_sample = torch.min(torch.flatten(tensor, start_dim=1), dim=1).values
    else:
        min_per_sample = torch.fill(tensor.shape[0], min, device=tensor.device)

    return (tensor - min_per_sample) / (max_per_sample - min_per_sample)


def visualize_mean_variance(
    ensemble_mask: list[torch.Tensor],
    phase: str,
    batch_idx: int,
    class_wise: bool = True,
    index_list: list[int] = None,
):
    """
    :param ensemble_mask: list of tensors of shape (num_samples, channels, height, width)
    :param phase: str, phase of the experiment
    :param batch_idx: int, batch index
    :param class_wise: bool, whether to visualize class-wise mean and variance
    :param index_list: list of int, indices of the samples to visualize
    """

    if index_list is None:
        print("No index list provided, using random indices")
        index_list = [random.randint(0, ensemble_mask.shape[1] - 1)]

    ensemble_mask = [rep[index_list] for rep in ensemble_mask]
    ensemble_mask = torch.stack(ensemble_mask, dim=0)

    if ensemble_mask.shape[2] == 1:
        # binary segmentation
        visualize_mean_variance_binary(ensemble_mask, phase, batch_idx)
        return

    num_classes = ensemble_mask.shape[2]

    if class_wise:
        visualize_mean_variance_class_wise(ensemble_mask, phase, batch_idx, num_classes)
        return
    else:
        raise NotImplementedError(
            "Non-class-wise variant for multiple classes not yet implemented"
        )


def visualize_mean_variance_class_wise(
    ensemble_mask: torch.Tensor, phase: str, batch_idx: int, num_classes: int
):
    mean_mask_classes = torch.empty(
        (
            ensemble_mask.shape[1],
            num_classes,
            ensemble_mask.shape[3],
            ensemble_mask.shape[4],
        )
    )
    std_mask_classes = torch.empty(
        (
            ensemble_mask.shape[1],
            num_classes,
            ensemble_mask.shape[3],
            ensemble_mask.shape[4],
        )
    )

    for class_idx in range(num_classes):
        class_mask = ensemble_mask[:, :, class_idx, :, :]

        mean_mask = torch.mean(class_mask, dim=0)
        std_mask = torch.std(class_mask, dim=0)

        mean_mask = torch.clamp(mean_mask, min=0, max=1)
        std_mask = torch.clamp(std_mask, min=0, max=1)

        mean_mask_classes[:, class_idx, :, :] = mean_mask
        std_mask_classes[:, class_idx, :, :] = std_mask

    for idx in range(mean_mask_classes.shape[0]):
        mask = torchvision.utils.make_grid(
            mean_mask_classes[idx].unsqueeze(1), nrow=min(num_classes, 4), padding=10
        )
        wandb.log(
            {
                "mean_mask": wandb.Image(
                    torchvision.utils.make_grid(
                        mean_mask_classes[idx].unsqueeze(1),
                        nrow=min(num_classes, 4),
                        padding=10,
                    ),
                    caption=f"Testing VarianceBIdx_{batch_idx}_Idx_{idx}",
                )
            }
        )
        wandb.log(
            {
                "std_mask": wandb.Image(
                    torchvision.utils.make_grid(
                        std_mask_classes[idx].unsqueeze(1),
                        nrow=min(num_classes, 4),
                        padding=10,
                    ),
                    caption=f"Testing VarianceBIdx_{batch_idx}_Idx_{idx}",
                )
            }
        )


def visualize_mean_variance_binary(
    ensemble_mask: torch.Tensor, phase: str, batch_idx: int
):
    mean_mask = torch.mean(ensemble_mask, dim=0)
    std_mask = torch.std(ensemble_mask, dim=0)

    mean_mask = torch.clamp(mean_mask, min=0, max=1)

    std_mask = torch.clamp(std_mask, min=0, max=1)

    for idx in range(mean_mask.shape[0]):
        wandb.log(
            {
                "mean_mask": create_wandb_image(
                    mean_mask[idx],
                    caption=f"Testing VarianceBIdx_{batch_idx}_Idx_{idx}",
                )
            }
        )
        wandb.log(
            {
                "std_mask": create_wandb_image(
                    std_mask[idx], caption=f"Testing VarianceBIdx_{batch_idx}_Idx_{idx}"
                )
            }
        )


def gif_over_timesteps(
    prediction: torch.Tensor,
    gt_mask_one_hot: torch.Tensor,
    save_path: str,
    mode: str = "probs",
    time: int = 200,
    softmax: bool = True,
):
    """
    :param mask: tensor of shape (timesteps, reps, classes, H, W)
    :param gt_mask: tensor of shape (classes, H, W)
    :param save_path: str, path to save the gif
    :param mode: str, mode of the gif, either "probs" or "argmax" or "probs_argmax"
    """

    assert len(prediction.shape) == 5
    assert len(gt_mask_one_hot.shape) == 3

    prediction = torch.mean(prediction, dim=1)
    prediction = torch.softmax(prediction, dim=1) if softmax else prediction

    if mode == "argmax" or mode == "probs_argmax":
        mask = torch.argmax(prediction, dim=1)

        if mode == "argmax":
            prediction = mask

        if mode == "probs_argmax":
            mask_one_hot = torch.zeros_like(prediction).scatter_(1, mask, 1)
            prediction = mask_one_hot * prediction

    frames = []
    for timestep in range(prediction.shape[0]):
        concat_mask = torch.cat(
            [prediction[timestep], gt_mask_one_hot], dim=0
        ).unsqueeze(1)
        grid = torchvision.utils.make_grid(
            concat_mask, nrow=min(prediction.shape[1], 4), padding=10
        )

        frame = grid.permute(1, 2, 0).numpy() * 255
        frames.append(Image.fromarray(frame.astype(np.uint8)))

    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=time,  # Milliseconds per frame
        loop=0,
    )

    # wandb.log({"gif_over_timesteps": wandb.Video(frames, caption=f"Gif over timesteps for mode {mode}")})


def visualize_component_map(
    component_map: torch.Tensor, title: str, batch_idx: int, merged: bool = True
):
    """
    :param component_map: tensor of shape (batch_size, num_classes, H, W)
    :param title: str, title of the visualization
    """

    num_classes = component_map.shape[1]

    if merged:
        for class_idx in range(num_classes):
            component_map[:, class_idx, :, :] = (
                component_map[:, class_idx, :, :] * class_idx
            )

        component_map = torch.argmax(component_map, dim=1)

        for idx in range(component_map.shape[0]):
            wandb.log(
                {
                    f"{title}": create_wandb_image(
                        normalize(component_map[idx]), f"BIdx_{batch_idx}_Idx_{idx}"
                    )
                }
            )

    else:
        for idx in range(component_map.shape[0]):
            concat_component_map = torch.cat(
                [
                    component_map[idx, class_idx, :, :]
                    for class_idx in range(num_classes)
                ],
                dim=1,
            )
        grid = torchvision.utils.make_grid(
            concat_component_map, nrow=min(num_classes, 4), padding=10
        )
        print(f"grid shape: {grid.shape}", flush=True)
        wandb.log({f"{title}": wandb.Image(grid, f"BIdx_{batch_idx}_Idx_{idx}")})
