import matplotlib.pyplot as plt
import torch

from PIL import Image



def visualize_nii(data, start=1, end=2):
    assert(start < end)
    elements = min(end - start, 10)

    if len(data.shape) == 4:
        raise ValueError("Data has 4 dimensions, expected 3. For 4d use visualize_nii_4d")

    fig, axes = plt.subplots(1, elements, figsize=(15, 5))
    
    for i, ax in enumerate(axes):
        ax.imshow(data[:, :, i])
        ax.axis('off')

    plt.show()


def visualize_nii_4d(data, save_location, slice=0):
    assert(len(data.shape) == 4 and data.shape[2] > slice)
    list_data = [Image.fromarray(data[:, :, slice, i], mode='L') for i in range(data.shape[3])]
    
    list_data[0].save(save_location, save_all=True, append_images=list_data[1:], duration=20, loop=0)


def visualize_sampling_res(image: torch.Tensor, pred_mask: torch.Tensor, true_mask: torch.Tensor, name: str=None, batch_idx: int=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    def to_numpy(tensor):
        if tensor.is_cuda:
            tensor = tensor.cpu()
        if tensor.requires_grad:
            tensor = tensor.detach()
        while len(tensor.shape) > 2:
            tensor = tensor.squeeze(0)
        return tensor.numpy()
    
    axes[0].imshow(to_numpy(image), cmap='gray')
    axes[0].axis('off')
    axes[0].set_title('Image')

    axes[1].imshow(to_numpy(pred_mask), cmap='gray')
    axes[1].axis('off')
    axes[1].set_title('Predicted Mask')

    axes[2].imshow(to_numpy(true_mask), cmap='gray')
    axes[2].axis('off')
    axes[2].set_title('True Mask')

    name = ('_' + str(name)) if name else ''
    name = name + (('_' + str(batch_idx)) if batch_idx else '')
    plt.savefig(f'../results/sampling{name}.png')
