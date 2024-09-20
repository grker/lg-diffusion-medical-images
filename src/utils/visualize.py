import matplotlib.pyplot as plt
from PIL import Image



def visualize_nii(data, start=1, end=2):
    assert(start < end)
    elements = min(end - start, 5)

    if len(data.shape) == 4:
        raise ValueError("Data has 4 dimensions, expected 3. For 4d use visualize_nii_4d")

    fig, axes = plt.subplots(1, elements, figsize=(15, 5))
    
    for i, ax in enumerate(axes):
        ax.imshow(data[:, :, i], cmap='gray')
        ax.axis('off')

    plt.show()


def visualize_nii_4d(data, save_location, slice=0):
    assert(len(data.shape) == 4 and data.shape[2] > slice)
    list_data = [Image.fromarray(data[:, :, slice, i], mode='L') for i in range(data.shape[3])]
    
    list_data[0].save(save_location, save_all=True, append_images=list_data[1:], duration=20, loop=0)