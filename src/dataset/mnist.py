import os
import struct
import torch
import numpy as np

from monai.transforms import Resize, Compose
from array import array
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

from utils.hydra_config import DatasetConfig


class MNISTDataset(Dataset):
    training_images_filepath: str
    training_labels_filepath: str
    test_images_filepath: str
    test_labels_filepath: str

    length: int
    images: torch.Tensor
    labels: torch.Tensor

    def __init__(self, folder_path):
        self.training_images_filepath = os.path.join(folder_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
        self.training_labels_filepath = os.path.join(folder_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
        self.test_images_filepath = os.path.join(folder_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
        self.test_labels_filepath = os.path.join(folder_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

        self.load_data()

    def load_mnist_images(self, labels_filepath, images_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())

        images = torch.empty(size, rows, cols)
        
        for i in range(size):
            img = torch.Tensor(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(rows, cols)
            images[i, :, :] = img

        labels = torch.Tensor(labels)            
        
        return images, labels
    

    def load_data(self):
        x_train, y_train = self.load_mnist_images(self.training_labels_filepath, self.training_images_filepath)
        x_test, y_test = self.load_mnist_images(self.test_labels_filepath, self.test_images_filepath)

        self.images = torch.cat((x_train, x_test))
        self.labels = torch.cat((y_train, y_test))
        

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
    


class M2NISTDataset(Dataset):
    data: torch.Tensor
    segmentations: torch.Tensor

    def __init__(self, config: DatasetConfig):
        self.norm = Normalize([0.5], [0.5])
        segmentations = torch.from_numpy(np.load(os.path.join(config.data_path, 'segmented.npy'))[:, :, :, -1])
        data = torch.from_numpy(np.load(os.path.join(config.data_path, 'combined.npy')))

        self.segmentations = self.prepare_data(data=segmentations, image_size=config.image_size)
        self.segmentations = torch.where(self.segmentations == 1.0, 0.0, 1.0)
        self.data = self.prepare_data(data=data, image_size=config.image_size, normalize=config.normalize, mode=(config.mode or 'bilinear'))

        
    def prepare_data(self, data: torch.Tensor, image_size: tuple[int, int] = None, normalize: bool = False, mode: str = 'nearest'):
        if image_size is not None:
            resize_fn = Resize(spatial_size=image_size, mode=mode)
            data = resize_fn(data)
        
        if normalize:
            data = data / 255.0
            # data = self.norm(data)

        return data.type(torch.float32).unsqueeze(1)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.segmentations[idx]
    
    def get_image_size(self):
        return self.data.shape[1:]
    
    def get_image_height(self):
        return self.data.shape[2]
    
    def get_image_width(self):
        return self.data.shape[3]
    
