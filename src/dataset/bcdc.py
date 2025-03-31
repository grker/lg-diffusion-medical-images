import os

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.hydra_config import DatasetConfig
from utils.mask_transformer import BaseMaskMapping, generate_mask_mapping


class BCDCDataset(Dataset):
    training_folder: str
    testing_folder: str

    images: torch.Tensor
    gt: torch.Tensor
    gt_train: torch.Tensor
    components: torch.Tensor

    mask_transformer: BaseMaskMapping

    topo_infos: list[str] = ["components"]

    def __init__(self, config: DatasetConfig):
        self.training_folder = os.path.join(os.getcwd(), config.data_path, "training")
        self.testing_folder = os.path.join(os.getcwd(), config.data_path, "testing")
        self.normalize = config.normalize

        self.mask_transformer = generate_mask_mapping(config.mask_transformer)

        self.load_data()

    def load_data(self):
        training_images, training_gt, training_gt_train, training_components = (
            self.load_partition(
                self.training_folder,
            )
        )

        testing_images, testing_gt, testing_gt_train, testing_components = (
            self.load_partition(
                self.testing_folder,
            )
        )

        self.images = torch.cat((training_images, testing_images), dim=0)

        if self.normalize:
            self.images = self.images / 255.0

        self.gt = torch.cat((training_gt, testing_gt), dim=0)
        self.gt_train = torch.cat((training_gt_train, testing_gt_train), dim=0)
        self.components = torch.cat((training_components, testing_components), dim=0)

        print(f"images shape: {self.images.shape}")
        print(f"gt shape: {self.gt.shape}")
        print(f"gt_train shape: {self.gt_train.shape}")
        print(f"components shape: {self.components.shape}")


        print(f"images min: {self.images.min()}")
        print(f"images max: {self.images.max()}")

        print(f"gt min: {self.gt.min()}")
        print(f"gt max: {self.gt.max()}")

        print(f"gt_train min: {self.gt_train.min()}")
        print(f"gt_train max: {self.gt_train.max()}")
        

    def load_partition(self, folder_path: str):
        images = torch.from_numpy(np.load(os.path.join(folder_path, "images.npy")))
        masks = torch.from_numpy(np.load(os.path.join(folder_path, "masks.npy")))
        components = torch.from_numpy(
            np.load(os.path.join(folder_path, "components.npy"))
        )

        gt = self.mask_transformer.dataset_to_gt_mask(masks.type(torch.float32))
        gt_train = self.mask_transformer.gt_to_train_mask(gt)

        return images, gt, gt_train, components

    def __len__(self):
        return self.images.shape[0]

    def get_image_size(self):
        return self.images.shape[1:]

    def get_image_height(self):
        return self.images.shape[2]

    def get_image_width(self):
        return self.images.shape[3]

    def get_topo_infos(self):
        return self.topo_infos
