import torch
import os
import nibabel as nib
import numpy as np
import cv2
from torchvision.transforms import Normalize
from torch.utils.data import Dataset
from monai.transforms import CenterSpatialCrop

from utils.hydra_config import DatasetConfig
from utils.mask_transformer import generate_mask_mapping, BaseMaskMapping


class ACDCDataset(Dataset):
    training_folder: str
    testing_folder: str
    cuts: list[int]
    image_size: tuple[int,int]

    data: torch.Tensor
    gt: torch.Tensor
    gt_train: torch.Tensor
    patient_metadata: list[dict]

    training_samples: int
    test_samples: int
    ratio: float

    mask_transformer: BaseMaskMapping

    def __init__(self, config: DatasetConfig):
        print(f"data path: {config.data_path}")
        
        if config.data_path.startswith('../') and not os.path.exists(os.path.join(os.getcwd(), config.data_path)):
            config.data_path = config.data_path[3:]
            print(f"new data path: {config.data_path}")

        self.training_folder = os.path.join(config.data_path, 'training')
        self.testing_folder = os.path.join(config.data_path, 'testing')

        self.image_size = config.image_size
        self.normalize = config.normalize
        self.mode = config.mode
        self.patient_metadata = []
        self.mask_transformer = generate_mask_mapping(config.mask_transformer)

        self.cropping = CenterSpatialCrop(roi_size=[154, 154])

        self.load_data()
        


    def load_data(self):
        training_data, training_gt = self.load_patients(self.training_folder)
        testing_data, testing_gt = self.load_patients(self.testing_folder)

        self.norm = Normalize(mean=[0.5], std=[0.5])

        self.training_samples = training_data.shape[0]
        self.test_samples = testing_data.shape[0]
        self.ratio = self.training_samples / (self.training_samples + self.test_samples)  

        self.data = torch.cat((training_data, testing_data), dim=0).unsqueeze(1).type(dtype=torch.float32)
        
        self.gt = self.mask_transformer.dataset_to_gt_mask(torch.cat((training_gt, testing_gt), dim=0).unsqueeze(1)).type(dtype=torch.float32)
        self.gt_train = self.mask_transformer.gt_to_train_mask(self.gt)
        
        print(f"gt train shape: {self.gt_train.shape}")
        print(f"data shape: {self.data.shape}")
        print(f"histogram of gt train: {torch.histc(self.gt_train, bins=10)}")


    def load_patients(self, folder_path):
        data, gt = torch.empty(0), torch.empty(0)
        index = 0
        self.idx_to_patient = []
        for patient in os.listdir(folder_path):
            if patient.startswith('patient'):
                patient_path = os.path.join(folder_path, patient)
                data_patient, gt_patient = self.load_patient(patient_path)
                data_patient, gt_patient, max_value = self.normalize_and_augment_patient_data(data_patient, gt_patient)

                data = torch.cat((data, data_patient), dim=0)
                gt = torch.cat((gt, gt_patient), dim=0)
        
        return data, gt
    
    def normalize_and_augment_patient_data(self, patient_data: torch.Tensor, patient_gt):
        max_value = torch.max(patient_data)
    
        return patient_data / max_value, patient_gt, max_value

    def load_patient(self, patient_path):
        frames = {}

        for file in os.listdir(patient_path):
            if file.endswith('.nii'):
                splits = file.split('_')
                frame = splits[1].split('.')[0]
                if frame in frames.keys() and frame.startswith('frame'):
                    frames[frame].append({'file': file, 'gt': len(splits) == 3})
                elif frame.startswith('frame'):
                    frames[frame] = [{'file': file, 'gt': len(splits) == 3}]
        
        data_patient, gt_patient = torch.empty(0), torch.empty(0)

        for frame in frames.values():
            data_tmp, gt_tmp = self.load_frame(patient_path, frame)
            data_patient = torch.cat((data_patient, data_tmp), dim=0) 
            gt_patient = torch.cat((gt_patient, gt_tmp), dim=0)
        
        return data_patient, gt_patient
        
    
    def load_frame(self, patient_path: str, frame_info: list[dict]):
        assert(len(frame_info) == 2 and frame_info[0]['gt'] != frame_info[1]['gt'])
        
        data, gt = None, None
        for frame in frame_info:
            if frame['gt']:
                gt_tmp = nib.load(os.path.join(patient_path, frame['file'])).get_fdata()
                gt_tmp = np.moveaxis(gt_tmp, -1, 0)
                gt_resized = torch.zeros((gt_tmp.shape[0], 154, 154))
                for i in range(gt_tmp.shape[0]):
                    # gt_resized[i] = cv2.resize(gt_tmp[i], self.image_size)
                    gt_resized[i] = self.cropping(torch.from_numpy(gt_tmp[i]).unsqueeze(0)).squeeze(0)
                gt =gt_resized
            else:
                data_tmp = nib.load(os.path.join(patient_path, frame['file'])).get_fdata()
                data_tmp = np.moveaxis(data_tmp, -1, 0)
                data_resized = torch.zeros((data_tmp.shape[0], 154, 154))
                for i in range(data_tmp.shape[0]):
                    # data_resized[i] = cv2.resize(data_tmp[i], self.image_size)
                    data_resized[i] = self.cropping(torch.from_numpy(data_tmp[i]).unsqueeze(0)).squeeze(0)
                data = data_resized
        
        good_indices = []

        for i, gt_mask in enumerate(gt):
            if torch.sum(gt_mask, dim=(-2,-1)) > 0:
                # only keep the non-empty segmentation masks
                good_indices.append(i)
        
        data = data[good_indices]
        gt = gt[good_indices]
        return data, gt

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # return self.data[idx], self.gt[idx]
        return self.data[idx], self.gt[idx], self.gt_train[idx]

    def get_image_size(self):
        return self.data.shape[1:]
    
    def get_image_height(self):
        return self.data.shape[2]
    
    def get_image_width(self):
        return self.data.shape[3]
    

class ACDCDatasetGTAutoEncoder(ACDCDataset):
    def __init__(self, folder_path: str, image_size: tuple[int,int] = (256,256), cuts: list[int] = [0,1,2,3,4,5,6,7,8,9]):
        super(ACDCDatasetGTAutoEncoder, self).__init__(folder_path, image_size, cuts)
    
    def __getitem__(self, idx):
        return self.gt[idx], self.gt[idx]