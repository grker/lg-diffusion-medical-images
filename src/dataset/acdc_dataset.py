import torch
import os
import nibabel as nib
import numpy as np
import cv2

from torch.utils.data import Dataset
from utils.hydra_config import DatasetConfig


class ACDCDataset(Dataset):
    training_folder: str
    testing_folder: str
    cuts: list[int]
    image_size: tuple[int,int]

    data: torch.Tensor
    gt: torch.Tensor
    patient_metadata: list[dict]

    training_samples: int
    test_samples: int
    ratio: float

    def __init__(self, config: DatasetConfig):
        self.training_folder = os.path.join(config.data_path, 'training')
        self.testing_folder = os.path.join(config.data_path, 'testing')

        self.image_size = config.image_size
        self.normalize = config.normalize
        self.mode = config.mode
        self.multiclass = config.multiclass
        self.patient_metadata = []
        self.load_data()


    def load_data(self):
        training_data, training_gt = self.load_patients(self.training_folder)
        testing_data, testing_gt = self.load_patients(self.testing_folder)

        self.training_samples = training_data.shape[0]
        self.test_samples = testing_data.shape[0]
        self.ratio = self.training_samples / (self.training_samples + self.test_samples)  

        self.data = torch.cat((training_data, testing_data), dim=0).unsqueeze(1).type(dtype=torch.float32)
        self.gt = torch.cat((training_gt, testing_gt), dim=0).unsqueeze(1)

        if not self.multiclass:
            self.gt = (self.gt > 0).type(torch.float32)

        print(f"shape of data: {self.data.shape}")
        print(f"shape of data: {self.gt.shape}")

        print(f"type of data: {self.data.type()}")
        print(f"type of gt: {self.gt.type()}")


    def load_patients(self, folder_path):
        data, gt = torch.empty(0), torch.empty(0)
        index = 0
        self.idx_to_patient = []
        for patient in os.listdir(folder_path):
            if patient.startswith('patient'):
                patient_path = os.path.join(folder_path, patient)
                data_patient, gt_patient = self.load_patient(patient_path)
                data_patient, gt_patient, patient_scaler = self.normalize_and_augment_patient_data(data_patient, gt_patient)

                self.idx_to_patient.append(index)
                self.patient_metadata.append({
                    'patient': patient,
                    'startIdx': index,
                    'frames': data_patient.shape[0],
                    'scaler': patient_scaler,
                })
                index += data_patient.shape[0]

                data = torch.cat((data, data_patient), dim=0)
                gt = torch.cat((gt, gt_patient), dim=0)
        
        return data, gt
    
    def normalize_and_augment_patient_data(self, patient_data: torch.Tensor, patient_gt):
        from sklearn.preprocessing import MinMaxScaler

        patient_scaler = MinMaxScaler()
        return patient_scaler.fit_transform(patient_data), patient_gt, patient_scaler

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

        print(f"data patient shape: {data_patient.shape}")
        
        return data_patient, gt_patient
        
    
    def load_frame(self, patient_path: str, frame_info: list[dict]):
        assert(len(frame_info) == 2 and frame_info[0]['gt'] != frame_info[1]['gt'])
        
        data, gt = None, None
        for frame in frame_info:
            if frame['gt']:
                gt_tmp = nib.load(os.path.join(patient_path, frame['file'])).get_fdata()
                gt_tmp = np.moveaxis(gt_tmp, -1, 0)
                gt_resized = np.zeros((gt_tmp.shape[0], self.image_size[0], self.image_size[1]))
                for i in range(gt_tmp.shape[0]):
                    gt_resized[i] = cv2.resize(gt_tmp[i], self.image_size)
                gt = torch.from_numpy(gt_resized)
            else:
                data_tmp = nib.load(os.path.join(patient_path, frame['file'])).get_fdata()
                data_tmp = np.moveaxis(data_tmp, -1, 0)
                data_resized = np.zeros((data_tmp.shape[0], self.image_size[0], self.image_size[1]))
                for i in range(data_tmp.shape[0]):
                    data_resized[i] = cv2.resize(data_tmp[i], self.image_size)
                data = torch.from_numpy(data_resized)
        
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
        return self.data[idx], self.gt[idx]

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