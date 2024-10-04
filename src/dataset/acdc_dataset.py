import torch
import os
import nibabel as nib
import numpy as np
import cv2

from torch.utils.data import Dataset


class ACDCDataset(Dataset):
    training_folder: str
    testing_folder: str
    cuts: list[int]
    image_size: tuple[int,int]

    data: torch.Tensor
    gt: torch.Tensor

    training_samples: int
    test_samples: int
    ratio: float

    def __init__(self, folder_path: str, image_size: tuple[int,int] = (256,256), cuts: list[int] = [0,1,2,3,4,5,6,7,8,9]):
        self.training_folder = os.path.join(folder_path, 'training')
        self.testing_folder = os.path.join(folder_path, 'testing')

        self.image_size = image_size
        self.cuts = [cut for cut in cuts if cut >= 0 and cut < 10]
        self.load_data()


    def load_data(self):
        training_data, training_gt = self.load_patients(self.training_folder)
        testing_data, testing_gt = self.load_patients(self.testing_folder)

        self.training_samples = training_data.shape[0]
        self.test_samples = testing_data.shape[0]
        self.ratio = self.training_samples / (self.training_samples + self.test_samples)  

        self.data = torch.cat((training_data, testing_data), dim=0)
        self.gt = torch.cat((training_gt, testing_gt), dim=0)


    def load_patients(self, folder_path):
        data, gt = torch.empty(0), torch.empty(0)
        for patient in os.listdir(folder_path):
            if patient.startswith('patient'):
                patient_path = os.path.join(folder_path, patient)
                data_patient, gt_patient = self.load_patient(patient_path)
                data = torch.cat((data, data_patient), dim=0)
                gt = torch.cat((gt, gt_patient), dim=0)
        
        return data, gt

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
        print(f"processing patient: {patient_path.split('/')[-1]}")
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
                    print(f"resizing image {data_tmp[i].shape}")
                    print(f"image type: {type(data_tmp[i])}")
                    print(f"image size: {self.image_size}")
                    print(f"image size type: {type(self.image_size)}")
                    data_resized[i] = cv2.resize(data_tmp[i], self.image_size)
                data = torch.from_numpy(data_resized)
        
        print(f"Shape: {data.shape}, GT: {gt.shape}")
        return data, gt

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.gt[idx]
    

class ACDCDatasetGTAutoEncoder(ACDCDataset):
    def __init__(self, folder_path: str, image_size: tuple[int,int] = (256,256), cuts: list[int] = [0,1,2,3,4,5,6,7,8,9]):
        super(ACDCDatasetGTAutoEncoder, self).__init__(folder_path, image_size, cuts)
    
    def __getitem__(self, idx):
        return self.gt[idx], self.gt[idx]