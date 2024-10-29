from typing import Union

import torch

class BaseMaskMapping:
    schema = {}
    foreground_classes: int=1

    def check_input(self, dictionary):
        for key in self.schema:
            if key not in dictionary:
                raise KeyError(f"Key {key} not found in the dictionary")
            elif type(dictionary[key]) != self.schema[key]:
                raise TypeError(f"Value of Key {key} has type {type(dictionary[key])} but should have been {self.schema[key]}")
            
        return dictionary
    
    def preprocess_gt_mask(self, mask: torch.Tensor):
        raise NotImplementedError()

    def to_train_mask(self, mask: torch.Tensor):
        raise NotImplementedError()
    
    def to_gt_mask(self, mask: torch.Tensor):
        raise NotImplementedError()
    
    def get_num_classes_fg(self):
        return self.foreground_classes
    

class IdentityMaskMapping(BaseMaskMapping):

    def preprocess_gt_mask(self, mask: torch.Tensor):
        return mask
    
    def to_train_mask(self, mask: torch.Tensor):
        return mask
    
    def to_gt_mask(self, mask: torch.Tensor):
        return mask
    
    def create_gt_mask_from_pred(self, mask: torch.Tensor):
        return mask
    


class BinarySegMaskMapping(BaseMaskMapping):
    schema = {
        "foreground": int,
        "background": int
    }
    
    
    def __init__(self, gt_setting: dict, train_setting: dict) -> None:
        self.gt = self.check_input(gt_setting)
        self.train = self.check_input(train_setting)

        self.set_threshold_func((self.train["foreground"] + self.train["background"]) / 2)

    
    def set_threshold_func(self, threshold: int, inverse: bool=False):
        if not inverse:
            if self.train["foreground"] > self.train["background"]:
                self.threshold_func = lambda tensor: torch.where(tensor > threshold, self.train["foreground"], self.train["background"])
            else:
                self.threshold_func = lambda tensor: torch.where(tensor > threshold, self.train["background"], self.train["foreground"])
        else:
            if self.train["foreground"] > self.train["background"]:
                self.threshold_func = lambda tensor: torch.where(tensor > threshold, self.train["background"], self.train["foreground"])
            else:
                self.threshold_func = lambda tensor: torch.where(tensor > threshold, self.train["foreground"], self.train["background"])


    # Ensures that the gt mask is binary
    def preprocess_gt_mask(self, mask: torch.Tensor):
        type_mask = mask.type()
        mask = torch.where(mask == self.gt["background"], self.gt["background"], self.gt["foreground"])
        return mask.type(type_mask)
    

    def to_train_mask(self, mask: torch.Tensor):
        type_mask = mask.type()
        mask = torch.where(mask == self.gt["background"], self.train["background"], self.train["foreground"])
        return mask.type(type_mask)
    
    def to_gt_mask(self, mask: torch.Tensor):
        type_mask = mask.type()
        mask = torch.where(mask == self.train["background"], self.gt["background"], self.gt["foreground"])
        return mask.type(type_mask)
    
    
    def create_gt_mask_from_pred(self, mask: torch.Tensor):
        train_mask = self.threshold_func(mask)
        return self.to_gt_mask(train_mask)
    
    def gt_mapping_for_visualization(self):
        class_labels = {
            self.gt["background"]: "background",
            self.gt["foreground"]: "foreground"
        }

        offset = self.gt["background"] + self.gt["foreground"] + 1

        class_labels_pred = {
            (self.gt["background"] + offset): "background",
            (self.gt["foreground"] + offset): "foreground"
        }

        return {
            "class_labels": class_labels,
            "class_labels_pred": class_labels_pred,
            "offset": offset
        }


def generate_mask_mapping(mask_type: str, gt_setting: dict, train_setting: dict) -> BaseMaskMapping:
    if mask_type == "binary":
        return BinarySegMaskMapping(gt_setting, train_setting)
    elif mask_type == "identity":
        return IdentityMaskMapping()
    else:
        raise ValueError(f"Maks type {mask_type} has not been implemented.")
