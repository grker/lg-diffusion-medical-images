import torch
import logging

from utils.hydra_config import MaskTransformerConfig

logger = logging.getLogger(__name__)


class BaseMaskMapping:
    dataset_mapping = dict=None
    gt_mapping = dict=None
    num_classes: int # always includes the background
    ensemble_mode: str

    def dataset_to_gt_mask(self, mask: torch.Tensor):
        raise NotImplementedError()
    
    def gt_to_dataset_mask(self, mask: torch.Tensor):
        raise NotImplementedError()
    
    def gt_to_train_mask(self, mask: torch.Tensor):
        raise NotImplementedError()
    
    def get_logits(self, mask: torch.Tensor):
        raise NotImplementedError()
    
    def get_segmentation(self, mask: torch.Tensor):
        raise NotImplementedError()
    
    def get_num_train_channels(self):
        raise NotImplementedError()
    
    def gt_mapping_for_visualization(self):
        raise NotImplementedError()
    
    def get_num_classes(self):
        return self.num_classes
    

class MultiClassSegMaskMapping(BaseMaskMapping):
    train_switch: bool
    prediction_type: str
    num_classes: int


    def __init__(self, dataset_mapping: dict, switch: bool=False, prediction_type: str="sample", ensemble_mode: str="mean") -> None:
        if prediction_type == "sample" and switch:
            logger.warning("Switch is set to True but prediction_type is set to sample. Switch will be ignored.")
            self.train_switch = False
        else:
            self.train_switch = switch
        
        self.gt_mapping = {}
        self.ensemble_mode = ensemble_mode

        if dataset_mapping["background"] is not None and isinstance(dataset_mapping["background"], int):
            self.gt_mapping["background"] = 0 # background class is always 0 --> ensures that the metrics work as expected
        else:
            raise ValueError(f"Background class is not defined in dataset_mapping: {dataset_mapping}.")
        
        index = 1
        items = sorted(dataset_mapping.items(), key=lambda item: item[1])
        for keys, value in items:
            if keys != "background":
                if not isinstance(value, int):
                    raise TypeError(f"Value of Key {keys} has type {type(value)} but should have been {int}")
                else: 
                    self.gt_mapping[keys] = index
                    index += 1

        self.dataset_mapping = dataset_mapping
        self.num_classes = len(dataset_mapping.keys())

    
    def dataset_to_gt_mask(self, mask: torch.Tensor):
        gt_mask = mask.clone()
        for class_name in self.dataset_mapping.keys():
            gt_mask = torch.where(mask == self.dataset_mapping[class_name], self.gt_mapping[class_name], gt_mask)
        return gt_mask.type(mask.type())
    
    def gt_to_dataset_mask(self, mask: torch.Tensor):
        dataset_mask = mask.clone()
        for class_name in self.gt_mapping.keys():
            dataset_mask = torch.where(mask == self.gt_mapping[class_name], self.dataset_mapping[class_name], dataset_mask)
        return dataset_mask.type(mask.type())
    
    def gt_to_train_mask(self, mask: torch.Tensor):
        fg_value = 0 if self.train_switch else 1
        train_mask = torch.empty(mask.shape[0], self.get_num_train_channels(), mask.shape[2], mask.shape[3])

        for class_name, values in self.gt_mapping.items():
            train_mask[:, values, :, :] = torch.where(mask == values, fg_value, 1-fg_value).squeeze(1)

        return train_mask.type(mask.type())
    
    def get_logits(self, mask: torch.Tensor):
        # Mask is of shape (reps, N, num_classes, H, W)
        # Output is of shape (reps, N, num_classes, H, W)
        return (-1) * mask if self.train_switch else mask
        

    def get_segmentation(self, logits: torch.Tensor):
        # Logits are of shape (reps, N, num_classes, H, W)
        # Output is of shape ((N, 1, H, W), (N, C, H, W))
        assert(logits.shape[2] == self.num_classes)

        if self.ensemble_mode == "mean":
            logits = torch.mean(logits, dim=0)
            segmentation = torch.argmax(logits, dim=1, keepdim=True)
        elif self.ensemble_mode == "median":
            logits = torch.median(logits, dim=0)
            segmentation = torch.argmax(logits, dim=1, keepdim=True)
        elif self.ensemble_mode == "majority":
            segmentations_across_reps = torch.argmax(logits, dim=2, keepdim=True).type(torch.int32)
            segmentation = torch.mode(segmentations_across_reps, dim=0)

        print(f"segmentation shape: {segmentation.shape}")
        assert(len(segmentation.shape) == 4)

        return segmentation, torch.zeros_like(logits, device=logits.device).scatter_(1, segmentation, 1)
    
    def gt_mapping_for_visualization(self):
        offset = self.num_classes + 1
        class_labels = {}
        class_labels_pred = {}

        for class_name, value in self.gt_mapping.items():
            class_labels[value] = class_name
            class_labels_pred[value + offset] = class_name
        
        return {
            "class_labels": class_labels,
            "class_labels_pred": class_labels_pred,
            "offset": offset
        }

    def get_num_train_channels(self):
        return self.num_classes
    

class BinarySegMaskMapping(BaseMaskMapping):
    train_mapping: dict
    threshold_func: callable

    
    def __init__(self, dataset_mapping: dict, train_mapping: dict, threshold: float=None, ensemble_mode: str="mean") -> None:
        self.num_classes = 2
        self.gt_mapping = {"background": 0, "foreground": 1}
        self.dataset_mapping = self.check_input(dataset_mapping)
        self.train_mapping = self.check_input(train_mapping)
        self.ensemble_mode = ensemble_mode

        threshold = (self.train_mapping["foreground"] + self.train_mapping["background"]) / 2 if threshold is None else threshold
        self.set_threshold_func(threshold)

    
    def check_input(self, dictionary):
        if dictionary is not None:
            if dictionary["background"] is None or not isinstance(dictionary["background"], int):
                raise ValueError(f"Background class is not defined in dataset_mapping: {dictionary}.")
            if dictionary["foreground"] is None or not isinstance(dictionary["foreground"], int):
                raise ValueError(f"Foreground class is not defined in dataset_mapping: {dictionary}.")
        else:
            raise ValueError(f"dataset_mapping is not defined.")
        return dictionary

    
    def set_threshold_func(self, threshold: int):
        if self.train_mapping["foreground"] > self.train_mapping["background"]:
            self.threshold_func = lambda tensor: torch.where(tensor > threshold, self.train_mapping["foreground"], self.train_mapping["background"])
        else:
            self.threshold_func = lambda tensor: torch.where(tensor > threshold, self.train_mapping["background"], self.train_mapping["foreground"])
        
    
    def dataset_to_gt_mask(self, mask: torch.Tensor):
        gt_mask = torch.where(mask == self.dataset_mapping["background"], self.gt_mapping["background"], self.gt_mapping["foreground"])
        return gt_mask.type(mask.type())
    
    def gt_to_dataset_mask(self, mask: torch.Tensor):
        dataset_mask = torch.where(mask == self.gt_mapping["background"], self.dataset_mapping["background"], self.dataset_mapping["foreground"])
        return dataset_mask.type(mask.type())
    
    def gt_to_train_mask(self, mask: torch.Tensor):
        train_mask = torch.where(mask == self.gt_mapping["background"], self.train_mapping["background"], self.train_mapping["foreground"])
        return train_mask.type(mask.type())
    
    def get_logits(self, mask: torch.Tensor):
        # Mask is of shape (reps, N, 1, H, W) 
        # Output is of shape (reps,N, 1, H, W)
        return mask
    
    def get_segmentation(self, logits: torch.Tensor):
        # Logits are of shape (reps, N, 1, H, W)
        # Output is of shape ((N, 1, H, W), (N, 2, H, W))
        if self.ensemble_mode == "mean":
            logits = torch.mean(logits, dim=0)
            segmentation = self.threshold_func(logits)
        elif self.ensemble_mode == "median":
            logits = torch.median(logits, dim=0)
            segmentation = self.threshold_func(logits)
        elif self.ensemble_mode == "majority":
            segmentations_across_reps = self.threshold_func(logits).type(torch.int32)
            segmentation, _ = torch.mode(segmentations_across_reps, dim=0)
        else:
            raise ValueError(f"Ensemble mode {self.ensemble_mode} has not been implemented.")
        
        assert(len(segmentation.shape) == 4)
        
        one_hot_shape = (segmentation.shape[0], 2, segmentation.shape[2], segmentation.shape[3])

        return segmentation, torch.zeros(one_hot_shape, device=logits.device).scatter_(1, segmentation, 1)
        
    
    def gt_mapping_for_visualization(self):
        class_labels = {
            self.gt_mapping["background"]: "background",
            self.gt_mapping["foreground"]: "foreground"
        }

        offset = self.gt_mapping["background"] + self.gt_mapping["foreground"] + 1

        class_labels_pred = {
            (self.gt_mapping["background"] + offset): "background",
            (self.gt_mapping["foreground"] + offset): "foreground"
        }

        return {
            "class_labels": class_labels,
            "class_labels_pred": class_labels_pred,
            "offset": offset
        }
    
    def get_num_train_channels(self):
        return 1


def generate_mask_mapping(config: MaskTransformerConfig) -> BaseMaskMapping:
    if config.mask_type == "binary":
        return BinarySegMaskMapping(config.dataset_mapping, config.train_mapping, config.threshold)
    elif config.mask_type == "multi_class":
        return MultiClassSegMaskMapping(config.dataset_mapping, config.train_switch, config.prediction_type)
    else:
        raise ValueError(f"Maks type {config.mask_type} has not been implemented.")
