import torch
import gudhi


def extract_persistence_intervals(batch: torch.Tensor, min_persistence: float):
    """
    Extracts the persistence intervals from a batch of images (segmentation predictions). Assumes that the values are in the range [0, 1].
    params:
        batch: torch.Tensor, shape (batch_size, height, width) 
        min_persistence: float >= 0.0, the minimum persistence value to be considered
    returns:
        list[list[tuple[int, tuple[int, int]]]], a list of persistence intervals for each image in the batch
    """
    
    return [gudhi.CubicalComplex(top_dimensional_cells=batch[i]).persistence(min_persistence=min_persistence) for i in range(batch.shape[0])]


def extract_birth_values(persistence_intervals: list[list[tuple[int, tuple[int, int]]]], dimension: int, num_considered_intervals: int):
    """
    Extracts the birth values from the <num_considered_intervals> largest persistence intervals of the given dimension.
    """

    return [[feature[1][0] for feature in persistence_intervals[i][dimension] if feature[0] == dimension][:num_considered_intervals] for i in range(len(persistence_intervals))]



def create_filtered_batch(batch: torch.Tensor, topo_features: dict, min_persistence: float):
    """
    params:
        batch: torch.Tensor, shape (batch_size, num_classes, height, width) or (ensemble_size, batch_size, num_classes, height, width)
        topo_features: dict, a dictionary containing the number of features for each dimension
    returns:
        torch.Tensor, shape (batch_size, num_classes, height, width)
    """

    if len(batch.shape) == 5:
        batch = torch.mean(batch, dim=0)

    num_fg_classes = batch.shape[1] - 1
    check_topofeatures(topo_features, num_fg_classes)
    filter_mask = torch.zeros_like(batch)

    for class_idx in range(1, num_fg_classes + 1):
        persistence_invervals = extract_persistence_intervals(batch[:, class_idx, :, :], min_persistence=min_persistence)
        if topo_features[class_idx][0] > 0:
            birth_values_0 = extract_birth_values(persistence_invervals, topo_features[class_idx][0], topo_features[class_idx][1])


    pass



def check_topofeatures(topo_features: dict, num_classes: int):
    """
    Check if the topo_features are valid.
    """
    
    if len(topo_features) != num_classes:
        raise ValueError(f"Expected {num_classes} topo_features definitions, but got {len(topo_features)}")
    
    idx_list = [i+1 for i in range(num_classes)]
    
    for class_idx, topo_feature in topo_features.items():
        if not isinstance(topo_feature, dict):
            raise ValueError(f"Topo feature for class {class_idx} is not a dictionary")
        if class_idx in idx_list:
            idx_list.remove(class_idx)
        else:
            raise ValueError(f"Topo feature for class {class_idx} is not in the idx list of the classes")
        
        if 0 in topo_feature.keys() and type(topo_feature[0]) == int and topo_feature[0] >= 0:
            if 1 in topo_feature.keys() and type(topo_feature[1]) == int and topo_feature[1] >= 0:
                continue
            else:
                raise ValueError(f"Topo feature for class {class_idx} does not contain homology dimension for class 1")
        else:
            raise ValueError(f"Topo feature for class {class_idx} does not contain homology dimension for class 0")
        
    
    return True