import torch
import gudhi

import gudhi


class PersistentHomologyLoss(torch.nn.Module):
    def __init__(self, config: dict):
        super(PersistentHomologyLoss, self).__init__()
        self.config = config

        self.check_topofeatures(self.config["topo_features"], self.config["num_classes"])



    def check_topofeatures(self, topo_features: dict, num_classes: int):
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
            
        
                
                

