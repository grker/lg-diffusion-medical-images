import torch
import gudhi

from utils.hydra_config import PersistanceHomologyConfig

class PersistentHomologyLoss(torch.nn.Module):
    def __init__(self, config: PersistanceHomologyConfig):
        super(PersistentHomologyLoss, self).__init__()
        self.num_classes = config["num_classes"]
        self.topo_features = self.check_topofeatures(config["topo_features"], self.num_classes)
        self.min_persistance = config["min_persistance"]

        self.train_switch = config["train_switch"]


    def forward(self, x: torch.Tensor):
        assert(len(x.shape) == 5, f"Tensor should be a 5 dimensional tensor (batch, classes, channels, height, width)")
        assert(x.shape[1] != self.num_classes, f"Shape does not match. Tensor has size {x.shape[1]} in dimension 1, but is expected to have {self.num_classes}.")

        x = x.squeeze(0)
        if not self.train_switch:
            x = torch.clamp(1 - x, 0.0, 1.0)

        loss = 0.0
        print(f"x shape in topo loss: {x.shape}")
        for batch_idx in range(x.shape[0]):
            for class_idx in range(x.shape[1]-1): #background can be ignored
                cp = gudhi.CubicalComplex(top_dimensional_cells=x[:, class_idx+1, :, :])
                persistence_list = cp.persistence()

                print(f"Persistence list: {persistence_list}")

                loss += self.loss_per_dimension([feature[1] for feature in persistence_list if feature[0] == 0], self.topo_features[class_idx+1][0]) #homology dimension 0
                loss += self.loss_per_dimension([feature[1] for feature in persistence_list if feature[0] == 1], self.topo_features[class_idx+1][1]) #homology dimension 1

        return loss / x.shape[0]
            

    def loss_per_dimension(self, pers_list: list, num_features: int):
        
        def persistence(birth_death_pair: tuple[int, int]):
            if birth_death_pair[1] > 1:
                return 1 - birth_death_pair[0]
            return birth_death_pair[1] - birth_death_pair[0]
        
        list_length = len(pers_list)
        loss = 0.0

        for i in range(num_features):
            if i < list_length:
                loss += (1 - persistence(pers_list[i])) ** 2
            else:
                loss += 1

        rest = num_features
        while rest < list_length:
            loss += persistence(pers_list[rest]) ** 2
            rest += 1

        print(f"Loss: {loss}")
        
        return loss
        


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
            
        
        return topo_features
                

