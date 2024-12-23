import torch


def alpha_smoothing_uniform(x_softmax: torch.Tensor, alpha:int=0.5):
    """
        Simple smoothing of the softmax output. 
        params:
            x_softmax: torch.Tensor, shape (num_classes, height, width)
            alpha: float, between 0 and 1, default 0.5
        returns:
            torch.Tensor, shape (num_classes, height, width)
    """
    uniform_dist_value = 1 / x_softmax.shape[0]
    return x_softmax * alpha + (1-alpha) * uniform_dist_value


def alpha_smoothing(x_softmax: torch.Tensor, alpha:int=0.5):
    """
        Smoothing of the softmax output. 
        params:
            x_softmax: torch.Tensor, shape (num_classes, height, width)
            alpha: float, between 0 and 1, default 0.5
        returns:
            torch.Tensor, shape (num_classes, height, width)
    """
    num_classes = x_softmax.shape[0]
    max_values, max_indices = torch.max(x_softmax, dim=0)
    max_values_one_hot = alpha * torch.zeros_like(x_softmax).scatter_(0, max_indices.unsqueeze(0), max_values.unsqueeze(0))

    return x_softmax - max_values_one_hot + (torch.sum(max_values_one_hot, dim=0)/num_classes).unsqueeze(0)
    

def laplace_smoothing(x_softmax: torch.Tensor, alpha: float=1.0, neighborhood: list[torch.Tensor]=None):
    """
        Laplace smoothing of the softmax output. 
        params:
            x_softmax: torch.Tensor, shape (num_classes, height, width)
            neighborhood: list[torch.Tensor], len(neighborhood) = num_classes
        returns:
            torch.Tensor, shape (num_classes, height, width)
    """
    num_classes = x_softmax.shape[0]
    neighborhood_mean = torch.zeros_like(x_softmax)
    
    class_list = torch.arange(num_classes)
    for class_idx in range(num_classes):
        neighbors = neighborhood[class_idx] if neighborhood is not None else class_list[class_idx != class_idx]
        neighborhood_mean[class_idx] = torch.mean(x_softmax[neighbors], dim=0)

    return torch.softmax(neighborhood_mean * alpha + (1-alpha) * x_softmax, dim=0)



def power_scaling(x_softmax: torch.Tensor, alpha: float=1.0):
    """
        Sharpening of the softmax output. 
        params:
            x_softmax: torch.Tensor, shape (num_classes, height, width)
            alpha: float, >= 1
        returns:
            torch.Tensor, shape (num_classes, height, width)
    """
    x_softmax = x_softmax.pow(alpha)
    return x_softmax / torch.sum(x_softmax, dim=0)

def temperature_scaling(x_softmax: torch.Tensor, temperature: float=0.2):
    """
        Temperature scaling of the softmax output. 
        params:
            x_softmax: torch.Tensor, shape (num_classes, height, width)
            temperature: float, < 1
        returns:
            torch.Tensor, shape (num_classes, height, width)
    """
    return torch.softmax(x_softmax / temperature, dim=0)




class AdjustProbs:
    def __init__(self, alpha_smoothing: float, alpha_sharpening: float, smoothing_function: str, sharpening_function: str, neighborhood: list[torch.Tensor]=None):
        self.set_smoothing_function(smoothing_function, alpha_smoothing, neighborhood)
        self.set_sharpening_function(sharpening_function, alpha_sharpening)

    def set_smoothing_function(self, smoothing_function: str, alpha_smoothing: float, neighborhood: list[torch.Tensor]=None):
        if smoothing_function == "alpha_smoothing":
            self.smoothing = lambda x: alpha_smoothing(x, alpha_smoothing)
        elif smoothing_function == "alpha_smoothing_uniform":
            self.smoothing = lambda x: alpha_smoothing_uniform(x, alpha_smoothing)
        elif smoothing_function == "laplace_smoothing":
            self.smoothing = lambda x: laplace_smoothing(x, alpha_smoothing, neighborhood)
        elif smoothing_function == "power_scaling":
            self.smoothing = lambda x: power_scaling(x, alpha_smoothing)
        else:
            raise ValueError(f"Invalid smoothing function: {smoothing_function}. Choose from: 'alpha_smoothing', 'alpha_smoothing_uniform', 'laplace_smoothing', 'power_scaling'")


    def set_sharpening_function(self, sharpening_function: str, alpha_sharpening: float):
        if sharpening_function == "power_scaling":
            self.sharpening = lambda x: power_scaling(x, alpha_sharpening)
        elif sharpening_function == "temperature_scaling":
            self.sharpening = lambda x: temperature_scaling(x, alpha_sharpening)
        else:
            raise ValueError(f"Invalid sharpening function: {sharpening_function}. Choose from: 'power_scaling', 'temperature_scaling'")

    def __call__(self, x_softmax: torch.Tensor, scale_map: torch.Tensor):
        if self.smoothing is None or self.sharpening is None:
            raise ValueError("Smoothing or sharpening function not set. Call set_smoothing_function and set_sharpening_function before calling __call__.")
        return self.adjust_probs(x_softmax, scale_map)
    

    def adjust_probs(self, x_softmax: torch.Tensor, scale_map: torch.Tensor):
        """
            Adjust the probabilities of the softmax output. 
            params:
                x_softmax: torch.Tensor, shape (num_classes, height, width)
                scale_map: torch.Tensor, shape (height, width)
            returns:
                torch.Tensor, shape (num_classes, height, width)
        """
        
        scale_map = scale_map.unsqueeze(0).expand(x_softmax.shape, -1, -1)
        x_smoothing = self.smoothing(x_softmax, scale_map)
        x_sharpening = self.sharpening(x_smoothing, scale_map)

        return scale_map * x_sharpening + (1-scale_map) * x_smoothing
    
