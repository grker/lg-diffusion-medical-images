import torch


def likelyhood_temperature_scaling(
    x_softmax: torch.Tensor, likelyhood: torch.Tensor, alpha: float = 1.0
):
    """
    Likelihood temperature scaling of the softmax output.
    params:
        x_softmax: torch.Tensor, shape (batch_size, num_classes, height, width)
        likelyhood: torch.Tensor, shape (batch_size, num_classes, height, width)
        alpha: float, between 0 and 1, default 1.0
    returns:
        torch.Tensor, shape (batch_size, num_classes, height, width)
    """
    # second part does not make sense !!
    x_softmax = x_softmax / (1 - likelyhood) * alpha + torch.softmax(
        x_softmax, dim=1
    ) * (1 - alpha)

    return x_softmax / torch.sum(x_softmax, dim=1, keepdim=True)
