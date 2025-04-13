import torch
from topologylayer.nn import (
    LevelSetLayer2D,
)
from topologylayer.nn.features import get_barcode_lengths


class PartialSumBarcodeLengthsFlexibleSkip(torch.nn.Module):
    """
    PartialSumBarcodeLengths with flexible skip takes an additional argument in the forward pass instead of having a fixed skip value initialized in the constructor like in the original PartialSumBarcodeLengths.
    """

    def __init__(self, dim, power=1):
        super().__init__()
        self.dim = dim
        self.power = power

    def forward(self, interval_info, skip):
        intervals, issublevel = interval_info

        lengths = get_barcode_lengths(intervals[self.dim], issublevel)
        sortl, indl = torch.sort(lengths, descending=True)

        return torch.sum(torch.pow(sortl[skip:], self.power))


class TopoLoss(torch.nn.Module):
    """
    TopoLoss is inspired by the TopLoss proposed in the paper "A Topology Layer for Machine Learning" by Brüel-Gabrielsson et al.
    It sums the lengths of the components and cycles of the persistence diagram, while ignoring the largest ones.
    """

    def __init__(
        self, alpha, average="sample_class", sublevel=False, size=None, with_cycles=True
    ):
        super().__init__()

        if size is None:
            self.persistence_layer = None
        else:
            self.set_up_persistence_layer(size, sublevel)

        self.with_cycles = with_cycles

        self.comp_lengths = PartialSumBarcodeLengthsFlexibleSkip(dim=0)
        if self.with_cycles:
            self.cycle_lengths = PartialSumBarcodeLengthsFlexibleSkip(dim=1)
        self.sublevel = sublevel
        self.alpha = alpha
        self.average = average

    def set_up_persistence_layer(self, size):
        if self.persistence_layer is None:
            self.persistence_layer = LevelSetLayer2D(size=size, sublevel=self.sublevel)

    def forward(
        self,
        prediction: torch.Tensor,
        betti_0: torch.Tensor,
        betti_1: torch.Tensor = None,
    ):
        """
        params:
            prediction: torch.Tensor, shape (batch_size, num_classes, height, width)
            betti_0: torch.Tensor, shape (batch_size, num_classes)
            betti_1: torch.Tensor, shape (batch_size, num_classes)
        """

        comp_loss = 0.0
        cycle_loss = 0.0
        for sample_idx in range(prediction.shape[0]):
            for class_idx in range(prediction.shape[1]):
                interval_info = self.persistence_layer(
                    prediction[sample_idx, class_idx]
                )

                comp_loss += self.comp_lengths(
                    interval_info, betti_0[sample_idx, class_idx]
                )
                if self.with_cycles:
                    cycle_loss += self.cycle_lengths(
                        interval_info, betti_1[sample_idx, class_idx]
                    )

        if self.average == "sample_class":
            comp_loss /= prediction.shape[0] * prediction.shape[1]
            cycle_loss /= prediction.shape[0] * prediction.shape[1]
        elif self.average == "sample":
            comp_loss /= prediction.shape[0]
            cycle_loss /= prediction.shape[0]

        return self.alpha * comp_loss + (1 - self.alpha) * cycle_loss


class TopoLoss_0(TopoLoss):
    def __init__(self, alpha, average="sample_class", sublevel=False, size=None):
        super().__init__(1.0, average, sublevel, size, with_cycles=False)
