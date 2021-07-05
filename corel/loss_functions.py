import torch
import torch.nn as nn
from torch.nn.functional import log_softmax

# Helper functions for dealing with AR Loss
def get_armask(shape, labels, device=None):
    if labels.dtype != torch.int64:
        raise TypeError("Labels must be a LongTensor with dtype=int64!")

    mask = torch.zeros(shape).to(device)
    arr = torch.arange(0, shape[0]).long().to(device)

    # want to maximize similarity to the correct classes, so this is negative.
    mask[arr, labels] = -1.
    return mask


def arloss(attraction_tensor, repulsion_tensor, lam):
    # combine up everything to accumulate across the entire batch
    loss_attraction = attraction_tensor.sum()
    loss_repulsion = repulsion_tensor.sum()
    arloss = -(lam * loss_attraction) + ((1. - lam) * loss_repulsion)
    return arloss / attraction_tensor.shape[0]



# Cosine-COREL combined loss function
class CosineARLoss(nn.Module):

    def __init__(self, lam, device=None):
        super(CosineARLoss, self).__init__()
        self.lam = lam
        self.device = device


    def forward(self, predictions, labels):
        mask = get_armask(predictions.shape, labels, device=self.device)

        # make the attractor and repulsor, mask them!
        attraction_tensor = mask * predictions
        repulsion_tensor = (mask + 1) * predictions

        # now, apply the special cosine-COREL rules, taking the argmax and squaring the repulsion
        repulsion_tensor, _ = repulsion_tensor.max(dim=1)
        repulsion_tensor = repulsion_tensor ** 2

        return arloss(attraction_tensor, repulsion_tensor, self.lam)



# Gaussian-COREL combined loss function
class GaussianARLoss(nn.Module):

    def __init__(self, lam, device=None):
        super(GaussianARLoss, self).__init__()
        self.lam = lam
        self.device = device


    def forward(self, predictions, labels):
        mask = get_armask(predictions.shape, labels, device=self.device)

        # in this case, use standard LogSoftmax, without AR.
        if self.lam == 0.5:
            softmax_predictions = log_softmax(predictions, dim=1)
            loss_tensor = mask * softmax_predictions
            return loss_tensor.mean()

        # otherwise, do it in the slightly less numerically stable way.
        attraction_tensor = mask * predictions * self.lam
        repulsion_tensor = torch.exp(predictions)
        repulsion_tensor = torch.log(repulsion_tensor.sum(dim=1) + 1e-10) * (1. - self.lam)
        return arloss(attraction_tensor, repulsion_tensor, self.lam)

