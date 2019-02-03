import torch
import torch.nn as nn
from corel.loss_functions import CosineARLoss, GaussianARLoss
from corel.prediction_functions import get_cosine_predictions, get_gaussian_predictions



class CORELWrapper(nn.Module):

    def __init__(self, model, rep_dim, n_classes, lam=0.5, device=None, corel_option='gaussian'):
        """
        Generic class for a COREL wrapper.
        :param model: a torch.nn module (e.g., your CNN) that projects up to the final hidden layer
        :param rep_dim: dimensionality of the final hidden layer the model projects to
        :param n_classes: number of classes for the classification task
        :param lam: float representing the attraction-repulsion parameter for AR-loss
        :param device: must be a torch.device object, e.g., 'cuda'
        :param corel_option: string, either 'gaussian' or 'cosine', for now.
        """
        assert(0.0 <= lam <= 1.0)

        # construct the bare necessities
        super(CORELWrapper, self).__init__()
        self.model = model
        self.lam = lam
        self.device = device
        self.corel_option = corel_option
        self.output_shape = (rep_dim, n_classes,)
        self.W = nn.Parameter(torch.Tensor(*self.output_shape))

        # define the loss constructor and the output predictor
        self.gamma = None
        self._loss_constructor = None
        self._prediction_function = None

        if corel_option == 'gaussian':
            self.gamma = nn.Parameter(torch.Tensor(1)) # learn gamma, no longer hparam
            self._loss_constructor = GaussianARLoss
            self._prediction_function = get_gaussian_predictions

        elif corel_option == 'cosine':
            self._loss_constructor = CosineARLoss
            self._prediction_function = get_cosine_predictions

        else:
            raise NotImplementedError(f'COREL option: \"{corel_option}\" is not implemented!')

        self.reset_parameters()


    def reset_parameters(self):
        self.W.data = nn.init.xavier_normal_(torch.Tensor(*self.output_shape))
        if self.corel_option == 'gaussian':
            self.gamma.data[0] = 0.5


    def forward(self, X, *args):
        hidden_reps = self.model(X, *args)
        predictions = self._prediction_function(hidden_reps, self.W, gamma=self.gamma, device=self.device)
        return predictions


    def get_output_weights(self):
        return self.W


    def get_loss_function(self):
        return self._loss_constructor(self.lam, self.device)
