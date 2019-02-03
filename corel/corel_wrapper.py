import torch
import torch.nn as nn
import corel.prediction_functions as cpred
from corel.loss_functions import CosineARLoss, GaussianARLoss

def xavier(shape):
    return nn.init.xavier_normal_(torch.Tensor(*shape))


class CORELWrapper(nn.Module):

    def __init__(self, model, rep_dim, n_classes, lam=0.5, device=None, corel_option='gaussian'):
        """
        Generic class for a COREL wrapper. There are three options, as discussed in the paper:
        Cosine, Gaussian, and we also integrate CCE into this for easy comparison.

        :param model: a torch.nn module (e.g., your CNN) that projects up to the final hidden layer
        :param rep_dim: dimensionality of the final hidden layer the model projects to
        :param n_classes: number of classes for the classification task
        :param lam: float representing the attraction-repulsion parameter for AR-loss
        :param device: must be a torch.device object, e.g., 'cuda'
        :param corel_option: string, either 'gaussian' or 'cosine', for now.
        """
        assert(0.0 <= lam <= 1.0)
        if lam != 0.5 and corel_option == 'cce':
            print('(weak warning): CCE will have no impact from lambda in this implementation!')

        # construct the bare necessities
        super(CORELWrapper, self).__init__()
        self.model = model
        self.lam = lam
        self.device = device
        self.corel_option = corel_option
        self.output_shape = (rep_dim, n_classes,)
        self.W = nn.Parameter(torch.Tensor(*self.output_shape))

        # define the loss constructor and the output predictor
        self.gamma = None # for gaussian
        self.bias = None # for CCE
        self._loss_constructor = None
        self._prediction_function = None

        if corel_option == 'gaussian':
            self.gamma = nn.Parameter(torch.Tensor(1)) # learn gamma, no longer hparam
            self._loss_constructor = GaussianARLoss
            self._prediction_function = cpred.get_gaussian_predictions

        elif corel_option == 'cosine':
            self._loss_constructor = CosineARLoss
            self._prediction_function = cpred.get_cosine_predictions

        elif corel_option == 'cce':
            self._loss_constructor = nn.CrossEntropyLoss
            self._prediction_function = cpred.get_cce_predictions
            self.bias = nn.Parameter(torch.Tensor(1, n_classes))

        else:
            raise NotImplementedError(f'COREL option: \"{corel_option}\" is not implemented!')

        self.reset_parameters()


    def reset_parameters(self):
        self.W.data = xavier(self.output_shape)

        if self.corel_option == 'gaussian':
            self.gamma.data[0] = 0.5

        elif self.corel_option == 'cce':
            self.bias.data = xavier((1, self.output_shape[1]))


    def forward(self, X, *args, **kwargs):
        hidden_reps = self.get_representations(X, *args, **kwargs)
        predictions = self._prediction_function(hidden_reps, self.W,
                                                bias=self.bias,
                                                gamma=self.gamma,
                                                device=self.device)
        return predictions


    def get_representations(self, X, *args, **kwargs):
        """
        Project the data up to the representation layer (maybe you want to do
        clustering on these representations!)
        """
        return self.model(X, *args, **kwargs)


    def get_loss_function(self):
        if self.corel_option == 'cce':
            return self._loss_constructor()
        return self._loss_constructor(self.lam, self.device)
