"""
This code file provides a simple example of how to use the API.
To minimize the length, we will simply use synthetic data.
"""
import torch
import torch.nn as nn
import torch.nn.functional as f
from corel.corel_wrapper import CORELWrapper

# first, let's define a super basic FFNN as a representation builder.
# Input -> 64 -> 32 dimensional hidden representations
class RepBuilderFFNN(nn.Module):
    def __init__(self, input_dim, rep_dim):
        super(RepBuilderFFNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, rep_dim)

    def forward(self, X):
        X = f.relu(self.layer1(X))
        X = f.relu(self.layer2(X))
        return X


# Now let's do the example!
def main():
    torch.random.manual_seed(1917)

    # params for synthetic data
    N_epochs = 100
    N_classes = 10
    N_samples = 500
    D_input = 16 # dimensionality of input samples

    # model params
    D_reps = 32 # dimensionality of the hidden representations
    device = torch.device('cpu')

    # build your original model (which does not have an "output_layer")
    representation_builder = RepBuilderFFNN(D_input, D_reps).to(device)

    # build the full model, as a wrapper over the original
    full_model = CORELWrapper(representation_builder,
                              D_reps,
                              N_classes,
                              lam=0.85,
                              device=device,
                              corel_option='gaussian').to(device)

    # get the loss function for this model (how convenient!!)
    loss_function = full_model.get_loss_function()

    # define the optimizer
    optimizer = torch.optim.Adam(full_model.parameters(), lr=0.001)

    # make the synthetic data
    X_data = (torch.rand(N_samples, D_input) * 10.).to(device)
    labels = torch.LongTensor([i % N_classes for i in range(N_samples)]).to(device)

    # make a training loop doing full training batches
    for i in range(N_epochs):
        full_model.train()
        predictions = full_model(X_data)
        loss = loss_function(predictions, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch {:3}:   {}'.format(i, loss.item()))

    print('\nFinished training!')
    if full_model.corel_option == 'gaussian':
        gamma = full_model.gamma.item()
        var = 0.5 * (1 / gamma)
        std = var ** 0.5

        print('Interestingly, we have learned the gamma parameter to be:')
        print('\tgamma = {}'.format(gamma))
        print('This means that the standard deviation and variance of the latent data is:')
        print('\tsigma   = {}'.format(std))
        print('\tsigma^2 = {}'.format(var))


if __name__ == '__main__':
    main()


