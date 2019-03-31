import torch
import torch.nn as nn
import torch.nn.functional as f


def get_cce_predictions(hidden_reps, output_weights, bias=None, **kwargs):
    M = f.linear(hidden_reps, output_weights.transpose(0, 1), bias=bias)
    return M


def get_cosine_predictions(hidden_reps, output_weights, **kwargs):
    # very easy to get the cosine similarity predictions!
    X = f.normalize(hidden_reps, p=2, dim=1) # normalize the rows!
    W = f.normalize(output_weights, p=2, dim=0) # normalize the weights!
    M = X.mm(W) # multiply matrices, M is now the matrix of cossims
    return M


def get_gaussian_predictions(hidden_reps, output_weights, gamma=0.5, device=None, **kwargs):
    # useful constants for later
    N, K = hidden_reps.shape[0], output_weights.shape[1]

    # recall the definition of squared l2 norm.
    # || x - w ||^2 = ||x||^2 + ||w||^2 - 2(x * w)
    xn = hidden_reps.norm(dim=1) ** 2  # vector of size N, ||hi|| ** 2
    wn = output_weights.norm(dim=0) ** 2   # vector of size K, ||uj|| ** 2

    # adding up the norms to get M_ij == ||hi||**2 + ||cj||**2
    ones = torch.ones(N, K).to(device)
    rows_xnorms = xn.view(N, 1) * ones
    cols_cnorms = ones * wn.view(1, K)
    M = rows_xnorms + cols_cnorms

    # getting the dot products
    dots = hidden_reps.mm(output_weights)

    # squared euclidean distance, augmented by gamma
    M = - gamma * (M - (2. * dots))
    return M

