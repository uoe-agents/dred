# Copyright (c) 2017 Roberta Raileanu
# 
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# This file is a modified version of:
# https://github.com/rraileanu/auto-drac/blob/master/ucb_rl2_meta/model.py
# 
# evaluate_logprob_continuous_bernoulli(), evaluate_logprob_bernoulli(),
# evaluate_logprob_one_hot_categorical(), evaluate_logprob_multinomial(),
# evaluate_logprob_diagonal_gaussian(), compute_kld_with_standard_gaussian(),
# sample_gaussian_without_reparametrisation(), sample_gaussian_with_reparametrisation() by Samuel Garcin

import math

import torch
import torch.nn as nn

from .common import init

class FixedCategorical(torch.distributions.Categorical):
    """
    Categorical distribution object
    """
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class Categorical(nn.Module):   
    """
    Categorical distribution (NN module)
    """
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)


def evaluate_logprob_continuous_bernoulli(X, *, logits):
    """
    Evaluates log-probability of the continuous Bernoulli distribution

    Args:
        X (Tensor):      data, a batch of shape (B, D)
        logits (Tensor): parameters of the continuous Bernoulli,
                         a batch of shape (B, D)

    Returns:
        logpx (Tensor): log-probabilities of the inputs X, a batch of shape (B,)
    """
    cb = torch.distributions.ContinuousBernoulli(logits=logits)
    return cb.log_prob(X).sum(dim=-1)


def evaluate_logprob_bernoulli(X, *, logits):
    """
    Evaluates log-probability of the continuous Bernoulli distribution

    Args:
        X (Tensor):      data, a batch of shape (B, D)
        logits (Tensor): parameters of the continuous Bernoulli,
                         a batch of shape (B, D)

    Returns:
        logpx (Tensor): log-probabilities of the inputs X, a batch of shape (B,)
    """
    cb = torch.distributions.Bernoulli(logits=logits)
    return cb.log_prob(X).mean(dim=-1)


def evaluate_logprob_one_hot_categorical(X, *, logits):
    """
    Evaluates log-probability of the continuous Bernoulli distribution

    Args:
        X (Tensor):      data, a batch of shape (B, C), entries are one-hot vectors representing category labels
        logits (Tensor): parameters of the C-categories OneHotCategorical distribution,
                         a batch of shape (B, C)

    Returns:
        logpx (Tensor): log-probabilities of the inputs X, a batch of shape (B,)
    """
    cb = torch.distributions.OneHotCategorical(logits=logits)
    return cb.log_prob(X)


def evaluate_logprob_multinomial(X, *, logits):
    """
    Evaluates log-probability of the continuous Bernoulli distribution

    Args:
        X (Tensor):      data, a batch of shape (B, C), entries are floats representing category probabilities
        logits (Tensor): parameters of the C-categories OneHotCategorical distribution,
                         a batch of shape (B, C)

    Returns:
        logpx (Tensor): log-probabilities of the inputs X, a batch of shape (B,)
    """
    return torch.distributions.Multinomial(logits=logits).log_prob(X)


def evaluate_logprob_diagonal_gaussian(Z, *, mean, std):
    """
    Evaluates log-probability of the diagonal Gaussian distribution

    Args:
        Z (Tensor):      latent vectors, a batch of shape (*, B, H)
        mean (Tensor):   mean of diagonal Gaussian, a batch of shape (*, B, H)
        std (Tensor):    std of diagonal Gaussian, a batch of shape (*, B, H)

    Returns:
        logqz (Tensor): log-probabilities of the inputs Z, a batch of shape (*, B)
                        where `*` corresponds to any additional dimensions of the input arguments,
                        for example a dimension representing the samples used to approximate
                        the expectation in the ELBO
    """
    gauss = torch.distributions.Normal(loc=mean, scale=std)
    return gauss.log_prob(Z).sum(dim=-1)


def compute_kld_with_standard_gaussian(q_mean, q_std):
    """
    Computes KL(q(z|x)||p(z)) between the variational diagonal
    Gaussian distribution q and standard Gaussian prior p

    Args:
        q_mean (Tensor):   mean of diagonal Gaussian q, a batch of shape (B, H)
        q_std (Tensor):    std of diagonal Gaussian q, a batch of shape (B, H)

    Returns:
        kld (Tensor): KL divergences between q and p, a batch of shape (B,)
    """

    q_var = q_std**2
    q_logvar = q_var.log()

    kld = -0.5 * (1 + q_logvar - q_mean ** 2 - q_var).mean(dim=-1)

    return kld


def sample_gaussian_with_reparametrisation(mean, std, *, num_samples=1):
    """
    Samples the Gaussian distribution using the reparametrisation trick

    Args:
        mean (Tensor):     mean of diagonal Gaussian q, a batch of shape (B, K)
        std (Tensor):      std of diagonal Gaussian q, a batch of shape (B, K)
        num_samples (int): The number of samples (M) to approximate the expectation in the ELBO

    Returns:
        Z (Tensor):        Samples Z from the diagonal Gaussian q, a batch of shape (num_samples, B, K)
    """

    eps = torch.randn(num_samples, *mean.shape, dtype=std.dtype, device=std.device)

    Z = mean + eps * std

    # Making sure that the variational samples are on the same device as the input argument (i.e. CPU or GPU)
    return Z.to(device=std.device)


def sample_gaussian_without_reparametrisation(mean, std, *, num_samples=1):
    """
    Samples the Gaussian distribution without attaching then to the computation graph

    Args:
        mean (Tensor):   mean of diagonal Gaussian q, a batch of shape (B, K)
        std (Tensor): std of diagonal Gaussian q, a batch of shape (B, K)
        num_samples (int): The number of samples (M) to approximate the expectation in the ELBO

    Returns:
        Z (Tensor):      Samples Z from the diagonal Gaussian q, a batch of shape (num_samples, B, K)
    """
    return sample_gaussian_with_reparametrisation(mean, std, num_samples=num_samples).detach()