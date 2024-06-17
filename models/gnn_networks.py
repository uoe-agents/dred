# Copyright (c) 2022-2024 Samuel Garcin
# 
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT

"""
Adapted from https://github.com/dmlc/dgl/blob/master/examples/mxnet/gin/gin.py

Original paper:
How Powerful are Graph Neural Networks
https://arxiv.org/abs/1810.00826
https://openreview.net/forum?id=ryGs6iA5Km
Author's implementation: https://github.com/weihua916/powerful-gnns
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GINConv

class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""
    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h


class MLP(nn.Module):
    """MLP with linear output"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, enable_batch_norm):
        """MLP layers construction
        Paramters
        ---------
        num_layers: int
            The number of linear layers
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        """
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.enable_batch_norm = enable_batch_norm

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()

            if self.enable_batch_norm:
                self.batch_norms = torch.nn.ModuleList()
            else:
                self.batch_norms = None

            self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))

            if self.enable_batch_norm:
                for layer in range(num_layers - 1):
                    self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = self.linears[i](h)
                if self.enable_batch_norm:
                    h = self.batch_norms[i](h)
                h = F.relu(h)
            return self.linears[-1](h)


class GINEncoderNoPooling(nn.Module):
    """GINEncoderNoPooling model"""
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, learn_eps,
                 neighbor_pooling_type, n_nodes, enable_batch_norm):
        """model parameters setting
        Paramters
        ---------
        num_layers: int
            The number of linear layers in the neural network
        num_mlp_layers: int
            The number of linear layers in mlps
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        final_dropout: float
            dropout ratio on the final linear layer
        learn_eps: boolean
            If True, learn epsilon to distinguish center nodes from neighbors
            If False, aggregate neighbors and center nodes altogether.
        neighbor_pooling_type: str
            how to aggregate neighbors (sum, mean, or max)
        """
        super(GINEncoderNoPooling, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps
        self.enable_batch_norm = enable_batch_norm

        # real output dim
        self.input_dim = input_dim
        self.output_dim = ((num_layers - 1) * hidden_dim + input_dim)*n_nodes
        self.hidden_dim = hidden_dim

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        if self.enable_batch_norm:
            self.batch_norms = torch.nn.ModuleList()
        else:
            self.batch_norms = None

        for layer in range(self.num_layers - 1):
            if layer == 0:
                mlp = MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim, self.enable_batch_norm)
            else:
                mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim, self.enable_batch_norm)

            self.ginlayers.append(
                GINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps))
            if self.enable_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(
                    nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(
                    nn.Linear(hidden_dim, output_dim))

        self.drop = nn.Dropout(final_dropout)

    def forward(self, g, h):
        # list of hidden representation at each layer (including input)
        hidden_rep = [h]

        for i in range(self.num_layers - 1):
            h = self.ginlayers[i](g, h)
            if self.enable_batch_norm:
                h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)

        output = []
        # perform pooling over all nodes in each graph in every layer
        for i, h in enumerate(hidden_rep):
            output.append(h.reshape(g.batch_size, -1, h.shape[-1]))
        output = torch.cat(output, -1)

        return output
