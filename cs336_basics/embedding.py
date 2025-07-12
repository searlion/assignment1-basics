import math

import torch
from torch import nn


class Embedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        factory_kwargs = {'device': device, 'dtype': dtype}

        # Calculate standard deviation (sigma) based on the formula:
        # variance = 2 / (d_in + d_out)
        variance = 2 / (self.num_embeddings + self.embedding_dim)
        std_dev = math.sqrt(variance)

        # The distribution is truncated at [-3*sigma, 3*sigma]
        lower_bound = -3 * std_dev
        upper_bound = 3 * std_dev

        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))

        # Initialize the weights using truncated normal distribution with the calculated params.
        # The underscore signifies an in-place operation.
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std_dev,
            a=lower_bound,
            b=upper_bound
        )

    def forward(self, token_ids: torch.Tensor):
        return self.weight[token_ids]