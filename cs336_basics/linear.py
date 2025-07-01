import math

import torch
from torch import nn

class Linear(nn.Module):
    """
    Constructs a linear transformation module.

    This module applies a linear transformation to the incoming data: y = xW^T.
    """
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """
        Initializes the Linear module.

        Args:
            in_features (int): Final dimension of the input tensor.
            out_features (int): Final dimension of the output tensor.
            device (torch.device | None, optional): Device to store parameters on. Defaults to None.
            dtype (torch.dtype | None, optional): Data type of the parameters. Defaults to None.
        """
        # Call the superclass constructor
        super().__init__()

        # Store dimensions for reference
        self.in_features = in_features
        self.out_features = out_features

        # A dictionary to hold keyword arguments for tensor creation
        factory_kwargs = {'device': device, 'dtype': dtype}

        # Construct the parameter as W, not W^T.
        # For an input x of shape (N, ..., in_features), the operation is x @ W.T
        # where W.T has shape (in_features, out_features).
        # Therefore, W must have the shape (out_features, in_features).
        weight = torch.empty((out_features, in_features), **factory_kwargs)

        # Calculate standard deviation (sigma) based on the formula:
        # variance = 2 / (d_in + d_out)
        variance = 2 / (self.in_features + self.out_features)
        std_dev = math.sqrt(variance)

        # The distribution is truncated at [-3*sigma, 3*sigma]
        lower_bound = -3 * std_dev
        upper_bound = 3 * std_dev

        # Initialize the weights using truncated normal distribution with the calculated params.
        # The underscore signifies an in-place operation.
        nn.init.trunc_normal_(
            weight,
            mean=0.0,
            std=std_dev,
            a=lower_bound,
            b=upper_bound
        )

        # Wrap the tensor in nn.Parameter to register it as a trainable parameter
        self.W = nn.Parameter(weight)


    def forward(self, x: torch.Tensor):
        """
        Applies the linear transformation to the input.

        Args:
            x (torch.Tensor): The input tensor of shape (N, ..., in_features).

        Returns:
            torch.Tensor: The output tensor of shape (N, ..., out_features).
        """
        # Apply the linear transformation using matrix multiplication.
        # We use the transpose of W to get the correct dimensions for the multiplication.
        return torch.einsum("...i, oi -> ...o", x, self.W)
