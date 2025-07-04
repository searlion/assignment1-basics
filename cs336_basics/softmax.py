import torch
from einops import reduce


def softmax(x: torch.Tensor, dim: int):
    """
    Applies the softmax function to a tensor along a specified dimension
    using the einops library.

    Uses the max-subtraction trick for numerical stability.

    Args:
        x (torch.Tensor): The input tensor.
        dim (int): The dimension along which to apply softmax.

    Returns:
        torch.Tensor: The output tensor with the same shape as the input.
    """
    # Handle negative dimension indexing, just like PyTorch
    if dim < 0:
        dim += x.ndim

    # Dynamically create the einops pattern string.
    # e.g., for a 4D tensor and dim=1, this creates:
    # 'd0 d1 d2 d3 -> d0 1 d2 d3'
    rank = x.ndim
    axis_names = [f'd{i}' for i in range(rank)]
    from_pattern = ' '.join(axis_names)
    to_pattern_list = list(axis_names)
    to_pattern_list[dim] = '1' # This is the key: we specify the target dim becomes size 1
    to_pattern = ' '.join(to_pattern_list)
    reduction_pattern = f'{from_pattern} -> {to_pattern}'

    # 1. Subtract the maximum value for numerical stability.
    # The pattern tells einops to reduce along the specified axis, keeping it as a
    # dimension of size 1, which is perfect for broadcasting.
    max_vals = reduce(x, reduction_pattern, 'max')
    x_centered = x - max_vals

    # 2. Exponentiate
    exps = torch.exp(x_centered)

    # 3. Sum the exponentiated values along the same dimension.
    sum_exps = reduce(exps, reduction_pattern, 'sum')

    # 4. Divide to get the final probabilities. Broadcasting works automatically.
    return exps / sum_exps
