import torch
from einops import reduce, rearrange
from torch import nn


class RMSNorm(nn.Module):

    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        # A dictionary to hold keyword arguments for tensor creation
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.d_model = d_model
        self.g = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor):
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # The pattern '... d -> ...' means:
        # - Take a tensor with any number of leading dimensions (...) and a final dimension (d)
        # - Reduce the final dimension (d) away, keeping the leading dimensions (...)
        sum_of_a_i_squared = reduce(x ** 2, '... d -> ...', 'sum')
        # Calculate the normalization factor
        norm_factor = torch.rsqrt(sum_of_a_i_squared / self.d_model + self.eps)

        # Reshape from (...) to (..., 1) for broadcasting
        norm_factor = rearrange(norm_factor, '... -> ... 1')

        # Apply normalization and gain
        # Broadcasting now works:
        # (batch, seq_len, d_model) * (batch, seq_len, 1) * (d_model,)
        rms_norm = x * norm_factor * self.g

        return rms_norm.to(in_dtype)