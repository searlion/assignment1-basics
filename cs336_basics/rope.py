import torch
import torch.nn as nn
from typing import Optional
from einops import rearrange, repeat


class RotaryPositionalEmbedding(nn.Module):
    """
    Implements Rotary Positional Embedding (RoPE) as described in https://arxiv.org/abs/2104.09864.

    This module precomputes the cosine and sine frequency waves at different
    positions and applies them to the input tensor during the forward pass.
    This version uses the 'einops' library for clearer tensor manipulation.
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: Optional[torch.device] = None):
        """
        Constructs the RoPE module and precomputes required buffers.

        Args:
            theta (float): The base for the geometric progression of frequencies.
                           A common value is 10000.0.
            d_k (int): The dimension of the query and key vectors. Must be even.
            max_seq_len (int): The maximum sequence length that the model will handle.
            device (torch.device | None, optional): The device to store buffers on.
                                                     Defaults to None.
        """
        super().__init__()

        if d_k % 2 != 0:
            raise ValueError(f"d_k must be even, but got {d_k}")

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # Precompute the frequencies (θ_i in the paper)
        # Shape: (d_k / 2,)
        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))

        # Precompute the positions (m in the paper)
        # Shape: (max_seq_len,)
        t = torch.arange(max_seq_len, device=device)

        # Create the full angle matrix m * θ_i using an outer product
        # Shape: (max_seq_len, d_k / 2)
        freqs_outer = torch.outer(t, freqs).float()

        # Use einops.repeat to duplicate each frequency for the pair of dimensions.
        # 's d -> s (d 2)' means for each sequence position 's', repeat each
        # frequency 'd' twice, creating a new dimension of size (d * 2).
        # Shape: (max_seq_len, d_k)
        cos_cache = repeat(torch.cos(freqs_outer), 's d -> s (d 2)')
        sin_cache = repeat(torch.sin(freqs_outer), 's d -> s (d 2)')

        self.register_buffer('cos_cache', cos_cache, persistent=False)
        self.register_buffer('sin_cache', sin_cache, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply Rotary Positional Embedding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (..., seq_len, d_k).
                              Can have arbitrary batch dimensions.
            token_positions (torch.Tensor): Tensor of shape (..., seq_len) specifying
                                            the absolute position of each token in x.

        Returns:
            torch.Tensor: Tensor of the same shape as x with RoPE applied.
        """
        if self.cos_cache.device != x.device:
            self.cos_cache = self.cos_cache.to(x.device)
            self.sin_cache = self.sin_cache.to(x.device)

        # Retrieve the precomputed cos and sin values for the given token positions.
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]

        # The rotation trick:
        # We want to transform a vector [x_0, x_1, x_2, x_3, ...]
        # into its rotated counterpart [-x_1, x_0, -x_3, x_2, ...].

        # 1. Split x into its even and odd components.
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        # 2. Negate the odd components.
        neg_x_odd = -x_odd

        # 3. Use einops.rearrange to interleave the pairs.
        # This takes the two tensors (neg_x_odd and x_even), stacks them on a new
        # final dimension, and then rearranges that into the final dimension.
        # This is equivalent to `torch.flatten(torch.stack([-x_odd, x_even], dim=-1), start_dim=-2)`
        stacked_halves = torch.stack([-x_odd, x_even], dim=-1)
        x_rotated = rearrange(stacked_halves, '... d two -> ... (d two)')

        # Apply the final rotation formula using broadcasting
        return x * cos + x_rotated * sin