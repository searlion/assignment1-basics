import torch
from torch import nn
from einops import rearrange
from typing import Dict

from cs336_basics.positionwise_feedforward import PositionwiseFeedForward
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.rope import RotaryPositionalEmbedding

# These type hints are used in the function signature as per the prompt.
# A try-except block is used for robustness in case torchtyping is not installed.
try:
    from torchtyping import Tensor, Float
except ImportError:
    # Fallback to standard PyTorch Tensor if torchtyping is unavailable.
    Tensor = torch.Tensor
    Float = torch.Tensor

# The following components are assumed to be pre-defined in the environment,
# as specified in the problem description.
# from prebuilt_components import RMSNorm, RotaryPositionalEmbedding, PositionwiseFeedForward


class TransformerBlock(nn.Module):
    """
    Implements a pre-norm Transformer block as depicted in the provided figure.

    This block consists of two main sub-layers:
    1. Causal Multi-Head Self-Attention with Rotary Positional Embeddings (RoPE).
    2. A Position-wise Feed-Forward network (using SwiGLU).

    Each sub-layer is surrounded by a residual connection, and normalization (RMSNorm)
    is applied to the input of each sub-layer (pre-norm).
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.d_head = d_model // num_heads

        # First sub-layer: Causal Multi-Head Self-Attention with RoPE
        self.ln1 = RMSNorm(d_model)
        # We use an nn.Module to group attention layers, matching the weight keys.
        self.attn = nn.Module()
        self.attn.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn.output_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn.rope = RotaryPositionalEmbedding(theta, self.d_head, max_seq_len)

        # Second sub-layer: Position-wise Feed-Forward Network
        self.ln2 = RMSNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff)

    def _attention_forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """Helper function to compute the attention output."""
        batch_size, seq_len, _ = x.shape

        # Project input into query, key, and value tensors
        q = self.attn.q_proj(x)
        k = self.attn.k_proj(x)
        v = self.attn.v_proj(x)

        # Reshape for multi-head attention
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        # Apply Rotary Positional Embeddings to queries and keys
        q = self.attn.rope(q, token_positions)
        k = self.attn.rope(k, token_positions)

        # Compute scaled dot-product attention scores
        scores = torch.einsum("b h i d, b h j d -> b h i j", q, k) / (self.d_head**0.5)

        # Apply causal mask to prevent attention to future tokens
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        scores.masked_fill_(causal_mask, -torch.inf)

        # Compute attention weights and apply to values
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.einsum("b h i j, b h j d -> b h i d", attention_weights, v)

        # Combine heads and apply final output projection
        output = rearrange(output, "b h s d -> b s (h d)")
        return self.attn.output_proj(output)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, d_model).
            token_positions (torch.Tensor): Tensor of shape (batch, seq_len) or (seq_len,)
                                            with absolute positions for RoPE.

        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, d_model).
        """
        # First residual connection with pre-normalization
        x = x + self._attention_forward(self.ln1(x), token_positions)
        # Second residual connection with pre-normalization
        x = x + self.ffn(self.ln2(x))
        return x