import torch
import torch.nn as nn
from einops import rearrange, repeat


class CausalMultiHeadSelfAttention(nn.Module):
    """
    Implements causal multi-head self-attention as described in
    "Attention Is All You Need" by Vaswani et al. [2017].

    This module projects the input into Query, Key, and Value tensors,
    splits them into multiple heads, and then applies scaled dot-product
    attention with a causal mask. The outputs of the heads are then
    concatenated and projected back to the original dimension.

    Uses einops for clear and efficient tensor manipulation.
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        # This layer expects a weight of shape (3 * d_model, d_model)
        self.to_qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.to_out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *_, seq_len, _ = x.shape
        qkv = self.to_qkv(x)
        # qkv has shape (..., seq_len, 3 * d_model)
        # 3 * d_model = 3 * num_heads * d_head. This decomposition will now work correctly.
        q, k, v = rearrange(qkv, '... t (qkv h d) -> qkv ... h t d', qkv=3, h=self.num_heads)
        dots = torch.einsum('... h i d, ... h j d -> ... h i j', q, k) / (self.d_head ** 0.5)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        dots.masked_fill_(causal_mask, float('-inf'))
        attn_weights = dots.softmax(dim=-1)
        output = torch.einsum('... h i j, ... h j d -> ... h i d', attn_weights, v)
        output = rearrange(output, '... h t d -> ... t (h d)')
        return self.to_out(output)