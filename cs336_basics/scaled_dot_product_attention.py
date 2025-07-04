import torch
import torch.nn.functional as F
import math
from typing import Optional
from einops import einsum


def scaled_dot_product_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Implements the scaled dot-product attention mechanism using einops.

    This function calculates the attention scores by taking the dot product of the query
    with the key, scaling it, applying an optional mask, applying softmax to get
    attention weights, and finally multiplying by the value. The matrix multiplications
    are performed using `einops.einsum` for clarity and robustness.

    Args:
        query (torch.Tensor): The query tensor of shape (batch_size, ..., seq_len, d_k).
        key (torch.Tensor): The key tensor of shape (batch_size, ..., seq_len, d_k).
        value (torch.Tensor): The value tensor of shape (batch_size, ..., seq_len, d_v).
        mask (Optional[torch.Tensor]): An optional boolean mask of shape (seq_len, seq_len).
            Positions with `False` will be masked out (i.e., have their attention
            probability set to 0). Defaults to None.

    Returns:
        torch.Tensor: The output of the attention mechanism, a tensor of shape
                      (batch_size, ..., seq_len, d_v).
    """
    # The dimension of the keys and queries for scaling.
    d_k = query.shape[-1]

    # 1. Compute the dot product of the query and key using einops.
    # The pattern specifies that we multiply query and key, contracting the d_k dimension.
    # '... s_q d_k' -> Query with sequence length s_q and feature dim d_k
    # '... s_k d_k' -> Key with sequence length s_k and feature dim d_k
    # '... s_q s_k' -> Output scores with dims for each query and key
    scores = einsum(query, key, "... s_q d_k, ... s_k d_k -> ... s_q s_k")

    # 2. Scale the scores by the square root of d_k.
    scaled_scores = scores / math.sqrt(d_k)

    # 3. Apply the optional mask.
    # The mask has shape (s_q, s_k) and will be broadcasted to match
    # the shape of scaled_scores (..., s_q, s_k).
    if mask is not None:
        # Where mask is False, fill with a very small number before softmax.
        scaled_scores = scaled_scores.masked_fill(mask == False, -torch.inf)

    # 4. Apply softmax to get attention weights.
    # Softmax is applied to the last dimension (s_k) to get a probability
    # distribution over the key sequence for each query.
    attention_weights = F.softmax(scaled_scores, dim=-1)

    # 5. Multiply the attention weights by the value tensor using einops.
    # The pattern specifies multiplying weights and values, contracting the s_k dimension.
    # '... s_q s_k' -> Attention weights
    # '... s_k d_v' -> Value with feature dim d_v
    # '... s_q d_v' -> Final output with feature dim d_v for each query
    output = einsum(attention_weights, value, "... s_q s_k, ... s_k d_v -> ... s_q d_v")

    return output