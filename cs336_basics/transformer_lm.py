import torch
from torch import nn
from typing import Dict

from cs336_basics.linear import Linear
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.transformer_block import TransformerBlock

# These type hints are used in the function signature as per the prompt.
# A try-except block is used for robustness in case torchtyping is not installed.
try:
    from torchtyping import Tensor, Float, Int
except ImportError:
    # Fallback to standard PyTorch Tensor if torchtyping is unavailable.
    Tensor = torch.Tensor
    Float = torch.Tensor
    Int = torch.Tensor

# The following components are assumed to be pre-defined in the environment,
# as specified in the problem description:
# - TransformerBlock (from the previous exercise)
# - RMSNorm
# - Linear
# - RotaryPositionalEmbedding (used by TransformerBlock)
# - PositionwiseFeedForward (used by TransformerBlock)


class TransformerLM(nn.Module):
    """
    A Transformer-based language model.

    This model consists of an embedding layer, a series of Transformer blocks,
    a final normalization layer, and a linear head to produce vocabulary logits.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        """
        Initializes the Transformer Language Model.

        Args:
            vocab_size (int): The size of the vocabulary.
            context_length (int): The maximum sequence length for RoPE pre-computation.
            d_model (int): The dimensionality of the model's embeddings.
            num_layers (int): The number of stacked Transformer blocks.
            num_heads (int): The number of attention heads in each Transformer block.
            d_ff (int): The inner dimension of the feed-forward networks.
            rope_theta (float): The base period for Rotary Positional Embeddings.
        """
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

    def to_config(self) -> Dict:
        """
        Returns the configuration parameters of the model.

        Returns:
            Dict: A dictionary containing all configuration parameters.
        """
        return {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "rope_theta": self.rope_theta,
        }

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass of the language model.

        Args:
            in_indices (torch.Tensor): Input token indices of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, seq_len, vocab_size).
        """
        _, seq_len = in_indices.shape
        device = in_indices.device

        # Create positional information for RoPE, required by the Transformer blocks.
        token_positions = torch.arange(seq_len, device=device)

        # 1. Get token embeddings
        x = self.token_embeddings(in_indices)

        # 2. Pass through Transformer blocks
        for layer in self.layers:
            x = layer(x, token_positions)

        # 3. Apply final normalization
        x = self.ln_final(x)

        # 4. Apply the language model head to get logits
        logits = self.lm_head(x)

        return logits