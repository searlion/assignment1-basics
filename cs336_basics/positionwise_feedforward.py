# In positionwise_feedforward.py

import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """
    Implements the SwiGLU (Swish-Gated Linear Unit) position-wise feed-forward network.

    This version is adapted to accept the hidden dimension `d_ff` directly,
    making it compatible with test harnesses that specify this dimension externally.

    The formula is: FFN_SwiGLU(x) = W_down(SiLU(x @ W_gate) * (x @ W_up))

    Args:
        d_model (int): The dimension of the input and output vectors.
        d_ff (int): The dimension of the internal feed-forward layer.
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()

        # The test adapter provides d_ff, so we use it directly instead of calculating it.

        # In LLaMA, these are called W1, W3, and W2 respectively.
        # W_gate (gate_proj) corresponds to w1_weight
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        # W_up (up_proj) corresponds to w3_weight
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        # W_down (down_proj) corresponds to w2_weight
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

        # SiLU (Sigmoid-weighted Linear Unit), also known as Swish
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SwiGLU network.

        Args:
            x (torch.Tensor): Input tensor of shape (..., d_model).

        Returns:
            torch.Tensor: Output tensor of shape (..., d_model).
        """
        # Apply the gate and up projections
        gate_output = self.gate_proj(x)
        up_output = self.up_proj(x)

        # Apply the SiLU activation and element-wise multiplication
        fused_output = self.activation(gate_output) * up_output

        # Apply the final down projection
        output = self.down_proj(fused_output)

        return output