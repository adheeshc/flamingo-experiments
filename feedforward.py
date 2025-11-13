"""Feed-Forward Network Layers"""

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """FeedForward Network

    Standard MLP block with LayerNorm, GELU activation, and dropout

    Args:
        dim: Input/output dimension
        mult: Multiplier for hidden dimension (default: 4)
        dropout: Dropout probability (default: 0.0)
    """

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            x: Input tensor [batch, seq_len, dim]

        Returns:
            Output tensor [batch, seq_len, dim]
        """
        return self.net(x)
