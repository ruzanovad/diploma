"""
Attention mechanism module for the im2latex project.

This module implements the additive (Bahdanau-style) attention mechanism used in 
sequence-to-sequence models.
The Attention class computes a context vector as a weighted sum of encoder outputs, 
where the weights are
determined by the similarity between the decoder hidden state and each encoder output.

Classes:
    Attention: Neural network module for additive attention, 
    supporting context computation for decoder steps.
"""

import torch
from torch import nn, Tensor


class Attention(nn.Module):
    """
    Additive (Bahdanau-style) attention mechanism.

    Args:
        enc_dim (int): Dimensionality of encoder outputs (default: 512)
        dec_dim (int): Dimensionality of decoder hidden state (default: 512)
        attn_dim (int): Dimensionality of intermediate attention space (default: 512)

    Forward Inputs:
        h (Tensor): Decoder hidden state at current time step, shape (batch_size, dec_dim)
        V (Tensor): Encoder output features, shape (batch_size, seq_len, enc_dim)

    Returns:
        context (Tensor): Context vector computed as the weighted sum of encoder features,
                          shape (batch_size, enc_dim)
    """

    def __init__(self, enc_dim: int = 512, dec_dim: int = 512, attn_dim: int = 512):
        super().__init__()

        # Linear layer to project decoder hidden state to attention space
        self.dec_attn = nn.Linear(dec_dim, attn_dim, bias=False)

        # Linear layer to project encoder outputs to attention space
        self.enc_attn = nn.Linear(enc_dim, attn_dim, bias=False)

        # Final layer to compute unnormalized attention scores
        self.full_attn = nn.Linear(attn_dim, 1, bias=False)

        # Softmax to normalize attention weights
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, h: Tensor, V: Tensor):
        """
        Compute attention context vector.

        Args:
            h (Tensor): Decoder hidden state at current time step, shape (batch_size, dec_dim)
            V (Tensor): Encoder output features, shape (batch_size, seq_len, enc_dim)

        Returns:
            context (Tensor): Weighted sum of encoder features, shape (batch_size, enc_dim)
        """
        # Project decoder hidden state: (b, attn_dim)
        attn_1 = self.dec_attn(h)

        # Project encoder outputs: (b, seq_len, attn_dim)
        attn_2 = self.enc_attn(V)

        # Combine decoder and encoder projections, then apply tanh and final linear layer
        # attn shape: (b, seq_len)
        attn = self.full_attn(torch.tanh(attn_1.unsqueeze(1) + attn_2)).squeeze(
            2
        )  # shape: (b, seq_len)

        # Compute attention weights
        alpha = self.softmax(attn)

        # Weighted sum of encoder features
        context = (alpha.unsqueeze(2) * V).sum(dim=1)  # shape: (b, enc_dim)
        return context
