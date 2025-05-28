"""
Defines the Decoder module for the im2latex model.

This module implements the Decoder class, which generates LaTeX token sequences from
encoded image features.
It uses an attention mechanism to focus on relevant parts
of the encoder output at each decoding step.
The Decoder supports configurable embedding, attention, and recurrent layers,
and is designed for use in sequence-to-sequence models for image-to-text tasks.

Classes:
    Decoder: Neural network module for autoregressive sequence decoding with attention.
"""

import torch
from torch import nn

from .attention import Attention


class Decoder(nn.Module):
    """
    Decoder module for sequence-to-sequence models with attention.

    This class implements a recurrent neural network (RNN) decoder with attention mechanism,
    suitable for tasks such as speech recognition or machine translation. The decoder takes
    the previous output tokens, attends to the encoder outputs, and generates the next token
    in the sequence.

        n_class (int): Number of output classes (vocabulary size).
        emb_dim (int, optional): Dimension of the embedding vectors. Default is 80.
        enc_dim (int, optional): Dimension of the encoder output features. Default is 512.
        dec_dim (int, optional): Dimension of the decoder hidden state. Default is 512.
        attn_dim (int, optional): Dimension of the attention mechanism. Default is 512.
        num_layers (int, optional): Number of layers in the LSTM. Default is 1.
        dropout (float, optional): Dropout probability for the LSTM. Default is 0.1.
        bidirectional (bool, optional): If True, use a bidirectional LSTM. Default is False.
        sos_id (int, optional): Start-of-sequence token ID. Default is 1.
        eos_id (int, optional): End-of-sequence token ID. Default is 2.

    Attributes:
        sos_id (int): Start-of-sequence token ID.
        eos_id (int): End-of-sequence token ID.
        embedding (nn.Embedding): Embedding layer for input tokens.
        attention (Attention): Attention mechanism module.
        concat (nn.Linear): Linear layer to combine embedding and attention context.
        rnn (nn.LSTM): LSTM layer(s) for decoding.
        out (nn.Linear): Output linear layer mapping to vocabulary size.
        logsoftmax (nn.LogSoftmax): LogSoftmax activation for output probabilities.

    Methods:
        init_weights(layer):
            Initializes the weights of the given layer. Uses orthogonal initialization
            for embeddings and LSTM weights.

        forward(y, encoder_out=None, hidden_state=None):
            Performs a forward pass of the decoder.

                y (Tensor): Input token indices of shape (batch_size, target_len).
                encoder_out (Tensor, optional): Encoder outputs for attention.
                hidden_state (tuple): Tuple of (h, c) hidden and cell states.

            Returns:
                out (Tensor): Log-probabilities over the vocabulary for the next token.
                hidden_state (tuple): Updated (h, c) hidden and cell states.
    """

    def __init__(
        self,
        n_class: int,
        emb_dim: int = 80,
        enc_dim: int = 512,
        dec_dim: int = 512,
        attn_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
        sos_id: int = 1,
        eos_id: int = 2,
    ):
        super().__init__()
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.embedding = nn.Embedding(n_class, emb_dim)
        self.attention = Attention(enc_dim, dec_dim, attn_dim)
        self.concat = nn.Linear(emb_dim + enc_dim, dec_dim)
        self.rnn = nn.LSTM(
            dec_dim,
            dec_dim,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        # self.layernorm = nn.LayerNorm((dec_dim))
        self.out = nn.Linear(dec_dim, n_class)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        self.apply(self.init_weights)

    def init_weights(self, layer):
        """
        Initializes the weights of the given layer.

        For nn.Embedding layers, applies orthogonal initialization to the embedding weights.
        For nn.LSTM layers, applies orthogonal initialization to all parameters
        whose names start with "weight".

        Args:
            layer (nn.Module): The layer whose weights are to be initialized.
        """
        if isinstance(layer, nn.Embedding):
            nn.init.orthogonal_(layer.weight)
        elif isinstance(layer, nn.LSTM):
            for name, param in self.rnn.named_parameters():
                if name.startswith("weight"):
                    nn.init.orthogonal_(param)

    def forward(self, y, encoder_out=None, hidden_state=None):
        """
        input:
            y: (bs, target_len)
            h: (bs, dec_dim)
            V: (bs, enc_dim, w, h)
        """
        # unpack the multi-layer state
        h_all, c_all = hidden_state  # (L, B, D)

        h_all = h_all.contiguous()
        c_all = c_all.contiguous()

        # pick only the top layer for attention
        h_top = h_all[-1]  # (B, D)

        # embed only once per time-step, then pick the last token
        emb = self.embedding(y)  # (B, T, E)
        last_emb = emb[:, -1]  # (B, E)

        # compute attention using h_top
        ctx = self.attention(h_top, encoder_out)  # (B, enc_dim)

        # build the single-step RNN input
        rnn_in = torch.cat([last_emb, ctx], dim=1)  # (B, E+enc_dim)
        rnn_in = self.concat(rnn_in)  # (B, dec_dim)
        rnn_in = rnn_in.unsqueeze(1)  # (B, 1, dec_dim)

        # run one step of the decoder LSTM
        out_seq, (h_n_all, c_n_all) = self.rnn(rnn_in, (h_all, c_all))
        # out_seq: (B, 1, dec_dim)

        # project to vocab and normalize
        logits = self.out(out_seq)  # (B, 1, vocab_size)
        logp = self.logsoftmax(logits)  # (B, 1, vocab_size)

        return logp, (h_n_all, c_n_all)
