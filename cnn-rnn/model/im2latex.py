"""
Image2Latex model definition for the im2latex project.

This module implements the main Image2Latex neural network, which combines an image encoder
and a sequence decoder to convert images of mathematical formulas into LaTeX markup.
The model supports multiple encoder architectures,
greedy and beam search decoding, and is designed for use in end-to-end image-to-sequence tasks.

Classes:
    Image2Latex: Neural network module for image-to-LaTeX sequence modeling, supporting training
    and inference.
"""

import torch
from torch import Tensor, nn

from .decoder import Decoder
from .encoder import *
from .text import Text


class Image2Latex(nn.Module):
    """
    Image2Latex model for converting images to LaTeX sequences.

    This class implements an encoder-decoder architecture for the image-to-LaTeX task,
    supporting multiple encoder types
    (e.g., ResNet, Transformer) and decoding strategies (greedy and beam search).
    The model takes an input image, encodes it
    into a feature representation, and decodes it into a sequence of LaTeX tokens.

        enc_layers (int, optional): Number of layers in the encoder (for transformer). Default is 2.
        nhead (int, optional): Number of attention heads (for transformer encoder). Default is 16.
        decode_type (str, optional): Decoding strategy, either "greedy" or "beamsearch".
        Default is "greedy".


    Methods:
        init_decoder_hidden_state(V):
            Initializes the decoder's hidden state (h, c) from the encoder output.

        forward(x, y, y_len):
            Performs a forward pass through the model for training, returning predictions
            for each timestep.

        decode(x, max_length=150):
            Decodes an input image into a LaTeX sequence using the specified decoding strategy.

        decode_greedy(x, max_length=150):
            Performs greedy decoding for sequence generation.

        decode_beam_search(x, max_length=150):
            Performs beam search decoding for sequence generation.
    """

    def __init__(
        self,
        n_class: int,
        enc_dim: int = 512,
        enc_type: str = "resnet_encoder",
        emb_dim: int = 80,
        dec_dim: int = 512,
        attn_dim: int = 512,
        cnn_channels: int = 32,
        enc_layers: int = 2,
        nhead: int = 16,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
        decode_type: str = "greedy",
        text: Text = None,
        beam_width: int = 5,
        sos_id: int = 1,
        eos_id: int = 2,
    ):
        """
        Initializes the Im2Latex model with configurable encoder and decoder architectures.

        Args:
            n_class (int): Number of output classes (vocabulary size).
            enc_dim (int, optional): Dimension of encoder output features. Default is 512.
            enc_type (str, optional): Type of encoder to use. Must be one of
                ["conv_row_encoder", "conv_encoder", "conv_bn_encoder", "resnet_encoder",
                "resnet_row_encoder", "transformer_encoder"]. Default is "resnet_encoder".
            emb_dim (int, optional): Dimension of token embeddings in the decoder. Default is 80.
            dec_dim (int, optional): Dimension of decoder hidden state. Default is 512.
            attn_dim (int, optional): Dimension of attention mechanism. Default is 512.
            cnn_channels (int, optional): Number of channels in CNN-based encoders. Default is 32.
            enc_layers (int, optional): Number of layers in the encoder (for transformer).
            Default is 2.
            nhead (int, optional): Number of attention heads (for transformer encoder).
            Default is 16.
            num_layers (int, optional): Number of layers in the decoder. Default is 1.
            dropout (float, optional): Dropout probability in the decoder. Default is 0.1.
            bidirectional (bool, optional): If True, use bidirectional decoder. Default is False.
            decode_type (str, optional): Decoding strategy, either "greedy" or "beamsearch".
            Default is "greedy".
            text (Text, optional): Optional text processor or vocabulary object. Default is None.
            beam_width (int, optional): Beam width for beam search decoding. Default is 5.
            sos_id (int, optional): Start-of-sequence token ID. Default is 1.
            eos_id (int, optional): End-of-sequence token ID. Default is 2.

        Raises:
            AssertionError: If `enc_type` is not a supported encoder type.
            AssertionError: If `decode_type` is not "greedy" or "beamsearch".
        """

        assert enc_type in [
            "conv_row_encoder",
            "conv_encoder",
            "conv_bn_encoder",
            "resnet_encoder",
            "resnet_row_encoder",
            "transformer_encoder",
        ], "Not found encoder"
        super().__init__()
        self.n_class = n_class
        # if enc_type == "conv_row_encoder":
        #     self.encoder = ConvWithRowEncoder(enc_dim=enc_dim)
        # elif enc_type == "conv_encoder":
        #     self.encoder = ConvEncoder(enc_dim=enc_dim)
        # elif enc_type == "conv_bn_encoder":
        #     self.encoder = ConvBNEncoder(enc_dim=enc_dim)
        if enc_type == "resnet_encoder":
            self.encoder = ResNetEncoder(enc_dim=enc_dim)
        elif enc_type == "resnet_row_encoder":
            self.encoder = ResNetWithRowEncoder(enc_dim=enc_dim)
        elif enc_type == "transformer_encoder":
            self.encoder = ConvTransformerEncoder(
                enc_dim=enc_dim,
                cnn_channels=cnn_channels,
                num_layers=enc_layers,
                nhead=nhead,
            )
        # enc_dim = self.encoder.enc_dim
        self.num_layers = num_layers
        self.decoder = Decoder(
            n_class=n_class,
            emb_dim=emb_dim,
            dec_dim=dec_dim,
            enc_dim=enc_dim,
            attn_dim=attn_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            sos_id=sos_id,
            eos_id=eos_id,
        )
        self.init_h = nn.Linear(enc_dim, dec_dim * num_layers)
        self.init_c = nn.Linear(enc_dim, dec_dim * num_layers)
        assert decode_type in ["greedy", "beamsearch"]
        self.decode_type = decode_type
        self.text = text
        self.beam_width = beam_width

    def init_decoder_hidden_state(self, V: Tensor):
        """
        Initializes the hidden state (h, c) for the decoder based on the encoder output.

        Args:
            V (Tensor): The output tensor from the encoder with shape
            (batch_size, seq_len, feature_dim).

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the initial hidden state (h) and
            cell state (c) for the decoder,
            both with shape (batch_size, hidden_dim).
        """
        bs = V.size(0)
        encoder_mean = V.mean(dim=1)  # (bs, enc_dim)

        # project to (bs, dec_dim * n_layers)
        h_flat = torch.tanh(self.init_h(encoder_mean))
        c_flat = torch.tanh(self.init_c(encoder_mean))

        # reshape into (n_layers, bs, dec_dim)
        # it is needed for decoder lstm with multiple layers (expects hidden state of shape
        # (num_layers, batch_size, dec_dim))
        h0 = h_flat.view(bs, self.num_layers, -1).transpose(0, 1)
        c0 = c_flat.view(bs, self.num_layers, -1).transpose(0, 1)

        return h0, c0

    def forward(self, x: Tensor, y: Tensor, y_len: Tensor):
        """
        Performs a forward pass through the model.

        Args:
            x (Tensor): Input tensor representing the source data (e.g., images).
            y (Tensor): Target sequence tensor (e.g., token indices for each timestep).
            y_len (Tensor): Tensor containing the lengths of each target sequence in the batch.

        Returns:
            Tensor: Predicted output sequences for each timestep, shape
            (batch_size, sequence_length, output_dim).
        """
        encoder_out = self.encoder(x)

        hidden_state = self.init_decoder_hidden_state(encoder_out)

        predictions = []
        # Original y_len is the number of real tokens; +1 to account for the SOS in y
        T = y.size(1)                       # exact number of time‐steps in y
        for t in range(T):                  # t = 0 .. T-1
            dec_input = y[:, t : t + 1]     # shape (B,1), safe slice
            out, hidden_state = self.decoder(dec_input, encoder_out, hidden_state)
            predictions.append(out.squeeze(1))

        predictions = torch.stack(predictions, dim=1)
        return predictions

    def decode(self, x: Tensor, max_length: int = 150):
        """
        Decodes the given input tensor into a sequence of tokens using
        the specified decoding strategy.

        Args:
            x (Tensor): The input tensor to decode, typically representing encoded features.
            max_length (int, optional): The maximum length of the decoded sequence. Defaults to 150.

        Returns:
            str: The decoded sequence as a string.

        Notes:
            The decoding strategy is determined by the `decode_type` attribute of the class.
            Supported strategies are "greedy" and "beamsearch".
        """
        predict = []
        if self.decode_type == "greedy":
            predict = self.decode_greedy(x, max_length)
        elif self.decode_type == "beamsearch":
            predict = self.decode_beam_search(x, max_length)
        return self.text.int2text(predict)

    def decode_greedy(self, x: Tensor, max_length: int = 150):
        """
        Performs greedy decoding for sequence generation using the encoder and decoder.

        Args:
            x (Tensor): Input tensor to be encoded, typically representing an image or sequence.
            max_length (int, optional): Maximum length of the output sequence. Defaults to 150.

        Returns:
            List[int]: List of predicted token indices representing the decoded sequence.
        """
        encoder_out = self.encoder(x)
        device = encoder_out.device  # <- get the correct device

        bs = encoder_out.size(0)
        hidden_state = self.init_decoder_hidden_state(encoder_out)

        y = (
            torch.LongTensor([self.decoder.sos_id]).view(bs, -1).to(device)
        )  # <- move to same device

        predictions = []
        for _ in range(max_length):
            out, hidden_state = self.decoder(y, encoder_out, hidden_state)

            k = int(out.argmax())

            predictions.append(k)

            y = (
                torch.LongTensor([k]).view(bs, -1).to(device)
            )  # <- also move to same device
        return predictions
    
    @torch.no_grad()
    def decode_beam_search(self, x: Tensor, max_length: int = 150):
        """
        Performs beam search decoding for sequence generation using the model's encoder and decoder.

        Args:
            x (Tensor): Input tensor to be encoded, typically representing an image or feature map.
            max_length (int, optional): Maximum length of the output sequence. Defaults to 150.

        Returns:
            List[int]: The most probable sequence of token indices decoded by beam search.

        Notes:
            - Assumes batch size of 1.
            - Uses the model's predefined beam width
            (`self.beam_width`).
            - The sequence starts with the decoder's start-of-sequence token
            (`self.decoder.sos_id`).
        """

        encoder_out = self.encoder(x)
        bs = encoder_out.size(0)  # 1

        hidden_state = self.init_decoder_hidden_state(encoder_out)

        list_candidate = [
            ([self.decoder.sos_id], hidden_state, 0)
        ]  # (input, hidden_state, log_prob)
        for _ in range(max_length):
            new_candidates = []
            for inp, state, log_prob in list_candidate:
                y = torch.LongTensor([inp[-1]]).view(bs, -1).to(device=x.device)
                out, hidden_state = self.decoder(y, encoder_out, state)

                topk = out.topk(self.beam_width)
                new_log_prob = topk.values.view(-1).tolist()
                new_idx = topk.indices.view(-1).tolist()
                for val, idx in zip(new_log_prob, new_idx):
                    new_inp = inp + [idx]
                    new_candidates.append((new_inp, hidden_state, log_prob + val))

            new_candidates = sorted(new_candidates, key=lambda x: x[2], reverse=True)
            list_candidate = new_candidates[: self.beam_width]

        return list_candidate[0][0]
