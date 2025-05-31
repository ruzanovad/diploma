"""
Encoder modules for the im2latex project.

This module provides several encoder architectures for extracting visual features from images:
    - ResNetEncoder: Uses a truncated ResNet-18 followed by a linear projection
    to obtain visual embeddings.
    - ResNetWithRowEncoder: Extends ResNet with a row-wise bidirectional LSTM
    to capture sequential patterns.
    - PositionalEncoding: Implements sinusoidal positional encoding
    for sequence models.
    - ConvTransformerEncoder: Combines convolutional feature extraction
    with a Transformer encoder backend.

These encoders are designed for use in image-to-sequence models,
supporting flexible feature extraction
for downstream sequence decoders.
"""

import torch
from torch import Tensor, nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torchvision


class ResNetEncoder(nn.Module):
    """
    A ResNet-based encoder that extracts visual features from an input image.

    This encoder removes the final average pooling and fully connected layers
    from a standard ResNet18 and adds a linear projection to a custom embedding
    dimension (`enc_dim`).

    Args:
        enc_dim (int): The dimensionality of the output visual embedding vectors.
    """

    def __init__(self, enc_dim: int):

        super().__init__()
        self.resnet = torchvision.models.resnet18()  # smallest model
        # Убираем последние слои: avgpool и fc
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        self.fc = nn.Linear(512, enc_dim)
        self.enc_dim = enc_dim

    def forward(self, x: Tensor):
        """
        Forward pass for image feature extraction.

        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, enc_dim),
                    where `seq_len = feature_map_height * feature_map_width`.
                    Each position is a visual embedding vector.
        """
        # Feature map from ResNet: (B, 512, H', W')
        out = self.resnet(x)

        # Change to shape: (B, H', W', 512)
        out = out.permute(0, 2, 3, 1)

        # Linear projection to enc_dim
        out = self.fc(out)

        # Flatten spatial dimensions: (B, H'*W', enc_dim)
        bs, _, _, d = out.size()
        out = out.view(bs, -1, d)

        return out


class ResNetWithRowEncoder(nn.Module):
    """
    ResNet-based visual encoder with a row-wise BiLSTM.

    This encoder extracts a spatial feature map using a truncated ResNet-18
    and applies a bidirectional LSTM along each row (horizontal axis)
    to capture sequential patterns across the image width.

    Args:
        enc_dim (int): Dimensionality of the output feature embeddings.
    """

    def __init__(self, enc_dim: int):
        super().__init__()
        # Базовый ResNet без avgpool и fc
        resnet = torchvision.models.resnet18()
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])

        # Преобразуем каналы из 512 → enc_dim
        self.fc = nn.Conv2d(512, enc_dim, kernel_size=1)

        self.enc_dim = enc_dim
        # BiLSTM по строкам (ширина — как временная ось)
        self.row_encoder = nn.LSTM(
            input_size=enc_dim,
            hidden_size=enc_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): Input image tensor of shape (B, C=1, H, W)

        Returns:
            Tensor: Encoded sequence of shape (B, H'*W', enc_dim),
                    where H' and W' are downsampled height and width
                    from ResNet's final feature map.
        """
        # Сделать 3 канала, как требует ResNet
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        # Feature map: (B, 512, H′, W′)
        features = self.resnet(x)

        # Conv 1x1: (B, enc_dim, H′, W′)
        features = self.fc(features)  # (B, enc_dim, H', W')

        # Prepare for row-wise LSTM
        # Permute: (B, enc_dim, H′, W′) → (B, H′, W′, enc_dim)
        features = features.permute(0, 2, 3, 1)

        B, H, W, C = features.shape

        # Flatten batch and rows into sequence batch
        features = features.reshape(B * H, W, C)  # (B * H', W', enc_dim)

        # Apply BiLSTM across each row (horizontally)
        features, _ = self.row_encoder(features)  # (B * H', W', enc_dim)

        # Reshape back to (B, H', W', enc_dim)
        features = features.view(B, H, W, C)

        # Flatten spatial dimensions for output
        features = features.reshape(B, -1, C)  # (B, H'*W', enc_dim)

        return features


class PositionalEncoding(nn.Module):
    """Implements the sinusoidal positional encoding for Transformer models.

    This encoding adds information about the relative or absolute position of
    tokens in a sequence to help the model understand order since Transformers
    have no inherent notion of sequence order.
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        # Register as buffer so it's saved with model but not trained
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Add positional encoding to input tensor.

        Args:
            x (Tensor): Input sequence of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor: Sequence with added positional information
        """
        x = x + self.pe[:, : x.size(1)]
        return x


class ConvTransformerEncoder(nn.Module):
    """Convolutional encoder with Transformer backend for processing spectrograms.

    Combines CNN feature extraction with Transformer contextual processing.
    Typically used as the encoder in audio-to-sequence models.
    """

    def __init__(self, enc_dim=128, cnn_channels=64, num_layers=2, nhead=4):
        """
        Initialize the ConvTransformerEncoder module.

        Args:
            enc_dim (int): Dimension of encoder output features
            cnn_channels (int): Base number of channels for CNN layers
            num_layers (int): Number of Transformer encoder layers
            nhead (int): Number of attention heads in Transformer
        """
        super().__init__()
        self.enc_dim = enc_dim
        self.cnn_channels = cnn_channels
        self.num_layers = num_layers
        self.nhead = nhead

        # CNN frontend to reduce size and extract features
        self.cnn = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, cnn_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.2),
            # Layer 2
            nn.Conv2d(
                cnn_channels, cnn_channels * 2, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(cnn_channels * 2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.2),
            # Layer 3
            nn.Conv2d(
                cnn_channels * 2, cnn_channels * 4, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(cnn_channels * 4),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.2),
            # Final projection to encoder dimension
            nn.Conv2d(cnn_channels * 4, enc_dim, kernel_size=1),
        )

        self.positional_encoding = PositionalEncoding(enc_dim)

        encoder_layer = TransformerEncoderLayer(
            d_model=enc_dim,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(self, x):
        """
        Processes an input tensor through a CNN, applies positional encoding, and passes the result through a transformer encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 1, Freq, Time), typically a mel-spectrogram.

        Returns:
            torch.Tensor: Output tensor of shape (B, Seq_len, enc_dim), where Seq_len is the sequence length after CNN and pooling, and enc_dim is the encoder dimension.
        """
        x = self.cnn(x)  # (B, enc_dim, F', T')
        x = x.mean(dim=2)  # Average over frequency → (B, enc_dim, T')
        x = x.permute(0, 2, 1)  # → (B, T', enc_dim)

        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)  # (B, T', enc_dim)
        return x
