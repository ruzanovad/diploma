import torch
from torch import nn, Tensor
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
        features = self.fc(features) # (B, enc_dim, H', W')

        # Prepare for row-wise LSTM
        # Permute: (B, enc_dim, H′, W′) → (B, H′, W′, enc_dim)
        features = features.permute(0, 2, 3, 1)

        B, H, W, C = features.shape

        # Flatten batch and rows into sequence batch
        features = features.reshape(B * H, W, C) # (B * H', W', enc_dim)

        # Apply BiLSTM across each row (horizontally)
        features, _ = self.row_encoder(features) # (B * H', W', enc_dim)

        # Reshape back to (B, H', W', enc_dim)
        features = features.view(B, H, W, C)

        # Flatten spatial dimensions for output
        features = features.reshape(B, -1, C) # (B, H'*W', enc_dim)

        return features
