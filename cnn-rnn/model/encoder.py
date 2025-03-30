import torch
from torch import nn, Tensor
import torchvision


class ResNetEncoder(nn.Module):
    def __init__(self, enc_dim: int):
        super().__init__()
        self.resnet = torchvision.models.resnet18()  # smallest model
        # Убираем последние слои: avgpool и fc
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        self.fc = nn.Linear(512, enc_dim)
        self.enc_dim = enc_dim

    def forward(self, x: Tensor):
        """
        x: (bs, c, w, h)
        """
        out = self.resnet(x)
        out = out.permute(0, 2, 3, 1)
        out = self.fc(out)
        bs, _, _, d = out.size()
        out = out.view(bs, -1, d)
        return out


class ResNetWithRowEncoder(nn.Module):
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

    def forward(self, x):
        """
        x: (B, C=1, H, W)
        output: (B, H'*W', enc_dim)
        """
        # Сделать 3 канала, как требует ResNet
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        # Feature map: (B, 512, H′, W′)
        features = self.resnet(x)

        # Conv 1x1: (B, enc_dim, H′, W′)
        features = self.fc(features)
        # Prepare for row-wise LSTM
        # Переставляем: (B, enc_dim, H′, W′) → (B, H′, W′, enc_dim)
        features = features.permute(0, 2, 3, 1)


        B, H, W, C = features.shape
        # Объединяем batch и строки: (B * H′, W′, enc_dim)
        features = features.reshape(B * H, W, C)

        # BiLSTM по строкам
        features, _ = self.row_encoder(features)

        # Вернуть назад: (B, H′, W′, enc_dim)
        features = features.view(B, H, W, C)

        # Переставить в нужный формат: (B, H′*W′, enc_dim)
        features = features.reshape(B, -1, C)

        return features
