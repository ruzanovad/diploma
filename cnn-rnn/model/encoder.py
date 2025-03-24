import torch
from torch import nn, Tensor
import torchvision


class ResNetEncoder(nn.Module):
    def __init__(self, enc_dim: int):
        super().__init__()
        self.resnet = torchvision.models.resnet18() # smallest model
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
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
