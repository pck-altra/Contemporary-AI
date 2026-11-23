"""Compact CIFAR CNNs. Same backbone; residual/SE are controlled switches."""

import torch
from torch import nn


class ConvBN(nn.Sequential):
    def __init__(self, cin, cout, stride=1, relu=True):
        super().__init__(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU() if relu else nn.Identity(),
        )


class SE(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(4, channels // reduction), 1),
            nn.ReLU(),
            nn.Conv2d(max(4, channels // reduction), channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


class Stage(nn.Module):
    def __init__(self, cin, cout, residual, se):
        super().__init__()
        self.down = ConvBN(cin, cout, stride=2)
        self.body = nn.Sequential(ConvBN(cout, cout), ConvBN(cout, cout, relu=False))
        self.attention = SE(cout) if se else nn.Identity()
        self.residual = residual

    def forward(self, x):
        x = self.down(x)
        y = self.attention(self.body(x))
        return torch.relu(x + y if self.residual else y)


class Classifier(nn.Module):
    def __init__(self, residual=False, se=False):
        super().__init__()
        self.stem = ConvBN(3, 32)
        self.stages = nn.Sequential(
            Stage(32, 32, residual, se), Stage(32, 64, residual, se), Stage(64, 128, residual, se)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(128, 10)

    def features(self, x):
        return self.stages(self.stem(x))

    def forward(self, x):
        x = self.features(x)
        return self.head(self.drop(self.pool(x).flatten(1)))
