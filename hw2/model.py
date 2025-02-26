import torch
import torch.nn as nn
from typing import Optional

class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, downsample: Optional[nn.Sequential] = None, batch_norm: bool = True) -> None:
        super().__init__()
        self.conv1: nn.Conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.relu: nn.ReLU = nn.ReLU(inplace=True)
        self.conv2: nn.Conv2d = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2: nn.BatchNorm2d = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.downsample: nn.Module = downsample
        self.stride: int = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, batch_norm: bool = True) -> None:
        super().__init__()
        self.conv1: nn.Conv2d = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(64) if batch_norm else nn.Identity()
        self.relu: nn.ReLU = nn.ReLU(inplace=True)
        self.maxpool: nn.MaxPool2d = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1: nn.Sequential = self._make_layer(64, 64, 2, 1, batch_norm)
        self.layer2: nn.Sequential = self._make_layer(64, 128, 2, 2, batch_norm)
        self.layer3: nn.Sequential = self._make_layer(128, 256, 2, 2, batch_norm)
        self.layer4: nn.Sequential = self._make_layer(256, 512, 2, 2, batch_norm)
        self.avgpool: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.fc: nn.Linear = nn.Linear(512, 10)
    
    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int, batch_norm: bool) -> nn.Sequential:
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
            )
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample, batch_norm))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1, batch_norm=batch_norm))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x