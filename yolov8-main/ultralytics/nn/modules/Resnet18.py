import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad

class ResNet18Block(nn.Module):
    """ResNet18 and ResNet34 block with standard convolution layers."""

    def __init__(self, c1, c2, s=1):
        """Initialize convolution with given parameters."""
        super().__init__()
        self.cv1 = Conv(c1, c2, k=3, s=s, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c2, k=1, s=s, act=False)) if s != 1 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv2(self.cv1(x)) + self.shortcut(x))

class ResNet18Layer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True),
                # Conv(c1, c1, k=3, s=2, act='relu'),
                # Conv(c1, c1, k=3, s=1, act='relu'),
                # Conv(c1, c2, k=3, s=1, act='relu'),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNet18Block(c1, c2, s)]
            blocks.extend([ResNet18Block(c2, c2, 1) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)
