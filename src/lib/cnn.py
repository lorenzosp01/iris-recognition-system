import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Load ResNet50
        self.net = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        # Modify the first convolutional layer for single-channel input
        self.net.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False
        )

        # Modify the fully connected layer
        self.net.fc = nn.Linear(in_features=2048, out_features=2048, bias=True)

    def forward(self, x):
        # Forward pass through ResNet50
        x = self.net(x)

        # Apply L2 normalization
        x = F.normalize(x, p=2, dim=1)  # Normalize across the feature dimension

        return x