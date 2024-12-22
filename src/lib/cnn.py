import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 14),
            nn.Conv2d(96, 256, kernel_size=5, stride=30, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 6),
            nn.Conv2d(256, 256, kernel_size=3, stride=6, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 3)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 3 * 3, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
