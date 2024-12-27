import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, input_size=(128, 128)):
        super(Net, self).__init__()
        self.conv_layers = nn.Sequential(
            # Layer 1: Conv1
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            # Layer 2: MaxPool1
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Layer 3: Conv2
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Layer 4: MaxPool2
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Layer 5: Conv3
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Layer 6: MaxPool3
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        # Fully Connected Layers
        self.fc_layers = nn.Sequential(
            nn.Linear(1024, 4096),  # Flattened size from the final MaxPool
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
