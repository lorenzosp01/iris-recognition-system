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

        # Calculate the flattened size of the final convolutional output
        test_input = torch.zeros(1, 1, *input_size)  # Batch size 1, input channels 1
        conv_output_size = self.conv_layers(test_input).view(1, -1).size(1)

        # Fully Connected Layers
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 1024),  # Flattened size from the final MaxPool
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)  # Output layer
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
