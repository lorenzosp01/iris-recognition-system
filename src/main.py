import cv2
import torch
from torch.utils.data import random_split, DataLoader
from torchvision import transforms as tt
from src.data.CasiaIrisDataset import CasiaIrisDataset
from src.lib.cnn import Net
from src.lib.cnn_utils import trainModel, to_device
from src.utils.utils import collate_fn

if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    datasetPath = "F:\\Dataset\\Casia"
    train_percentage = 0.8
    transform = tt.Compose([tt.Resize(128),
                            tt.ToTensor()])

    dataset = CasiaIrisDataset(datasetPath, transform=[transform])

    dataset.train()

    train_size = int(train_percentage * len(dataset))  # portion for training
    test_size = len(dataset) - train_size  # remaining portion for testing
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Assume your dataset is already loaded
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True, num_workers=0, collate_fn=collate_fn)

    net = to_device(Net(), device)
    trainModel(device, net, train_dataloader, num_epochs=10, learning_rate=1e-3)
