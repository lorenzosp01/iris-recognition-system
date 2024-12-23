import torch
from torch.utils.data import random_split, DataLoader
from torchvision import transforms as tt
from src.data.CasiaIrisDataset import CasiaIrisDataset
from src.lib.cnn import Net
from src.lib.cnn_utils import trainModel, to_device

if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    datasetPath = "F:\\Dataset\\Casia"
    train_percentage = 0.8
    transform = tt.Compose([tt.CenterCrop(128), tt.ToTensor()])

    dataset = CasiaIrisDataset(datasetPath, transform=[transform])

    dataset.train()

    train_size = int(train_percentage * len(dataset))  # portion for training
    test_size = len(dataset) - train_size  # remaining portion for testing
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Assume your dataset is already loaded
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True, num_workers=0)

    batch = next(iter(train_dataloader))
    anchor, positive, negative, anchor_label = batch
    print(type(anchor), anchor.shape)
    print(type(positive), positive.shape)
    print(type(negative), negative.shape)
    print(type(anchor_label), anchor_label.shape)

