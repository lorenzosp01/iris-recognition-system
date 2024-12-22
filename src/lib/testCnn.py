import os.path

from mpl_toolkits.mplot3d.proj3d import transform
from torchvision import datasets, transforms as tt
import torch
from src.lib.cnn import Net
from src.lib.cnn_utils import save_model, load_model, to_device, trainModel, testModel

if __name__=="__main__":
    datasetPath = "F:\Dataset\Casia\casia-cnn-full-eye"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    loadExistingModel = False
    modelName = "./models/modelCNN.pth"

    if loadExistingModel:
        net = load_model(modelName)
    else:
        net = to_device(Net(), device)

    transform = tt.Compose([tt.ToTensor()])

    train = datasets.ImageFolder(datasetPath, transform=transform)
    test = datasets.ImageFolder(datasetPath, transform=transform)
    train_dl = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test, batch_size=32, shuffle=False)

    trainModel(device, net, train_dl, epochs=10, learning_rate=0.001)

    testModel(device, net, test_dl)







    if not loadExistingModel:
        save_model(modelName, net)