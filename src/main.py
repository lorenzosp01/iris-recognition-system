import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Subset, DataLoader
from torchvision import transforms as tt, models
from torchvision.models import ResNet50_Weights

from data.CasiaIrisDataset import CasiaIrisDataset, split_dataset_gallery_test
from lib.cnn import Net
from lib.cnn_utils import trainModel, save_model, testIdentificationSystem, load_model

if __name__=="__main__":
    datasetPath = "../../../bigdata/Casia"

    saveModel = True
    modelPath = "./models/model.pth"

    transform = tt.Compose([tt.ToTensor()])

    dataset = CasiaIrisDataset(datasetPath, transform=[transform], centered=True)

    seed = 20
    torch.manual_seed(seed)

    print("Splitting dataset...")
    all_user_ids = dataset.get_user_ids()
    unique_users = list(set(all_user_ids))

    train_users, test_users = train_test_split(
        unique_users, test_size=0.2, random_state=seed
    )

    train_indices = [i for i, user_id in enumerate(all_user_ids) if user_id in train_users]
    test_indices = [i for i, user_id in enumerate(all_user_ids) if user_id in test_users]

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    print("Training dataset size:", len(train_dataset))
    print("Testing dataset size:", len(test_dataset))

    dataset.train()

    if saveModel:
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=False)

        net = Net().to('cuda')
        trainModel(net, train_dataloader, num_epochs=10, learning_rate=1e-3)

        print("Saving model...")
        save_model(modelPath, net)
    else:
        net = load_model(modelPath)
        if net is None:
            print("Model not found")
            exit(1)
        net.to('cuda')

    dataset.eval()

    gallery, test = split_dataset_gallery_test(test_dataset, gallery_ratio=0.6, seed=seed)

    gallery = Subset(test_dataset, gallery)
    test = Subset(test_dataset, test)

    gallery_dataset = DataLoader(dataset=gallery, batch_size=32, shuffle=False, num_workers=4, pin_memory=False)
    test_dataset = DataLoader(dataset=test, batch_size=32, shuffle=False, num_workers=4, pin_memory=False)

    print("Testing identification system...")
    print(testIdentificationSystem(net, test_dataset, gallery_dataset))


