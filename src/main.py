import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from torchvision import transforms as tt
from src.data.CasiaIrisDataset import CasiaIrisDataset
from src.lib.cnn import Net
from src.lib.cnn_utils import trainModel, save_model, testIdentificationSystem, load_model

if __name__=="__main__":
    datasetPath = "F:\\Dataset\\Casia"

    saveModel = True
    modelPath = ".\\models\\model.pth"

    transform = tt.Compose([tt.Resize((128, 128)),
                               tt.ToTensor()])

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
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True, pin_memory_device='cuda')

        net = Net().to('cuda')
        trainModel('cuda', net, train_dataloader, num_epochs=10, learning_rate=1e-3)


        if saveModel:
            print("Saving model...")
            save_model(modelPath, net)
    else:
        net = load_model(modelPath)
        if net is None:
            print("Model not found")
            exit(1)
        net.to('cuda')

    test_dataloader = DataLoader(dataset=test_dataset, batch_size=32, num_workers=8, pin_memory=True, pin_memory_device='cuda')

    print("Testing identification system...")
    testIdentificationSystem('cuda', net, test_dataloader)


