import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from torchvision import transforms as tt
from data.CasiaIrisDataset import CasiaIrisDataset, split_dataset_gallery_test
from lib.cnn import Net
from lib.cnn_utils import trainModel, save_model, load_model, identification_test_all_vs_all
from src.lib.cnn_utils import plot_far_frr_roc

if __name__=="__main__":
    datasetPath = "F:\\Dataset\\Casia"

    saveModel = False
    modelPath = "..\\models\\model.pth"

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
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=False)

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

    all_vs_all_dataset = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=False)

    DI, FA, GR, DIR, FRR, FAR, GRR = identification_test_all_vs_all(net, all_vs_all_dataset, threshold_step=0.01)

    plot_far_frr_roc(FAR, FRR, GRR)


    #gallery, test = split_dataset_gallery_test(test_dataset, gallery_ratio=0.6, seed=seed)

    #gallery = Subset(test_dataset, gallery)
    #test = Subset(test_dataset, test)

    #gallery_dataset = DataLoader(dataset=gallery, batch_size=32, shuffle=False, num_workers=4, pin_memory=False)
    #probe_dataset = DataLoader(dataset=test, batch_size=32, shuffle=False, num_workers=4, pin_memory=False)

    #print("Testing identification system...")
    #metrics = identification_test(net, probe_dataset, gallery_dataset)






