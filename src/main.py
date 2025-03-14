import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as tt
from lib.cnn import Net
from lib.cnn_utils import trainModel, save_model, load_model, identification_test_all_vs_all, verification_all_vs_all, \
    generate_embeddings, identification_test_probe_vs_gallery, verification_probe_vs_gallery
from src.data.CasiaIrisDataset import CasiaIrisDataset
from src.data.datasetUtils import splitDataset, split_dataset_gallery_test
from src.utils.plotting import plot_far_frr_roc


if __name__=="__main__":
    datasetPath = "F:\\Dataset\\Casia"

    loadModel = True
    training = False
    testing = True
    modelPath = "..\\models\\modelNormalizedEyeMargin040.pth"

    transform = tt.Compose([
        tt.ToTensor(),  # Convert to tensor
    ])

    dataset = CasiaIrisDataset(datasetPath, transform=[transform], normalized=True)

    train_dataset, val_dataset, test_dataset = splitDataset(dataset, 0.2, 0.1)

    if loadModel:
        net = load_model(modelPath)
        if net is None:
            print("Model not found")
            exit(1)
        net.to('cuda')
    else:
        net = Net().to('cuda')

    if training:
        dataset.train()

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=False)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=False)

        print("Training model...")
        training_loss, validation_loss = trainModel(net, train_dataloader, val_dataloader, num_epochs=20, epoch_checkpoint=2, margin=0.4)

        print("Saving model...")
        save_model(modelPath, net)

    if testing:

        dataset.eval()
        ## Test the model with all vs all  ------------------------------------------------------------------------------------
        all_vs_all_dataset = DataLoader(dataset=test_dataset, batch_size=64, num_workers=8, pin_memory=True)

        net.eval()

        embedding_list, labels_list = generate_embeddings('cuda', net, all_vs_all_dataset)

        embedding_array = embedding_list.numpy()  # Convert the tensor to numpy array for cdist
        M = - cosine_similarity(embedding_array)
        M = np.round(M, 3)
        M = (M + 1) / 2  # Normalize the cosine similarity to [0, 1]

        thresholds, DIR, GRR, FAR, FRR = identification_test_all_vs_all(M, labels_list)

        plot_far_frr_roc(thresholds, FAR, FRR, GRR, DIR=np.array(DIR), roc=True, titleRoc="ROC Curve - Identification - All vs all", titleEer="FAR, FRR, Rank-1, GRR and EER - Identification - All vs all")

        GARs, FARs, FRRs, GRRs = verification_all_vs_all(M, labels_list)

        plot_far_frr_roc(thresholds, FARs, FRRs, GRRs, DIR=None, titleEer="FAR, FRR, Rank-1, GRR and EER - Verification - All vs all")

        ## Test the model with all vs all  ------------------------------------------------------------------------------------

        gallery, test = split_dataset_gallery_test(test_dataset, gallery_ratio=0.6, seed=20)

        gallery = Subset(test_dataset, gallery)
        test = Subset(test_dataset, test)

        gallery_dataset = DataLoader(dataset=gallery, batch_size=64, shuffle=False, num_workers=8, pin_memory=False)
        probe_dataset = DataLoader(dataset=test, batch_size=64, shuffle=False, num_workers=8, pin_memory=False)

        embedding_list_gallery, labels_list_gallery = generate_embeddings('cuda', net, gallery_dataset)
        embedding_list_probe, labels_list_probe = generate_embeddings('cuda', net, probe_dataset)

        embedding_list_gallery = embedding_list_gallery.numpy()
        embedding_list_probe = embedding_list_probe.numpy()

        M = - cosine_similarity(embedding_list_probe, embedding_list_gallery)
        M = (np.round(M, 4) + 1) / 2

        thresholds, DIR, GRR, FAR, FRR = identification_test_probe_vs_gallery(M, labels_list_probe, labels_list_gallery)

        plot_far_frr_roc(thresholds, FAR, FRR, GRR, DIR=np.array(DIR), roc=True, titleRoc="ROC Curve - Identification - Probe vs Gallery", titleEer="FAR, FRR, Rank-1, GRR and EER - Indentification - Probe vs Gallery")

        GARs, FARs, FRRs, GRRs = verification_probe_vs_gallery(M, labels_list_probe, labels_list_gallery)

        plot_far_frr_roc(thresholds, FARs, FRRs, GRRs, DIR=None, titleEer="FAR, FRR, Rank-1, GRR and EER - Verification - Probe vs Gallery")





