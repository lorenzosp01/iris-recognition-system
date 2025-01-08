import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as tt
from lib.cnn import Net
from lib.cnn_utils import trainModel, save_model, load_model, identification_test_all_vs_all, verification_all_vs_all, \
    generate_embeddings, identification_test_probe_vs_gallery, verification_probe_vs_gallery
from src.data.CasiaIrisDataset import CasiaIrisDataset, split_dataset_gallery_test
from src.data.datasetUtils import splitDataset
from src.utils.plotting import plot_far_frr_roc


if __name__=="__main__":
    datasetPath = "F:\\Dataset\\Casia"

    loadModel = True
    training = False
    testing = True
    modelPath = "..\\models\\modelFullEyeMargin0.2.pth"

    # Calculate the padding
    original_width, original_height = (512, 128)
    desired_size = max(original_width, original_height)  # Make the image square
    top_bottom_padding = (desired_size - original_height) // 2  # Even padding for top and bottom
    left_right_padding = (desired_size - original_width) // 2  # Even padding for left and right

    # Define the transform
    transformPadding = tt.Compose([
        tt.Pad((left_right_padding, top_bottom_padding, left_right_padding, top_bottom_padding), fill=0),
        # Add black padding
        tt.Resize((256, 256)),  # Resize to 256x256
        tt.ToTensor(),  # Convert to tensor
    ])

    transform = tt.Compose([
        tt.ToTensor(),  # Convert to tensor
    ])

    dataset = CasiaIrisDataset(datasetPath, transform=[transform], centered=True)

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
        trainModel(net, train_dataloader, val_dataloader, num_epochs=4, epoch_checkpoint=2, margin=0.5)

        print("Saving model...")
        save_model(modelPath, net)

    if testing:
        ## Test the model with all vs all  ------------------------------------------------------------------------------------
        dataset.eval()

        all_vs_all_dataset = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=False)

        net.eval()

        embedding_list, labels_list = generate_embeddings(net, all_vs_all_dataset)

        embedding_array = embedding_list.numpy()  # Convert the tensor to numpy array for cdist
        M = - cosine_similarity(embedding_array)
        M = (np.round(M, 4) + 1) / 2  # Normalize the cosine similarity to [0, 1]

        thresholds, DIR, GRR, FAR, FRR = identification_test_all_vs_all(M, labels_list)

        plot_far_frr_roc(thresholds, FAR, FRR, GRR, DIR=np.array(DIR))

        GARs, FARs, FRRs, GRRs = verification_all_vs_all(M, labels_list)

        plot_far_frr_roc(thresholds, FARs, FRRs, GRRs, DIR=None)

        ## Test the model with all vs all  ------------------------------------------------------------------------------------

        gallery, test = split_dataset_gallery_test(test_dataset, gallery_ratio=0.6, seed=20)

        gallery = Subset(test_dataset, gallery)
        test = Subset(test_dataset, test)

        gallery_dataset = DataLoader(dataset=gallery, batch_size=64, shuffle=False, num_workers=4, pin_memory=False)
        probe_dataset = DataLoader(dataset=test, batch_size=64, shuffle=False, num_workers=4, pin_memory=False)

        embedding_list_gallery, labels_list_gallery = generate_embeddings(net, gallery_dataset)
        embedding_list_probe, labels_list_probe = generate_embeddings(net, probe_dataset)

        embedding_list_gallery = embedding_list_gallery.numpy()
        embedding_list_probe = embedding_list_probe.numpy()

        M = - cosine_similarity(embedding_list_gallery, embedding_list_probe)
        M = (np.round(M, 4) + 1) / 2

        thresholds, DIR, GRR, FAR, FRR = identification_test_probe_vs_gallery(M, labels_list_probe, labels_list_gallery)

        plot_far_frr_roc(thresholds, FAR, FRR, GRR, DIR=np.array(DIR))

        GARs, FARs, FRRs, GRRs = verification_probe_vs_gallery(M, labels_list_probe, labels_list_gallery)

        plot_far_frr_roc(thresholds, FARs, FRRs, GRRs, DIR=None)





