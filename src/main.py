import numpy as np
from skimage.filters.rank import threshold
from torch.utils.data import DataLoader
from torchvision import transforms as tt
from lib.cnn import Net
from lib.cnn_utils import trainModel, save_model, load_model, identification_test_all_vs_all
from src.data.CasiaIrisDataset import CasiaIrisDataset
from src.data.datasetUtils import splitDataset
from src.utils.plotting import plot_far_frr_roc

if __name__=="__main__":
    datasetPath = "F:\\Dataset\\Casia"

    loadModel = True
    training = False
    testing = True
    modelPath = "..\\models\\checkpoints\\model_epoch_2_margin_0.5_loss_0.041099133393705176.pth"

    # Calculate the padding
    original_width, original_height = (512, 128)
    desired_size = max(original_width, original_height)  # Make the image square
    top_bottom_padding = (desired_size - original_height) // 2  # Even padding for top and bottom
    left_right_padding = (desired_size - original_width) // 2  # Even padding for left and right

    # Define the transform
    transformNormalized = tt.Compose([
        #tt.Pad((left_right_padding, top_bottom_padding, left_right_padding, top_bottom_padding), fill=0),
        # Add black padding
        #tt.Resize((256, 256)),  # Resize to 256x256
        tt.ToTensor(),  # Convert to tensor
    ])

    dataset = CasiaIrisDataset(datasetPath, transform=[transformNormalized], centered=True)

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
        trainModel(net, train_dataloader, val_dataloader, num_epochs=16, epoch_checkpoint=2, margin=0.5)

        print("Saving model...")
        save_model(modelPath, net)

    if testing:
        dataset.eval()

        all_vs_all_dataset = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=False)

        thresholds, DIR, GRR, FAR, FRR = identification_test_all_vs_all(net, all_vs_all_dataset)

        plot_far_frr_roc(thresholds, FAR, FRR, GRR, np.array(DIR))


    #gallery, test = split_dataset_gallery_test(test_dataset, gallery_ratio=0.6, seed=seed)

    #gallery = Subset(test_dataset, gallery)
    #test = Subset(test_dataset, test)

    #gallery_dataset = DataLoader(dataset=gallery, batch_size=32, shuffle=False, num_workers=4, pin_memory=False)
    #probe_dataset = DataLoader(dataset=test, batch_size=32, shuffle=False, num_workers=4, pin_memory=False)

    #print("Testing identification system...")
    #metrics = identification_test(net, probe_dataset, gallery_dataset)






