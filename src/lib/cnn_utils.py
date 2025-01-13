import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.lib.cnn import Net


def save_model(model_path, model):
    # Get the directory from the file path
    directory = os.path.dirname(model_path)

    # Check if the directory exists
    if not os.path.exists(directory):
        # Create the directory if it does not exist
        os.makedirs(directory)
        print(f"Directory created: {directory}")

    torch.save(model.state_dict(), model_path)


def load_model(model_path):
    try:
        model = Net()
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        return model
    except FileNotFoundError:
        return None


def trainModel(net, train_dl, val_dl, num_epochs, learning_rate=1e-3, epoch_checkpoint=1, margin=0.2):
    # Optimizer and loss function initialization
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss_fn = torch.nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y),
                                                     margin=margin)

    # Set the model to training mode
    net.train()
    loss_values = []
    val_loss = []
    print("Starting training...")

    for epoch in range(num_epochs):

        # Initialize epoch loss for logging
        epoch_loss = 0
        # Loop through the training data loader
        for batch in tqdm(train_dl, desc=f"Processing batches in epoch {epoch}", unit="batch"):
            # Move data to device (GPU or CPU)
            anchors = batch[0].to('cuda')
            positives = batch[1].to('cuda')
            negatives = batch[2].to('cuda')

            optimizer.zero_grad()  # Zero out gradients from the previous step

            # Forward pass
            anchor_outputs = net(anchors)
            positive_outputs = net(positives)
            negative_outputs = net(negatives)
            del anchors, positives, negatives

            # Calculate the triplet loss
            loss = loss_fn(anchor_outputs, positive_outputs, negative_outputs)
            del anchor_outputs, positive_outputs, negative_outputs
            # Backward pass and optimization step
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            del batch
            torch.cuda.empty_cache()

        loss_values.append(epoch_loss / len(train_dl))
        # Optional: Print epoch loss after processing all batches
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_dl)}")

        if epoch % epoch_checkpoint == 0 and epoch != 0:
            val_loss.append(validateModel(net, val_dl, loss_fn))
            net.train()
            save_model(
                f"..\\models\\checkpoints\\model_epoch_{epoch+10}_margin_{margin}_loss_{epoch_loss / len(train_dl)}.pth",
                net)

    plt.figure()
    plt.plot(loss_values, label=f'Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

    return loss_values, val_loss


def validateModel(net, val_dl, loss_fn):
    net.eval()
    val_loss = 0

    with torch.no_grad():
        for batch in tqdm(val_dl, desc=f"Validating model", unit="batch"):
            anchors = batch[0].to('cuda')
            positives = batch[1].to('cuda')
            negatives = batch[2].to('cuda')

            anchor_outputs = net(anchors)
            positive_outputs = net(positives)
            negative_outputs = net(negatives)
            del anchors, positives, negatives

            loss = loss_fn(anchor_outputs, positive_outputs, negative_outputs)
            del anchor_outputs, positive_outputs, negative_outputs
            val_loss += loss.item()

            del batch
            torch.cuda.empty_cache()

    avg_val_loss = val_loss / len(val_dl)
    print(f"Validation Loss: {avg_val_loss}")
    return avg_val_loss


def cosine_distance(x1, x2):
    """Compute cosine distance between two tensors"""
    # Ensure the inputs are 2D (batch_size x embedding_size)
    # F.cosine_similarity expects tensors to have the same shape or be broadcastable
    x1 = x1.unsqueeze(0) if x1.dim() == 1 else x1
    x2 = x2.unsqueeze(0) if x2.dim() == 1 else x2

    sim = F.cosine_similarity(x1, x2, dim=-1)  # Compute cosine similarity over the last dimension
    return 1 - sim  # Cosine distance is 1 - cosine similarity


def generate_embeddings(net, dataloader):
    embedding_list = []
    labels_list = []
    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Generating embeddings", total=len(dataloader)):
            images = images.to('cuda')
            embedding_list.append(net(images).cpu())
            labels_list.extend(labels)
            del images
            torch.cuda.empty_cache()

    return torch.cat(embedding_list, dim=0), np.array(labels_list)


def identification_test_all_vs_all(M, labels_list, threshold_step=0.001, log=False):
    DIR = []
    GRR = []
    FAR = []
    FRR = []
    THS = []
    n = len(labels_list)
    thr = 0

    # Calcolo dei totali per GAR e FAR
    # unique_labels = set(labels)
    while thr <= 1:
        THS.append(thr)
        DI = np.zeros(n)
        GR = 0
        FA = 0
        TG = TI = (n - 1)

        for i in range(n):
            label_i = labels_list[i]
            row_i = M[i, :]
            row_i = np.delete(row_i, i)
            indexes_of_sorted_row_i = np.argsort(row_i)
            min_index = indexes_of_sorted_row_i[0]

            if row_i[min_index] <= thr:
                if labels_list[i] == labels_list[min_index]:
                    DI[0] += 1

                    # Check for impostor case
                    impostor_found = False
                    for k in indexes_of_sorted_row_i:
                        if row_i[k] <= thr and labels_list[k] != label_i:
                            FA += 1
                            impostor_found = True
                            break
                    if not impostor_found:
                        GR += 1
                else:
                    FA += 1
                    for k in indexes_of_sorted_row_i:
                        if row_i[k] <= thr and labels_list[k] == label_i:
                            DI[k] += 1
                            break
            else:
                GR += 1

        dir_t = np.zeros(n)
        for k in range(n):
            dir_t[k] = (DI[k] / TG) + dir_t[max(k - 1, 0)]

        DIR.append(dir_t)
        GRR.append(GR / TI)
        FAR.append(FA / TI)
        FRR.append(1 - dir_t[0])

        if log:
            print("Threshold:", thr)
            print("DIR[0]:", dir_t[0])
            print("GRR:", GR / TI)
            print("FAR:", FA / TI)
            print("FRR:", 1 - dir_t[0])
            print("--------------")
        thr += threshold_step

    return THS, DIR, GRR, FAR, FRR


def identification_test_probe_vs_gallery(M, labels_lists_probe, labels_lists_gallery, threshold_step=0.001, log=False):
    DIR = []
    GRR = []
    FAR = []
    FRR = []
    THS = []
    n = len(labels_lists_probe)
    thr = 0

    # Calcolo dei totali per GAR e FAR
    # unique_labels = set(labels)
    while thr <= 1 + threshold_step:
        THS.append(thr)
        DI = np.zeros(len(labels_lists_gallery))
        GR = 0
        FA = 0
        TG = TI = n
        for i in range(n):
            label_i = labels_lists_probe[i]

            row_i = M[i, :]
            sorted_row_i = np.argsort(row_i)
            min_index = sorted_row_i[0]

            if row_i[min_index] <= thr:
                if label_i == labels_lists_gallery[min_index]:
                    DI[0] += 1
                    for k in sorted_row_i:
                        if row_i[k] <= thr and labels_lists_gallery[k] != label_i:
                            FA += 1
                            break
                else:
                    FA += 1
                    for k in sorted_row_i:
                        if row_i[k] <= thr and labels_lists_gallery[k] == label_i:
                            DI[k] += 1
                            break
            else:
                GR += 1

        dir_t = np.zeros(len(labels_lists_gallery))
        for i in range(len(labels_lists_gallery)):
            dir_t[i] = (DI[i] / TG) + dir_t[max(i - 1, 0)]

        DIR.append(dir_t)
        GRR.append(GR / TI)
        FAR.append(FA / TI)
        FRR.append(1 - dir_t[0])

        if log:
            print("Threshold:", thr)
            print("DIR[0]:", dir_t[0])
            print("GRR:", GR / TI)
            print("FAR:", FA / TI)
            print("FRR:", 1 - dir_t[0])
            print("--------------")
        thr += threshold_step

    return THS, DIR, GRR, FAR, FRR


def verification_all_vs_all(M, labels_list, threshold_step=0.001, log=False):
    # Initialize result lists for thresholds
    TG = len(labels_list)  # Total genuine pairs (one per user)
    TI = TG * (len(set(labels_list)) - 1)  # Total impostor pairs

    # Initialize metrics
    GARs, FARs, FRRs, GRRs = [], [], [], []

    thr = 0
    while thr <= 1:
        GA, FA, FR, GR = 0, 0, 0, 0

        for i in range(len(M)):
            genuine_label = labels_list[i]
            min_distances = {}

            # Iterate over all other groups (columns) with the same label
            for j in range(len(M)):
                if i == j:
                    continue

                label_j = labels_list[j]
                if label_j not in min_distances:
                    min_distances[label_j] = M[i, j]
                else:
                    min_distances[label_j] = min(min_distances[label_j], M[i, j])

            for label, diff in min_distances.items():
                if diff <= thr:
                    if label == genuine_label:
                        GA += 1
                    else:
                        FA += 1
                else:
                    if label == genuine_label:
                        FR += 1
                    else:
                        GR += 1

        # Calculate rates
        GARs.append(GA / TG if TG > 0 else 0)
        FARs.append(FA / TI if TI > 0 else 0)
        FRRs.append(FR / TG if TG > 0 else 0)
        GRRs.append(GR / TI if TI > 0 else 0)

        if log:
            print(
                f"Threshold: {thr:.2f} | GAR: {GARs[-1]:.4f}, FAR: {FARs[-1]:.4f}, FRR: {FRRs[-1]:.4f}, GRR: {GRRs[-1]:.4f}")

        thr += threshold_step

    return GARs, FARs, FRRs, GRRs


def verification_probe_vs_gallery(M, labels_lists_probe, labels_lists_gallery, threshold_step=0.001, log=False):
    # Initialize result lists for thresholds
    TG = len(labels_lists_probe)  # Total genuine pairs (one per user)
    TI = TG * (len(set(labels_lists_probe)) - 1)  # Total impostor pairs

    # Initialize metrics
    GARs, FARs, FRRs, GRRs = [], [], [], []

    thr = 0
    while thr <= 1 + threshold_step:
        GA, FA, FR, GR = 0, 0, 0, 0

        for i in range(len(labels_lists_probe)):
            genuine_label = labels_lists_probe[i]
            min_distances = {}

            # Iterate over all other groups (columns) with the same label
            for j in range(len(labels_lists_gallery)):
                if i == j:
                    continue

                label_j = labels_lists_gallery[j]
                if label_j not in min_distances:
                    min_distances[label_j] = M[i, j]
                else:
                    min_distances[label_j] = min(min_distances[label_j], M[i, j])

            for label, diff in min_distances.items():
                if diff <= thr:
                    if label == genuine_label:
                        GA += 1
                    else:
                        FA += 1
                else:
                    if label == genuine_label:
                        FR += 1
                    else:
                        GR += 1

        # Calculate rates
        GARs.append(GA / TG if TG > 0 else 0)
        FARs.append(FA / TI if TI > 0 else 0)
        FRRs.append(FR / TG if TG > 0 else 0)
        GRRs.append(GR / TI if TI > 0 else 0)

        if log:
            print(
                f"Threshold: {thr:.2f} | GAR: {GARs[-1]:.4f}, FAR: {FARs[-1]:.4f}, FRR: {FRRs[-1]:.4f}, GRR: {GRRs[-1]:.4f}")

        thr += threshold_step

    return GARs, FARs, FRRs, GRRs
