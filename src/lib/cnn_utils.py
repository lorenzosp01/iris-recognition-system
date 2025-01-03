import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics import auc, roc_curve, roc_auc_score
from tqdm import tqdm
import os
from lib.cnn import Net


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

def trainModel(net, train_dl, num_epochs, learning_rate=1e-3):
    # Optimizer and loss function initialization
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss_fn = torch.nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y),
                                                     margin=0.2)

    # Set the model to training mode
    net.train()
    loss_values = []
    print("Starting training...")

    for epoch in range(num_epochs):

        if epoch%3 == 0:
            save_model(f"./models/model-epoch-{epoch}.pth", net)

        # Initialize epoch loss for logging
        epoch_loss = 0
        # Loop through the training data loader
        for batch in tqdm(train_dl, desc=f"Processing batches in epoch {epoch + 1}", unit="batch"):

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
    plt.figure()
    plt.plot(loss_values, label=f'Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()


def cosine_distance(x1, x2):
    """Compute cosine distance between two tensors"""
    # Ensure the inputs are 2D (batch_size x embedding_size)
    # F.cosine_similarity expects tensors to have the same shape or be broadcastable
    x1 = x1.unsqueeze(0) if x1.dim() == 1 else x1
    x2 = x2.unsqueeze(0) if x2.dim() == 1 else x2

    sim = F.cosine_similarity(x1, x2, dim=-1)  # Compute cosine similarity over the last dimension
    return 1 - sim  # Cosine distance is 1 - cosine similarity

def generate_embeddings(net, dataloader, device='cuda'):
    embeddings_list = list()
    labels_list = list()
    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            embeddings = net(images)
            embeddings_list.append(embeddings)
            labels_list.append(labels)
            del images, embeddings
            torch.cuda.empty_cache()
    return torch.cat(embeddings_list, dim=0), torch.cat(labels_list, dim=0)

def plot_far_frr_roc(FAR, FRR, GRR, threshold_step=0.01):
    print(type(FAR))
    print(type(FRR))
    # Ensure FAR and FRR are arrays
    if isinstance(FAR, dict):
        FAR = np.array(list(FAR.values()))
    elif not isinstance(FAR, np.ndarray):
        FAR = np.array(FAR)

    if isinstance(FRR, dict):
        FRR = np.array(list(FRR.values()))
    elif not isinstance(FRR, np.ndarray):
        FRR = np.array(FRR)

    if isinstance(GRR, dict):
        GRR = np.array(list(GRR.values()))
    elif not isinstance(GRR, np.ndarray):
        GRR = np.array(GRR)

    # Generate synthetic thresholds
    t = np.arange(0, 1.0+threshold_step, threshold_step)

    # Find Equal Error Rate (EER)
    bestMatch = 1
    for i in range(len(FAR)):
        if FRR[i] == FAR[i]:
            eer = FRR[i]
            eer_threshold = t[i]
            break
        elif abs(FRR[i] - FAR[i]) < bestMatch:
                bestMatch = abs(FRR[i] - FAR[i])
                eer = (FRR[i] + FAR[i]) / 2
                eer_threshold = t[i]

    # Find ZeroFAR and ZeroFRR points
    zero_far_index = np.where(FAR <= 0.01)[0][0] if np.any(FAR <= 0.01) else -1
    zero_frr_index = np.where(FRR <= 0.01)[0][-1] if np.any(FRR <= 0.01) else -1

    # Plot FAR and FRR curves
    plt.figure()
    plt.plot(t, FAR, label="FAR(t)", color="blue")
    plt.plot(t, FRR, label="FRR(t)", color="black")

    # Highlight EER, ZeroFAR, and ZeroFRR points
    plt.scatter(eer_threshold, eer, color="red", label="EER: {:.2f}".format(eer))
    if zero_far_index != -1:
        plt.scatter(t[zero_far_index], FRR[zero_far_index], color="blue", label="ZeroFAR")
    if zero_frr_index != -1:
        plt.scatter(t[zero_frr_index], FAR[zero_frr_index], color="blue", label="ZeroFRR")

    # Annotations
    plt.annotate("EER", (eer_threshold, eer), textcoords="offset points", xytext=(-20, 10), ha='center', color='red')
    if zero_far_index != -1:
        plt.annotate("ZeroFAR", (t[zero_far_index], FAR[zero_far_index]), textcoords="offset points", xytext=(-30, -15),
                     ha='center', color='blue')
    if zero_frr_index != -1:
        plt.annotate("ZeroFRR", (t[zero_frr_index], FRR[zero_frr_index]), textcoords="offset points", xytext=(-30, -15),
                     ha='center', color='blue')

    # Plot formatting for FAR, FRR
    plt.axvline(x=eer_threshold, linestyle="--", color="gray", alpha=0.7)
    plt.axhline(y=eer, linestyle="--", color="gray", alpha=0.7)
    plt.xlabel("t (Threshold)")
    plt.ylabel("Error")
    plt.title("FAR, FRR, and EER Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # ROC Curve Plot (lower plot)
    tpr = 1 - FRR  # True Positive Rate
    # Sort the FAR and TAR values (this ensures FAR is increasing)
    sorted_indices = np.argsort(FAR)
    FAR = FAR[sorted_indices]
    tpr_sorted = tpr[sorted_indices]
    # AUC calculation
    roc_auc = auc(FAR, tpr_sorted)

    plt.figure()
    plt.plot(FAR, tpr_sorted, color='green', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.xlabel("False Positive Rate (FAR)")
    plt.ylabel("True Positive Rate (1 - FRR)")
    plt.ylim([0, 1.05])
    plt.title("ROC Curve")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7)  # Diagonal line for random classifier
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # GRR Plot in a separate graph if provided
    plt.figure()
    plt.plot(t, GRR, label="GRR(t)", color="green")
    plt.xlabel("t (Threshold)")
    plt.ylabel("Recognition Rate")
    plt.title("Genuine Recognition Rate (GRR)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

def identification_test_all_vs_all(net, all_vs_all_dataset, threshold_step=0.01):
    # Move the model to GPU
    net.to('cuda')
    net.eval()

    embedding_list = []
    labels_list = []
    with torch.no_grad():
        for images, labels, _ in tqdm(all_vs_all_dataset, desc="Generating embeddings", total=len(all_vs_all_dataset)):
            images = images.to('cuda')
            embedding_list.append(net(images).cpu())
            labels_list.extend(labels)
            del images
            torch.cuda.empty_cache()

    embedding_list = torch.cat(embedding_list, dim=0)
    labels_list = np.array(labels_list)

    G = len(labels_list)

    embedding_array = embedding_list.numpy()  # Convert the tensor to numpy array for cdist
    distance_matrix = cdist(embedding_array, embedding_array, metric='cosine')

    # Initialize metrics for each threshold
    thresholds = np.arange(0, 1.0+threshold_step, threshold_step)

    DI = {t: [] for t in thresholds}  # Genuine case detection counts
    FA = {t: 0 for t in thresholds}  # False accept counts
    GR = {t: 0 for t in thresholds}  # Genuine reject counts
    DIR = {t: [] for t in thresholds}  # Genuine identification rate
    FRR = {t: [] for t in thresholds}  # False rejection rate
    FAR = {t: [] for t in thresholds}  # False acceptance rate
    GRR = {t: [] for t in thresholds}  # Genuine rejection rate

    # Iterate over each threshold0
    for t in thresholds:
        for i in range(G):
            # Sort distances for each row (ignoring self-comparison)
            sorted_indices = np.argsort(distance_matrix[i])
            sorted_distances = distance_matrix[i, sorted_indices]
            sorted_labels = labels_list[sorted_indices]

            # Remove the identity element (self comparison)
            sorted_distances = sorted_distances[1:]
            sorted_labels = sorted_labels[1:]

            # Check if the first element is a potential accept
            if sorted_distances[0] <= t:
                # Genuine case detected
                if labels_list[i] == sorted_labels[0]:
                    if not DI[t]:
                        DI[t].append(1)  # Genuine identification at rank 1
                    else:
                        DI[t][0] += 1  # Genuine identification at rank 1
                    # Parallel impostor case: find first non-matching label with distance <= t
                    impostor_found = False
                    for k in range(1, len(sorted_distances)):
                        if sorted_labels[k] != labels_list[i] and sorted_distances[k] <= t:
                            FA[t] += 1  # False accept
                            impostor_found = True
                            break
                    if not impostor_found:
                        GR[t] += 1  # Genuine reject

                # If not the first rank but a genuine case, look for higher ranks
                else:
                    impostor_found = False
                    for k in range(1, len(sorted_distances)):
                        if len(DI[t]) < k+1:
                            DI[t].append(0)
                        if labels_list[i] == sorted_labels[k] and sorted_distances[k] <= t:
                            DI[t][k] += 1  # Higher rank genuine identification
                            impostor_found = True
                            break
                    if not impostor_found:
                        FA[t] += 1  # False accept if no match at higher ranks
            else:
                GR[t] += 1  # Impostor case directly counted

        # Compute DIR, FRR, FAR, GRR for each threshold t
        # DIR(t, k) = DI(t, k) / TG, where TG is the total number of genuine cases
        TG = len(labels_list)
        DIR[t].append(DI[t][0] / TG)  # Genuine identification rate at rank 1
        FRR[t] = 1 - DIR[t][0]  # False rejection rate

        # FAR(t) = FA / TI, where TI is the total number of impostors
        TI = TG
        FAR[t] = FA[t] / TI  # False acceptance rate

        # GRR(t) = GR / TI
        GRR[t] = GR[t] / TI  # Genuine rejection rate

        # Calculate DIR(t, k) for higher ranks
        k = 1
        while k <= len(DI[t]) and DI[t][k - 1] != 0:  # Ensure we don't go out of bounds
            # For k == 1, we just use DI[t][0] / TG
            if k == 1:
                DIR[t].append(DI[t][k - 1] / TG)
            # For k > 1, we use the previous rank value to calculate cumulative DIR
            elif k > 1 and k - 2 >= 0:
                DIR[t].append((DI[t][k - 1] / TG) + DIR[t][k - 2])
            k += 1

    return DI, FA, GR, DIR, FRR, FAR, GRR









