import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from lib.cnn import Net
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score, roc_curve, auc
)

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
            saveModel(f"./models/model-epoch-{epoch}.pth", net)

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

def testIdentificationSystem(net, test_dl, gallery_dl):
    TG = len(test_dl.dataset)  # Total genuine samples
    TI = len(gallery_dl.dataset)  # Total impostor samples

    # Initialize metrics
    DI = {}  # Dictionary to store DI(t, k)
    FA = {}  # False Accept for each threshold
    GR = {}  # Genuine Reject for each threshold

    # Extract features for the gallery
    gallery_features = []
    gallery_labels = []
    for batch in gallery_dl:
        images, labels = batch[0].to('cuda'), batch[1].to('cuda')
        with torch.no_grad():
            features = net(images)
        gallery_features.append(features)
        gallery_labels.extend(labels)
        del batch, images, labels
        torch.cuda.empty_cache()

    gallery_features = torch.cat(gallery_features).cuda()
    gallery_labels = torch.tensor(gallery_labels).cuda()

    # Loop over the test dataset
    for batch in test_dl:
        test_images, test_labels = batch[0].to('cuda'), batch[1].to('cuda')
        with torch.no_grad():
            test_features = net(test_images)
        del test_images

        for i, test_feature in enumerate(test_features):
            label_i = test_labels[i]

            # Compute cosine similarity with gallery features
            similarities = F.cosine_similarity(test_feature.unsqueeze(0), gallery_features, dim=1)
            distances = 1 - similarities  # Convert to distance

            # Sort gallery entries by distance
            sorted_indices = torch.argsort(distances)
            sorted_distances = distances[sorted_indices]
            sorted_labels = gallery_labels[sorted_indices]

            for t in torch.arange(0, 1.1, 0.1):  # Thresholds from 0 to 1 with step 0.1
                if t not in DI:
                    DI[t] = [0] * 10
                    FA[t] = 0
                    GR[t] = 0

                k = 1

                # Check the closest match
                if sorted_distances[0] <= t:
                    if sorted_labels[0] == label_i:
                        DI[t][0] += 1  # Genuine match at rank 1
                    else:
                        FA[t] += 1  # Impostor accepted

                # Look for higher ranks
                while k < 10:
                    if k >= len(sorted_distances):
                        break

                    if sorted_labels[k] == label_i and sorted_distances[k] <= t:
                        DI[t][k] += 1  # Genuine match at rank k
                        break
                    elif sorted_labels[k] != label_i and sorted_distances[k] <= t:
                        FA[t] += 1  # Impostor accepted
                    k += 1

                if k == len(sorted_distances) or sorted_distances[k] > t:
                    GR[t] += 1  # Genuine reject
        del test_labels
        del batch
        torch.cuda.empty_cache()

    # Calculate metrics
    DIR = {t: [DI[t][k] / TG if k < len(DI[t]) else 0 for k in range(10)] for t in DI}
    FRR = {t: 1 - DIR[t][0] for t in DIR}
    FAR = {t: FA[t] / TI for t in DI}
    GRR = {t: GR[t] / TI for t in DI}

    # Plot FAR and FRR vs. Threshold
    thresholds = list(DIR.keys())
    FRR_values = [FRR[t] for t in thresholds]
    FAR_values = [FAR[t] for t in thresholds]

    plt.figure(figsize=(10, 5))

    # Plot FAR and FRR
    plt.plot(thresholds, FRR_values, label='FRR(t)', color='red')
    plt.plot(thresholds, FAR_values, label='FAR(t)', color='blue')
    plt.xlabel('Threshold (t)')
    plt.ylabel('Error Rate')
    plt.title('FRR and FAR vs Threshold')
    plt.axvline(x=thresholds[FRR_values.index(min(FRR_values, key=lambda x: abs(x - FAR_values[FRR_values.index(x)])))], color='black', linestyle='--', label='EER')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot ROC Curve
    plt.figure(figsize=(10, 5))
    plt.plot(FAR_values, [1 - fr for fr in FRR_values], label='ROC Curve', color='green')
    plt.xlabel('FAR')
    plt.ylabel('True Acceptance Rate (1 - FRR)')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid()
    plt.show()

    return DIR, FRR, FAR, GRR
