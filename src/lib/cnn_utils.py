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

def testIdentificationSystem(net, test_dl, gallery_dl, threshold=0.01):
    """
    Test the identification system and compute EER, FRR, and Rank-1.

    Args:
        net (torch.nn.Module): The trained model used to generate embeddings.
        test_dl (DataLoader): DataLoader for the test dataset.
        gallery_dl (DataLoader): DataLoader for the gallery dataset.
        threshold (float): Threshold for cosine similarity to determine matches.

    Returns:
        dict: A dictionary containing EER, FRR, and Rank-1 metrics.
    """
    net.eval()  # Set the model to evaluation mode

    # Step 1: Generate gallery embeddings
    print("Generating gallery embeddings...")
    gallery_embeddings = []
    gallery_labels = []

    with torch.no_grad():
        for batch in tqdm(gallery_dl, desc="Gallery Embedding Generation"):
            inputs, labels = batch[0].to('cuda'), batch[1]  # Images and labels
            outputs = net(inputs)  # Generate embeddings
            gallery_embeddings.append(outputs.cpu())
            gallery_labels.extend(labels.numpy())

    gallery_embeddings = torch.cat(gallery_embeddings, dim=0)  # Combine into a single tensor
    gallery_labels = torch.tensor(gallery_labels)

    # Step 2: Generate test embeddings and compute matches
    print("Generating test embeddings and evaluating...")
    all_scores = []
    all_ground_truths = []
    rank1_matches = 0
    total_tests = 0

    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Test Evaluation"):
            inputs, labels = batch[0].to('cuda'), batch[1]  # Test images and labels
            test_embeddings = net(inputs)  # Generate embeddings

            # Compare test embeddings with gallery embeddings
            for i, test_embedding in enumerate(test_embeddings):
                # Compute cosine similarity between test embedding and all gallery embeddings
                distances = 1 - F.cosine_similarity(test_embedding.unsqueeze(0), gallery_embeddings.to('cuda'), dim=1)

                # Find the gallery embedding with the minimum distance
                sorted_indices = torch.argsort(distances)
                min_index = sorted_indices[0]

                # Rank-1 evaluation
                if gallery_labels[min_index] == labels[i]:
                    rank1_matches += 1

                total_tests += 1

                # Save the score (distance) and ground truth for EER/FRR calculation
                all_scores.append(distances[min_index].cpu().item())
                all_ground_truths.append((gallery_labels[min_index] == labels[i]).item())

    # Step 3: Compute EER, FRR, and Rank-1
    all_scores = np.array(all_scores)
    all_ground_truths = np.array(all_ground_truths)

    # Rank-1 accuracy
    rank1_accuracy = rank1_matches / total_tests

    # ROC Curve and EER
    fpr, tpr, thresholds = roc_curve(all_ground_truths, -all_scores)  # Negate scores for correct ROC order
    fnr = 1 - tpr
    eer_threshold_index = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_threshold_index] + fnr[eer_threshold_index]) / 2

    # FRR at a specific threshold
    predictions = (all_scores < threshold).astype(int)
    frr = 1 - np.mean(predictions[all_ground_truths == 1])

    # Plot ROC Curve
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

    # Return metrics
    return {
        "EER": eer,
        "FRR": frr,
        "Rank-1": rank1_accuracy
    }
