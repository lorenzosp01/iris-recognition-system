import numpy as np
import torch
import torch.nn.functional as F
from src.lib.cnn import Net
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)



def save_model(model_path, model):
    torch.save(model.state_dict(), model_path)

def load_model(model_path):
    try:
        model = Net()
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        return model
    except FileNotFoundError:
        return None

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def trainModel(device, net, train_dl, num_epochs, learning_rate=1e-3):
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss_fn = torch.nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y))

    print("Start training")
    for epoch in range(num_epochs):
        net.train()
        for batch in train_dl:
            anchors = to_device(batch[0], device)
            positives = to_device(batch[1], device)
            negatives = to_device(batch[2], device)
            optimizer.zero_grad()

            anchor_outputs = net(anchors)
            positive_outputs = net(positives)
            negative_outputs = net(negatives)

            loss = loss_fn(anchor_outputs, positive_outputs, negative_outputs)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
        del loss
        del anchor_outputs
        del positive_outputs
        del negative_outputs
        del anchors
        del positives
        del negatives
        del batch

def testModel(device, net, test_dl):
    y_test = []
    prob = []
    with torch.no_grad():
        for data in test_dl:
            images, labels = data
            images, labels = to_device(images, device), to_device(labels, device)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            y_test.extend(labels.cpu().numpy())
            prob.extend(torch.nn.functional.softmax(outputs.cpu(), dim=1)[:, 1])
        fprChest, tprChest, thresholds = roc_curve(y_test, prob)
        roc_aucChest = roc_auc_score(y_test, prob)

        distances = np.sqrt(fprChest ** 2 + (1 - tprChest) ** 2)
        best_threshold = thresholds[np.argmin(distances)]
        new_preds = [1 if score > best_threshold else 0 for score in prob]

        cm = confusion_matrix(y_test, new_preds)
        dispChest = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NEGATIVE", "POSITIVE"])
        accuracyChest = accuracy_score(y_test, new_preds)
        precisionChest = precision_score(y_test, new_preds)
        recallChest = recall_score(y_test, new_preds)
        f1Chest = f1_score(y_test, new_preds)