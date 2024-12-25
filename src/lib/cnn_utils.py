import torch
import torch.nn.functional as F
from src.lib.cnn import Net
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
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
    loss_fn = torch.nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y), margin=0.2)

    print("Starting train")
    net.train()
    for epoch in range(num_epochs):
        for batch in train_dl:
            anchors = to_device(batch[0], device)
            positives = to_device(batch[1], device)
            negatives = to_device(batch[2], device)
            print("Passed to device")
            optimizer.zero_grad()

            anchor_outputs = net(anchors)
            positive_outputs = net(positives)
            negative_outputs = net(negatives)
            print("Outputs calculated")

            loss = loss_fn(anchor_outputs, positive_outputs, negative_outputs)
            print("Loss calculated")
            loss.backward()
            print("Backward")
            optimizer.step()
            print("Step")
            ##TODO Porco dio si rompe qua
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

def testIdentificationSystem(device, net, test_dl, threshold=0.35):
    net.eval()  # Modalità di valutazione (disabilita dropout e batch norm)

    all_labels = []  # Etichette reali (identità)
    all_predictions = []  # Predizioni

    with torch.no_grad():
        for batch in test_dl:
            anchors = to_device(batch[0], device)  # Immagine dell'anchor
            anchor_labels = batch[4]  # Etichetta dell'anchor (identità)
            # La parte del percorso anchor_path non è necessaria, lo includiamo solo se vuoi
            # anchor_paths = batch[2]

            # Calcola l'output (embedding) per l'anchor
            anchor_outputs = net(anchors)

            # Confronta con tutte le immagini nel test set (o un subset se troppo grande)
            for test_batch in test_dl:  # Loop attraverso tutto il test set per confronto
                test_anchors = to_device(test_batch[0], device)
                test_labels = test_batch[4]

                # Calcola gli embeddings per gli altri esempi nel test set
                test_outputs = net(test_anchors)

                # Calcola la distanza (1 - cosine similarity) tra l'anchor e gli altri esempi
                distances = 1 - F.cosine_similarity(anchor_outputs, test_outputs)

                # Predizione: se la distanza è sotto la soglia, sono della stessa identità
                predictions = distances < threshold

                # Aggiungi le etichette reali e le predizioni
                all_labels.extend(anchor_labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())

    # Calcola le metriche
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    return accuracy, precision, recall, f1