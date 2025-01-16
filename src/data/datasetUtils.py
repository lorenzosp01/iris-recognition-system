import random
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from collections import defaultdict

def splitDataset(dataset, train_test_split_size=0.2, train_val_split_size=0.1):
    seed = 20
    torch.manual_seed(seed)

    print("Splitting dataset...")

    all_user_ids = dataset.get_user_ids()
    unique_users = list(set(all_user_ids))

    train_and_val_users, test_users = train_test_split(
        unique_users, test_size=train_test_split_size, random_state=seed
    )

    train_users, val_users = train_test_split(
        train_and_val_users, test_size=train_val_split_size, random_state=seed
    )

    train_indices = [i for i, user_id in enumerate(all_user_ids) if user_id in train_users]
    val_users_indices = [i for i, user_id in enumerate(all_user_ids) if user_id in val_users]
    test_indices = [i for i, user_id in enumerate(all_user_ids) if user_id in test_users]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_users_indices)
    test_dataset = Subset(dataset, test_indices)

    print("Training dataset size:", len(train_dataset))
    print("Validation dataset size:", len(val_dataset))
    print("Testing dataset size:", len(test_dataset))

    return train_dataset, val_dataset, test_dataset

def split_dataset_gallery_test(dataset, gallery_ratio=0.4, seed=42, full_probe_ratio=0.05):

    label_to_indices = defaultdict(list)

    for idx, (input_data, label, p) in enumerate(dataset):
        label_to_indices[label].append(idx)

    all_probe_labels = random.sample(list(label_to_indices.keys()),
                                k=max(1, int(len(label_to_indices) * full_probe_ratio)))

    random.seed(seed)
    gallery_indices = []
    test_indices = []

    for label, indices in label_to_indices.items():
        if label in all_probe_labels:
            test_indices.extend(indices)
        else:
            random.shuffle(indices)

            n_gallery = max(1, int(len(indices) * gallery_ratio))

            gallery_indices.extend(indices[:n_gallery])
            test_indices.extend(indices[n_gallery:])

    return gallery_indices, test_indices
