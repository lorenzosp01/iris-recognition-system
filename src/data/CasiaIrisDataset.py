import json
import random
from PIL import Image
import pandas as pd
import os
from iris import IrisTemplate
from torch.utils.data import Dataset


class CasiaIrisDataset(Dataset):
    def __init__(self, root_dir, transform=[], centered=False, normalized=False, encoding=False):
        self.root_dir = root_dir
        if centered:
            self.image_dir = os.path.join(root_dir, "CASIA-Iris-Thousand-Centered")
        elif normalized:
            self.image_dir = os.path.join(root_dir, "CASIA-Iris-Thousand-Normalized")
        elif encoding:
            self.image_dir = os.path.join(root_dir, "CASIA-Iris-Thousand-Encoded")
        else:
            raise ValueError("No dataset type specified")

        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.labeldict = {}
        self.filedict = {}
        self.train_mode = False
        self.data = pd.read_csv(os.path.join(root_dir, 'iris_thousands_updated.csv'))
        print(f"Dataset size: {len(self.data)}")

        count = 0
        print("Loading dataset...")
        for user in os.listdir(self.image_dir):
            for eye in os.listdir(os.path.join(self.image_dir, user)):
                subdir_path = os.path.join(user, eye)
                for img in os.listdir(os.path.join(self.image_dir, user, eye)):
                    if img.endswith('.jpg'):
                        self.image_paths.append(os.path.join(self.image_dir, user, eye, img))
                        eyeN = 0 if "L" in eye else 1
                        label = int(user) + (eyeN*1000)
                        self.labels.append(label)
                        count += 1

                        if label not in self.labeldict:
                            self.labeldict[label] = (count - 1, count - 1)
                        else:
                            self.labeldict[label] = (self.labeldict[label][0], count - 1)

                        if label not in self.filedict:
                            self.filedict[label] = [os.path.join(subdir_path, img)]
                        else:
                            self.filedict[label].append(os.path.join(subdir_path, img))

    def __len__(self):
        return len(self.image_paths)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

    def get_user_ids(self):
        user_ids = []
        for label in self.labels:
            # Extract the unique user ID from the label
            # Assuming labels are structured such that user ID is encoded directly as the first part of the label
            user_id = label % 1000  # Extracts user ID (ignoring left/right eye distinction)
            user_ids.append(user_id)
        return user_ids

    def __getitem__(self, idx):
        anchor, anchor_label, anchor_path = self.loadItem(idx)
        if not self.train_mode:
            return anchor, anchor_label, anchor_path

        positive_start, positive_end = self.labeldict[anchor_label]
        positive_idx = random.randint(positive_start, positive_end)
        positive, positive_label, _ = self.loadItem(positive_idx)

        if positive_label != anchor_label:
            print("Error: Positive label mismatch")

        while True:
            negative_idx = random.randint(0, len(self.image_paths) - 1)
            negative, negative_label, _ = self.loadItem(negative_idx)
            if negative_label != anchor_label:
                break
        return anchor, positive, negative, anchor_label

    def loadItem(self, idx):
        # Load image file
        image_paths = os.path.join(self.image_dir, self.image_paths[idx])
        # Load label
        label = self.labels[idx]
        if "Encoded" in self.image_dir:
            with open(image_paths) as f:
                image = IrisTemplate.deserialize(json.load(f))
        else:
            image =  Image.open(image_paths)

            # Apply transformations
            if len(self.transform) > 0:
                transform_idx = idx // len(self.data)
                image = self.transform[transform_idx](image)

        return image, label, image_paths
