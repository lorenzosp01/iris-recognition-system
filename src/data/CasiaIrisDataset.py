import random
import cv2
import pandas as pd
import os
from torch.utils.data import Dataset
from src.utils.irisExtractor import get_cropped_iris_image


class CasiaIrisDataset(Dataset):
    def __init__(self, root_dir, transform=[]):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "CASIA-Iris-Thousand")
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.labeldict = {}
        self.filedict = {}
        self.train_mode = False
        self.data = pd.read_csv(os.path.join(root_dir, 'iris_thousands.csv'))
        print(f"Dataset size: {len(self.data)}")

        count = 0
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
        image_paths = self.image_paths[idx]
        image =  cv2.imread(image_paths, cv2.IMREAD_GRAYSCALE)

        # Load label
        label = self.labels[idx]

        # Apply transformations
        if len(self.transform) > 0:
            transform_idx = idx // len(self.data)
            image = self.transform[transform_idx](image)

        return image, label, image_paths
