import pandas as pd
import os
import sys
import cv2
from torch.utils.data import Dataset

class CasiaIrisDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

        print("Loading dataset...")
        self.data = pd.read_csv(os.path.join(root_dir, 'iris_thousands.csv'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.loadItem(idx)

    def loadItem(self, idx):
        im_idx = idx % len(self.data)

        label = self.data.iloc[im_idx]['Label']
        label, eye = label.split("-")
        img_path = self.data.iloc[im_idx]['ImagePath']

        img_path = img_path.split("/kaggle/input/casia-iris-thousand/")[1]

        if sys.platform == "win32":
            img_path = img_path.replace("/", "\\")

        relative_path = os.path.join(self.root_dir, img_path)

        try:
            img = cv2.imread(relative_path)
            return label, eye, img
        except FileNotFoundError:
            raise Exception(f"Image {img_path} not found in dataset")