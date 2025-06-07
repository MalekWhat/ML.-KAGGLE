from torch.utils.data import Dataset
import os
import json
from PIL import Image
import torch

class SunCoordDataset(Dataset):
    def __init__(self, path, train=True, transform=None):
        self.path = os.path.join(path, "train" if train else "test")
        self.transform = transform

        with open(os.path.join(self.path, "format.json")) as fp:
            self.format = json.load(fp)

        self.lenght = len(self.format)
        self.files = tuple(self.format.keys())
        self.targets = tuple(self.format.values())

    def __getitem__(self, item):
        path_file = os.path.join(self.path, self.files[item])
        img = Image.open(path_file).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(self.targets[item], dtype=torch.float32)
    
    def __len__(self):
        return self.lenght