import os
import cv2
import torch
from torch.utils.data import Dataset

class MiewIDDataset(Dataset):
    def __init__(self, csv, images_dir, transforms=None):

        self.csv = csv#.reset_index()
        self.augmentations = transforms
        self.images_dir = images_dir

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]
        
        image_path = os.path.join(self.images_dir, row['file_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']

        
        return {"image": image, "label":torch.tensor(row['name']), "image_idx": self.csv.index[index]}