import os

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, data, transform=None):
        self.img_dir = img_dir
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item, label = list(self.data[idx].keys())[0], list(self.data[idx].values())[0]
        img_path = os.path.join(self.img_dir, str(item) + '.jpg')
        # add .to(torch.float) to convert the dtype from uint8 to float32 for using transforms.Normalize(...)
        image = read_image(img_path).to(torch.float)
        if self.transform:
            image = self.transform(image)
        return image, label
