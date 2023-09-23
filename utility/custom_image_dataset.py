import os

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class CustomImageDatasetAE(Dataset):
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


class CustomImageDatasetGAN(Dataset):
    def __init__(self, img_dir, data, transform=None):
        self.img_dir = img_dir
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        category = [item['category'] for item in self.data[idx] if item['category'] != 'tops'][0]
        top = [item['item_id'] for item in self.data[idx]['items'] if item['category'] == 'tops'][0]
        compatible_item = [item['item_id'] for item in self.data[idx]['items'] if item['category'] != 'tops'][0]
        not_compatible_item = [item['item_id'] for item in self.data[idx]['not_compatible']][0]
        img_path_t = os.path.join(self.img_dir, str(top) + '.jpg')
        img_path_c = os.path.join(self.img_dir, str(compatible_item) + '.jpg')
        img_path_nc = os.path.join(self.img_dir, str(not_compatible_item) + '.jpg')
        # add .to(torch.float) to convert the dtype from uint8 to float32 for using transforms.Normalize(...)
        image_t = read_image(img_path_t).to(torch.float)
        image_c = read_image(img_path_c).to(torch.float)
        image_nc = read_image(img_path_nc).to(torch.float)
        labels = ['tops', str(category), str(category)]
        if self.transform:
            image_t = self.transform(image_t)
            image_c = self.transform(image_c)
            image_nc = self.transform(image_nc)
        return [image_t, image_c, image_nc], labels