import os
from torch.utils.data import Dataset
from PIL import Image


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

        image = Image.open(img_path)
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
        category = [item['category'] for item in self.data[idx]['items'] if item['category'] != 'tops'][0]
        labels = ['tops', str(category), str(category)]
        top = [item['item_id'] for item in self.data[idx]['items'] if item['category'] == 'tops'][0]
        compatible_item = [item['item_id'] for item in self.data[idx]['items'] if item['category'] != 'tops'][0]
        not_compatible_item = [item['item_id'] for item in self.data[idx]['not_compatible']][0]

        img_path_t = os.path.join(self.img_dir, str(top) + '.jpg')
        img_path_c = os.path.join(self.img_dir, str(compatible_item) + '.jpg')
        img_path_nc = os.path.join(self.img_dir, str(not_compatible_item) + '.jpg')

        image_t = Image.open(img_path_t)
        image_c = Image.open(img_path_c)
        image_nc = Image.open(img_path_nc)
        if self.transform:
            image_t = self.transform(image_t)
            image_c = self.transform(image_c)
            image_nc = self.transform(image_nc)

        return [image_t, image_c, image_nc], labels
