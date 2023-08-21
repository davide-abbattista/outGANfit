import os
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
        accessory = [item['item_id'] for item in self.data[idx]['items'] if item['category'] == 'accessories'][0]
        bottom = [item['item_id'] for item in self.data[idx]['items'] if item['category'] == 'bottoms'][0]
        shoes = [item['item_id'] for item in self.data[idx]['items'] if item['category'] == 'shoes'][0]
        top = [item['item_id'] for item in self.data[idx]['items'] if item['category'] == 'tops'][0]
        img_path_a = os.path.join(self.img_dir, str(accessory) + '.jpg')
        img_path_b = os.path.join(self.img_dir, str(bottom) + '.jpg')
        img_path_s = os.path.join(self.img_dir, str(shoes) + '.jpg')
        img_path_t = os.path.join(self.img_dir, str(top) + '.jpg')
        image_a = read_image(img_path_a)
        image_b = read_image(img_path_b)
        image_s = read_image(img_path_s)
        image_t = read_image(img_path_t)
        labels = ['accessories', 'bottoms', 'shoes', 'tops']
        if self.transform:
            image_a = self.transform(image_a)
            image_b = self.transform(image_b)
            image_s = self.transform(image_s)
            image_t = self.transform(image_t)
        return [image_a, image_b, image_s, image_t], labels
