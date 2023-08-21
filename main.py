import json

import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import transforms

from utility.custom_image_dataset import CustomImageDataset

with open('.\preprocessing\json\\filtered\\train_set.json', 'r') as train_data:
    train_set = json.load(train_data)

# with open('.\preprocessing\json\\filtered\\validation_set.json', 'r') as validation_data:
#     validation_set = json.load(validation_data)
#
# with open('.\preprocessing\json\\filtered\\test_set.json', 'r') as test_data:
#     test_set = json.load(test_data)

transform = transforms.Compose([transforms.Resize(100)])

trainset = CustomImageDataset(img_dir='.\images', data=train_set, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = next(dataiter)

# plot the images in the first outfit sample of the batch, along with the corresponding labels
images = [el[0].numpy() for el in images]
labels = [el[0] for el in labels]
fig = plt.figure(figsize=(6, 2))
for idx in np.arange(4):
    ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
    img = images[idx]
    plt.imshow(np.transpose(img, (1, 2, 0)))
    ax.set_title(labels[idx])
plt.show()
