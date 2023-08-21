import json

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import transforms

from utils import CustomImageDataset

# Create a json file with all the items that belongs to the specified wanted categories

# with open('.\json\polyvore_item_metadata.json', 'r') as polyvore_item_metadata:
#     data = json.load(polyvore_item_metadata)
#
# filtered_data = {key: value for key, value in data.items() if
#                  value.get("semantic_category") == "bottoms" or value.get("semantic_category") == "tops" or value.get(
#                      "semantic_category") == "shoes" or value.get("semantic_category") == "accessories"}
# filtered_json = json.dumps(filtered_data, indent=4)
#
# with open('.\json\\filtered\\filtered_polyvore_item_metadata.json', 'w') as filtered_polyvore_item_metadata_file:
#     filtered_polyvore_item_metadata_file.write(filtered_json)

with open('.\json\\filtered\\filtered_polyvore_item_metadata.json', 'r') as filtered_polyvore_item_metadata:
    data = json.load(filtered_polyvore_item_metadata)

# ----------------------------------------------------------------------------------------------------------------------

# Filter the outfit itemsets in the training data to eliminate the items that don't belong to the wanted categories

# item_ids = list(data.keys())

# with open('.\json\train.json', 'r') as train:
#     train_data = json.load(train)
# with open('.\json\polyvore_outfit_titles.json', 'r') as polyvore_outfit_titles:
#     outfit_titles = json.load(polyvore_outfit_titles)

# filtered_outfits = []
# categories = []
# for outfit in train_data:
#     filtered_items = [item for item in outfit['items'] if item['item_id'] in item_ids]
#     categories = [data[item["item_id"]]["semantic_category"] for item in filtered_items]
#
#     # consider the outfit only if it contains items of all the four wanted categories
#     if len(set(categories)) == 4:
#         outfit['items'] = filtered_items
#         for item in outfit['items']:
#             del item['index']
#         outfit['items'] = [item | {"category": data[item["item_id"]]["semantic_category"]} for item in outfit['items']]
#         outfit['outfit_description'] = outfit_titles[outfit["set_id"]]["url_name"]
#         del outfit["set_id"]
#
#         # remove multiple items for the same categories in the outfit itemset
#         bottom = 0
#         shoes = 0
#         top = 0
#         accessory = 0
#         to_remove = []
#         for i, item in enumerate(outfit['items']):
#             category = data[item["item_id"]]["semantic_category"]
#             if category == 'bottoms':
#                 bottom += 1
#                 if bottom > 1:
#                     to_remove.append(i)
#             elif category == 'shoes':
#                 shoes += 1
#                 if shoes > 1:
#                     to_remove.append(i)
#             elif category == 'tops':
#                 top += 1
#                 if top > 1:
#                     to_remove.append(i)
#             elif category == 'accessories':
#                 accessory += 1
#                 if accessory > 1:
#                     to_remove.append(i)
#         for i in sorted(to_remove, reverse=True):
#             del outfit['items'][i]
#
#         filtered_outfits.append(outfit)
#
# with open('.\json\\filtered\\filtered_train.json', 'w') as filtered_train_file:
#     json.dump(filtered_outfits, filtered_train_file, indent=4)

with open('.\json\\filtered\\filtered_train.json', 'r') as filtered_train:
    train_data = json.load(filtered_train)

# ----------------------------------------------------------------------------------------------------------------------

# with open('.\json\test.json', 'r') as test:
#     test_data = json.load(test)
#
# filtered_outfits = []
# categories = []
# for outfit in test_data:
#     filtered_items = [item for item in outfit['items'] if item['item_id'] in item_ids]
#     categories = [data[item["item_id"]]["semantic_category"] for item in filtered_items]
#
#     # consider the outfit only if it contains items of all the four wanted categories
#     if len(set(categories)) == 4:
#         outfit['items'] = filtered_items
#         for item in outfit['items']:
#             del item['index']
#         outfit['items'] = [item | {"category": data[item["item_id"]]["semantic_category"]} for item in outfit['items']]
#         outfit['outfit_description'] = outfit_titles[outfit["set_id"]]["url_name"]
#         del outfit["set_id"]
#
#         # remove multiple items for the same categories in the outfit itemset
#         bottom = 0
#         shoes = 0
#         top = 0
#         accessory = 0
#         to_remove = []
#         for i, item in enumerate(outfit['items']):
#             category = data[item["item_id"]]["semantic_category"]
#             if category == 'bottoms':
#                 bottom += 1
#                 if bottom > 1:
#                     to_remove.append(i)
#             elif category == 'shoes':
#                 shoes += 1
#                 if shoes > 1:
#                     to_remove.append(i)
#             elif category == 'tops':
#                 top += 1
#                 if top > 1:
#                     to_remove.append(i)
#             elif category == 'accessories':
#                 accessory += 1
#                 if accessory > 1:
#                     to_remove.append(i)
#         for i in sorted(to_remove, reverse=True):
#             del outfit['items'][i]
#
#         filtered_outfits.append(outfit)
#
# with open('.\json\\filtered\\filtered_test.json', 'w') as filtered_test_file:
#     json.dump(filtered_outfits, filtered_test_file, indent=4)

with open('.\json\\filtered\\filtered_test.json', 'r') as filtered_test:
    test_data = json.load(filtered_test)

# ----------------------------------------------------------------------------------------------------------------------

# with open('.\json\valid.json', 'r') as valid:
#     valid_data = json.load(valid)
#
# filtered_outfits = []
# categories = []
# for outfit in valid_data:
#     filtered_items = [item for item in outfit['items'] if item['item_id'] in item_ids]
#     categories = [data[item["item_id"]]["semantic_category"] for item in filtered_items]
#
#     # consider the outfit only if it contains items of all the four wanted categories
#     if len(set(categories)) == 4:
#         outfit['items'] = filtered_items
#         for item in outfit['items']:
#             del item['index']
#         outfit['items'] = [item | {"category": data[item["item_id"]]["semantic_category"]} for item in outfit['items']]
#         outfit['outfit_description'] = outfit_titles[outfit["set_id"]]["url_name"]
#         del outfit["set_id"]
#
#         # remove multiple items for the same categories in the outfit itemset
#         bottom = 0
#         shoes = 0
#         top = 0
#         accessory = 0
#         to_remove = []
#         for i, item in enumerate(outfit['items']):
#             category = data[item["item_id"]]["semantic_category"]
#             if category == 'bottoms':
#                 bottom += 1
#                 if bottom > 1:
#                     to_remove.append(i)
#             elif category == 'shoes':
#                 shoes += 1
#                 if shoes > 1:
#                     to_remove.append(i)
#             elif category == 'tops':
#                 top += 1
#                 if top > 1:
#                     to_remove.append(i)
#             elif category == 'accessories':
#                 accessory += 1
#                 if accessory > 1:
#                     to_remove.append(i)
#         for i in sorted(to_remove, reverse=True):
#             del outfit['items'][i]
#
#         filtered_outfits.append(outfit)
#
# with open('.\json\\filtered\\filtered_valid.json', 'w') as filtered_valid_file:
#     json.dump(filtered_outfits, filtered_valid_file, indent=4)

with open('.\json\\filtered\\filtered_valid.json', 'r') as filtered_valid:
    valid_data = json.load(filtered_valid)

# ----------------------------------------------------------------------------------------------------------------------

transform = transforms.Compose([transforms.Resize(100)])

trainset = CustomImageDataset(img_dir='../images', data=train_data, transform=transform)
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
