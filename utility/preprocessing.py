import json
import random
import os
import csv

import numpy as np

from utility.embeddings import get_most_different_item

def filter_csv(input_filename):
    allowed_category_ids = []
    with open(input_filename, 'r', newline='') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            category = row[2]
            # if (category == 'bottoms' and 'pant' in row[1].lower()) or (
            #         category == 'bottoms' and 'pants' in row[1].lower()) or (
            #         category == 'bottoms' and 'jeans' in row[1].lower()) or (
            #         category == 'tops' and 'shirt' in row[1].lower()) or (
            #         category == 'tops' and 'sleeveless' in row[1].lower()) or (
            #         category == 'tops' and 'tunic' in row[1].lower()) or (
            #         category == 'tops' and 'tank' in row[1].lower()) or (
            #         category == 'tops' and 'bra' in row[1].lower()) or (
            #         category == 'tops' and 'bodie' in row[1].lower()) or (
            #         category == 'tops' and 'polos' in row[1].lower()) or (category == 'shoes') or (
            #         category == 'accessories'):
            if (category == 'bottoms') or (category == 'tops') or (category == 'accessories') or (category == 'shoes'):
                allowed_category_ids.append(row[0])
    return list(set(allowed_category_ids))


def specified_categories_filter(items, allowed_category_ids):
    filtered_data = {key: value.get("semantic_category") for key, value in items.items() if
                     value.get("category_id") in allowed_category_ids}
    return json.dumps(filtered_data, indent=4)


# def specified_categories_filter(items, allowed_category_ids):
#     accessories_ids = [53, 56, 58, 59, 68, 69, 231, 251, 269, 270, 299, 300, 301, 302, 306, 4460, 4463, 4467, 4468, 4470,
#                     4472, 4473, 51, 52, 57, 61]
#     dict = {str(num): 0 for num in accessories_ids}
#     filtered_data = {key: value for key, value in items.items() if
#                      value.get("category_id") in allowed_category_ids}
#     for key, value in items.items():
#         if value.get("category_id") in dict.keys():
#             dict[value.get("category_id")] += 1
#     return json.dumps(filtered_data, indent=4), dict


def eliminate_multiple_outfit_instances(dataset):
    processed_outfit_ids = []
    to_remove = []
    for i, outfit in enumerate(dataset):
        if outfit['set_id'] not in processed_outfit_ids:
            processed_outfit_ids.append(outfit['set_id'])
        else:
            to_remove.append(i)
    if to_remove:
        print(f"There are {len(to_remove)} outfit instances in the dataset to remove")
        for i in sorted(to_remove, reverse=True):
            del dataset[i]
    else:
        print('There are no multiple outfit instances in the dataset')
    return dataset


# def outfit_filter(dataset, filtered_items, outfit_titles):
#     filtered_outfits = []
#     item_ids = list(filtered_items.keys())
#     for outfit in dataset:
#         items = [item for item in outfit['items'] if item['item_id'] in item_ids]
#         categories = [filtered_items[item["item_id"]] for item in items]
#
#         # consider the outfit only if it contains items of all the four wanted categories
#         if len(set(categories)) == 4:
#             outfit['items'] = items
#             for item in outfit['items']:
#                 del item['index']
#
#             outfit['items'] = [item | {"category": filtered_items[item["item_id"]]} for item in
#                                outfit['items']]
#             outfit['outfit_description'] = outfit_titles[outfit["set_id"]]["url_name"]
#             del outfit["set_id"]
#
#             # # remove multiple items for the same categories in the outfit itemset
#             # bottom = 0
#             # shoes = 0
#             # top = 0
#             # accessory = 0
#             # to_remove = []
#             # for i, item in enumerate(outfit['items']):
#             #     category = filtered_items[item["item_id"]]
#             #     if category == 'bottoms':
#             #         bottom += 1
#             #         if bottom > 1:
#             #             to_remove.append(i)
#             #     elif category == 'shoes':
#             #         shoes += 1
#             #         if shoes > 1:
#             #             to_remove.append(i)
#             #     elif category == 'tops':
#             #         top += 1
#             #         if top > 1:
#             #             to_remove.append(i)
#             #     elif category == 'accessories':
#             #         accessory += 1
#             #         if accessory > 1:
#             #             to_remove.append(i)
#             # for i in sorted(to_remove, reverse=True):
#             #     del outfit['items'][i]
#
#             # Create different outfits if there are multiple items for the same categories in the outfit itemset
#             if len(categories) > 4:
#                 outfit_dict = {'accessories': [], 'bottoms': [], 'shoes': [], 'tops': []}
#                 for item in outfit['items']:
#                     category = item['category']
#                     if category == 'bottoms':
#                         outfit_dict['bottoms'].append(item)
#                     elif category == 'shoes':
#                         outfit_dict['shoes'].append(item)
#                     elif category == 'tops':
#                         outfit_dict['tops'].append(item)
#                     elif category == 'accessories':
#                         outfit_dict['accessories'].append(item)
#
#                 values = list(outfit_dict.values())
#
#                 comb = np.array(np.meshgrid(*values), dtype=object).T
#                 comb = comb.reshape(round(comb.size / len(values)), len(values))
#                 for r in comb:
#                     new_outfit = {'items': list(r),
#                                   'outfit_description': outfit['outfit_description']}
#                     filtered_outfits.append(new_outfit)
#             else:
#                 filtered_outfits.append(outfit)
#
#     return filtered_outfits


def outfit_filter(dataset, filtered_items, outfit_titles):
    item_ids = list(filtered_items.keys())

    compatible_tops_accessories = []
    compatible_tops_bottoms = []
    compatible_tops_shoes = []

    for outfit in dataset:
        items = [item for item in outfit['items'] if item['item_id'] in item_ids]
        categories = [filtered_items[item["item_id"]] for item in items]

        # consider the outfit only if it contains items of all the four wanted categories
        if len(set(categories)) >= 2 and 'tops' in categories:
            outfit['items'] = items
            for item in outfit['items']:
                del item['index']

            outfit['items'] = [item | {"category": filtered_items[item["item_id"]]} for item in
                               outfit['items']]
            outfit['outfit_description'] = outfit_titles[outfit["set_id"]]["url_name"]
            del outfit["set_id"]

            outfit_description = outfit['outfit_description']
            top = [item for item in outfit['items'] if item['category'] == 'tops'][0]
            for item in outfit['items']:
                if item['category'] == 'accessories':
                    compatible_tops_accessories.append({'items': [top | item], 'outfit_description': outfit_description})
                if item['category'] == 'bottoms':
                    compatible_tops_bottoms.append({'items': [top | item], 'outfit_description': outfit_description})
                if item['category'] == 'shoes':
                    compatible_tops_shoes.append({'items': [top | item], 'outfit_description': outfit_description})

    return compatible_tops_accessories, compatible_tops_bottoms, compatible_tops_shoes


def train_validation_test_split(dataset, test_ratio, shuffle=False, seed=None):
    if seed:
        shuffle = True
        random.seed(seed)
    if shuffle:
        random.shuffle(dataset)
    val_ratio = test_ratio / (1 - test_ratio)
    test_idx = round(len(dataset) * (1 - test_ratio))
    val_idx = round(test_idx * (1 - val_ratio))
    return dataset[: val_idx], dataset[val_idx: test_idx], dataset[test_idx:]


def add_not_compatible_items(dataset_json, item_ids, item_categories, embeddings):
    for outfit in dataset_json:
        outfit['not_compatible'] = []
        items = outfit['items']
        for item in items:
            category = item['category']
            if category != 'tops':
                different_item = get_most_different_item(item['item_id'], item_ids, item_categories, embeddings)
                outfit['not_compatible'].append({'item_id': different_item, 'category': category})


def remove_images(folder_path, images_to_keep):
    for filename in os.listdir(folder_path):
        if filename not in images_to_keep:
            image_path = os.path.join(folder_path, filename)
            os.remove(image_path)
            print(f"Removed: {filename}")
