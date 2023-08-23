import json
import random


def specified_categories_filter(items):
    filtered_data = {key: value for key, value in items.items() if
                     value.get("semantic_category") == "bottoms" or value.get(
                         "semantic_category") == "tops" or value.get(
                         "semantic_category") == "shoes" or value.get("semantic_category") == "accessories"}
    return json.dumps(filtered_data, indent=4)


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


def outfit_filter(dataset, filtered_items, outfit_titles):
    filtered_outfits = []
    item_ids = list(filtered_items.keys())
    for outfit in dataset:
        items = [item for item in outfit['items'] if item['item_id'] in item_ids]
        categories = [filtered_items[item["item_id"]]["semantic_category"] for item in items]

        # consider the outfit only if it contains items of all the four wanted categories
        if len(set(categories)) == 4:
            outfit['items'] = items
            for item in outfit['items']:
                del item['index']
            outfit['items'] = [item | {"category": filtered_items[item["item_id"]]["semantic_category"]} for item in
                               outfit['items']]
            outfit['outfit_description'] = outfit_titles[outfit["set_id"]]["url_name"]
            del outfit["set_id"]

            # remove multiple items for the same categories in the outfit itemset
            bottom = 0
            shoes = 0
            top = 0
            accessory = 0
            to_remove = []
            for i, item in enumerate(outfit['items']):
                category = filtered_items[item["item_id"]]["semantic_category"]
                if category == 'bottoms':
                    bottom += 1
                    if bottom > 1:
                        to_remove.append(i)
                elif category == 'shoes':
                    shoes += 1
                    if shoes > 1:
                        to_remove.append(i)
                elif category == 'tops':
                    top += 1
                    if top > 1:
                        to_remove.append(i)
                elif category == 'accessories':
                    accessory += 1
                    if accessory > 1:
                        to_remove.append(i)
            for i in sorted(to_remove, reverse=True):
                del outfit['items'][i]

            filtered_outfits.append(outfit)

    return filtered_outfits


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
