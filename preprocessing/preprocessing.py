import random
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
import json
import csv

from utility.fid import FID
from utility.utils import read_json, write_json, write
from utility.embeddings import generate_embeddings, get_most_different_item, get_embeddings
from training.ae_train import AutoencoderTrainer


def filter_csv(macrocategories, input_filename):
    allowed_category_ids = {}
    with open(input_filename, 'r', newline='') as file:
        csv_reader = csv.reader(file)
        if macrocategories:
            # select the category IDs related to the specified macro-categories
            for row in csv_reader:
                category = row[2]
                if (category == 'bottoms') or (category == 'tops') or (category == 'accessories') or (
                        category == 'shoes'):
                    if row[0] not in allowed_category_ids.keys():
                        allowed_category_ids[row[0]] = category
        else:
            # select the category IDs related to the specified sub-categories
            for row in csv_reader:
                category = row[2]
                if (
                        category == 'bottoms' and 'pants' in row[1].lower()) or (
                        category == 'bottoms' and 'jeans' in row[1].lower()) or (
                        category == 'tops' and 'shirt' in row[1].lower()) or (
                        category == 'tops' and 'sleeveless' in row[1].lower()) or (
                        category == 'tops' and 'tunic' in row[1].lower()) or (
                        category == 'tops' and 'tank' in row[1].lower()) or (
                        category == 'tops' and 'bra' in row[1].lower()) or (
                        category == 'tops' and 'bodie' in row[1].lower()) or (
                        category == 'tops' and 'polos' in row[1].lower()) or (
                        category == 'accessories' and 'sunglasses' in row[1].lower()) or (
                        category == 'shoes' and 'sneakers' in row[1].lower()
                ):
                    if row[0] not in allowed_category_ids.keys():
                        allowed_category_ids[row[0]] = category
    return allowed_category_ids


def specified_categories_filter(items, allowed_category_ids):
    filtered_data = {key: allowed_category_ids[value.get("category_id")] for key, value in items.items() if
                     value.get("category_id") in list(allowed_category_ids.keys())}
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


def compatible_couples_generator(dataset, filtered_items):
    item_ids = list(filtered_items.keys())

    compatible_tops_accessories = []
    compatible_tops_bottoms = []
    compatible_tops_shoes = []

    for outfit in dataset:
        items = [item for item in outfit['items'] if item['item_id'] in item_ids]
        categories = [filtered_items[item["item_id"]] for item in items]

        # consider the outfit only if it contains a top and at least another item belonging to one of the other three
        # categories ('bottoms', 'shoes', 'accessories')
        if len(set(categories)) >= 2 and 'tops' in categories:
            outfit['items'] = items
            for item in outfit['items']:
                del item['index']

            outfit['items'] = [item | {"category": filtered_items[item["item_id"]]} for item in
                               outfit['items']]
            del outfit["set_id"]

            # create multiple compatible couple instances if the outfit contains more than one item beyond the top one
            top = [item for item in outfit['items'] if item['category'] == 'tops'][0]
            for item in outfit['items']:
                if item['category'] == 'accessories':
                    compatible_tops_accessories.append(
                        {'items': [top, item]})
                if item['category'] == 'bottoms':
                    compatible_tops_bottoms.append({'items': [top, item]})
                if item['category'] == 'shoes':
                    compatible_tops_shoes.append({'items': [top, item]})

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


def add_not_compatible_items(dataset_json, metric=None, item_ids=None, item_categories=None, embeddings=None):
    if metric == 'FID':
        fid = FID(range=255)
    for outfit in dataset_json:
        outfit['not_compatible'] = []
        items = outfit['items']
        if metric is None:
            # select the not compatible items based on embeddings distances
            for item in items:
                category = item['category']
                if category != 'tops':
                    different_item = get_most_different_item(item['item_id'], item_ids, item_categories, embeddings)
                    outfit['not_compatible'].append({'item_id': different_item, 'category': category})
        elif metric == 'random':
            # select the not compatible items randomly
            for item in items:
                category = item['category']
                if category != 'tops':
                    different_item_idx = np.random.randint(0, len(dataset_json))
                    while dataset_json[different_item_idx] is outfit:
                        different_item_idx = np.random.randint(0, len(dataset_json))
                    different_item = \
                        [item for item in dataset_json[different_item_idx]['items'] if
                         item['category'] == category][0]
                    outfit['not_compatible'].append(different_item)
        elif metric == 'FID':
            # select the not compatible items based on FID score
            for item in items:
                category = item['category']
                if category != 'tops':
                    different_items = []
                    while len(different_items) < 10:
                        idx = np.random.randint(0, len(dataset_json))
                        if dataset_json[idx] is not outfit:
                            for item in dataset_json[idx]['items']:
                                if item['category'] == category:
                                    different_items.append(item)
                    item_id = item['item_id']
                    transform = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                                    ])
                    item_img = transform(Image.open(f'../images/{item_id}.jpg'))
                    other_images = [transform(Image.open(f'../images/{item["item_id"]}.jpg')) for item in
                                    different_items]
                    fid_scores = []
                    for other_img in other_images:
                        fid_scores.append(fid.calculate_fid(item_img, other_img))
                    different_item = different_items[np.argmax(fid_scores)]
                    outfit['not_compatible'].append(different_item)
    return dataset_json


class Preprocesser:
    def __init__(self, macrocategories=True, autoencoder=True, not_compatible_items_metric=None, ae_trained=False):
        self.autoencoder = autoencoder
        self.macrocategories = macrocategories
        self.not_compatible_items_metric = not_compatible_items_metric
        self.ae_trainded = ae_trained

    def preprocess(self):
        # Obtain a list containing the wanted category IDs
        allowed_category_ids = filter_csv(self.macrocategories, './csv/categories.csv')

        # Create a json file with all the items that belongs to the specified wanted categories
        items_json = read_json('./json/polyvore_item_metadata.json')
        filtered_items_json = specified_categories_filter(items_json, allowed_category_ids)
        write('./json/filtered/filtered_polyvore_item_metadata.json', filtered_items_json)
        filtered_items_json = read_json('./json/filtered/filtered_polyvore_item_metadata.json')

        if self.autoencoder:
            # Create the json files containing the train, validation and test data for the training of the autoencoder
            # used to obtain the images embeddings
            ae_train_set, ae_validation_set, ae_test_set = train_validation_test_split(
                [{key: value} for key, value in filtered_items_json.items()], test_ratio=0.2)
            write_json('./json/filtered/ae_train_set.json', ae_train_set)
            write_json('./json/filtered/ae_validation_set.json', ae_validation_set)
            write_json('./json/filtered/ae_test_set.json', ae_test_set)

            if not self.ae_trainded:
                # Train the autoencoder
                ae_trainer = AutoencoderTrainer(train_set_path='../preprocessing/json/filtered/ae_train_set.json',
                                                validation_set_path='../preprocessing/json/filtered/ae_validation_set'
                                                                    '.json',
                                                test_set_path='../preprocessing/json/filtered/ae_test_set.json')
                ae_trainer.train_and_test()

            # Generate the embeddings
            generate_embeddings(filtered_items_json)

        # Eliminate the multiple outfit instances in the overall data
        d1 = read_json('./json/train.json')
        d2 = read_json('./json/test.json')
        d3 = read_json('./json/valid.json')
        dataset_json = eliminate_multiple_outfit_instances(d1 + d2 + d3)

        # Create the compatibility couples (top-accessory, top-bottom, top-shoes) for the training of the three
        # GANs of the architecture
        compatible_tops_accessories, compatible_tops_bottoms, compatible_tops_shoes = \
            compatible_couples_generator(dataset_json, filtered_items_json)

        # Put into each sample an accessory, a bottom or a pair of shoes that is not compatible with the top
        # (used to train the compatibility discriminator)
        if self.autoencoder:
            item_ids, item_categories, embeddings = get_embeddings()
            compatible_tops_accessories = add_not_compatible_items(compatible_tops_accessories, item_ids,
                                                                   item_categories,
                                                                   embeddings)
            compatible_tops_bottoms = add_not_compatible_items(compatible_tops_bottoms, item_ids, item_categories,
                                                               embeddings)
            compatible_tops_shoes = add_not_compatible_items(compatible_tops_shoes, item_ids, item_categories,
                                                             embeddings)
        else:
            compatible_tops_accessories = add_not_compatible_items(compatible_tops_accessories,
                                                                   metric=self.not_compatible_items_metric)
            compatible_tops_bottoms = add_not_compatible_items(compatible_tops_bottoms,
                                                               metric=self.not_compatible_items_metric)
            compatible_tops_shoes = add_not_compatible_items(compatible_tops_shoes,
                                                             metric=self.not_compatible_items_metric)

        # Create the json files containing the train, validation and test data for the training of the GANs
        gan_train_set_ta, gan_validation_set_ta, gan_test_set_ta = train_validation_test_split(
            compatible_tops_accessories, test_ratio=0.2)
        gan_train_set_tb, gan_validation_set_tb, gan_test_set_tb = train_validation_test_split(
            compatible_tops_bottoms, test_ratio=0.2)
        gan_train_set_ts, gan_validation_set_ts, gan_test_set_ts = train_validation_test_split(
            compatible_tops_shoes, test_ratio=0.2)
        write_json('./json/filtered/gan_train_set_ta.json', gan_train_set_ta)
        write_json('./json/filtered/gan_validation_set_ta.json', gan_validation_set_ta)
        write_json('./json/filtered/gan_test_set_ta.json', gan_test_set_ta)
        write_json('./json/filtered/gan_train_set_tb.json', gan_train_set_tb)
        write_json('./json/filtered/gan_validation_set_tb.json', gan_validation_set_tb)
        write_json('./json/filtered/gan_test_set_tb.json', gan_test_set_tb)
        write_json('./json/filtered/gan_train_set_ts.json', gan_train_set_ts)
        write_json('./json/filtered/gan_validation_set_ts.json', gan_validation_set_ts)
        write_json('./json/filtered/gan_test_set_ts.json', gan_test_set_ts)
