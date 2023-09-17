import json

from utility.preprocessing import filter_csv, specified_categories_filter, outfit_filter, train_validation_test_split, \
    eliminate_multiple_outfit_instances, add_not_compatible_items, remove_images

from utility.embeddings import get_embeddings

# Obtain a list containing the wanted category ids

allowed_category_ids = filter_csv('./csv/categories.csv')

# Create a json file with all the items that belongs to the specified wanted categories

with open('.\json\polyvore_item_metadata.json', 'r') as polyvore_item_metadata:
    items_json = json.load(polyvore_item_metadata)

filtered_items_json = specified_categories_filter(items_json, allowed_category_ids)

with open('.\json\\filtered\\filtered_polyvore_item_metadata.json', 'w') as filtered_polyvore_item_metadata_file:
    filtered_polyvore_item_metadata_file.write(filtered_items_json)

# ----------------------------------------------------------------------------------------------------------------------

# Create the json files containing the train, validation and test data for the learning of the autoencoder used to
# obtain the images embeddings

with open('.\json\\filtered\\filtered_polyvore_item_metadata.json', 'r') as filtered_polyvore_item_metadata:
    filtered_items_json = json.load(filtered_polyvore_item_metadata)

ae_train_set, ae_validation_set, ae_test_set = train_validation_test_split(
    [{key: value} for key, value in filtered_items_json.items()], test_ratio=0.2)

with open('.\json\\filtered\\ae_train_set.json', 'w') as ae_train_set_file:
    json.dump(ae_train_set, ae_train_set_file, indent=4)

with open('.\json\\filtered\\ae_validation_set.json', 'w') as ae_validation_set_file:
    json.dump(ae_validation_set, ae_validation_set_file, indent=4)

with open('.\json\\filtered\\ae_test_set.json', 'w') as ae_test_set_file:
    json.dump(ae_test_set, ae_test_set_file, indent=4)

# ----------------------------------------------------------------------------------------------------------------------

# Filter the outfit itemsets in the overall data to eliminate the items that don't belong to the wanted categories

with open('.\json\\train.json', 'r') as d1:
    d1 = json.load(d1)
with open('.\json\\test.json', 'r') as d2:
    d2 = json.load(d2)
with open('.\json\\valid.json', 'r') as d3:
    d3 = json.load(d3)

dataset_json = eliminate_multiple_outfit_instances(d1 + d2 + d3)

with open('.\json\polyvore_outfit_titles.json', 'r') as polyvore_outfit_titles:
    outfit_titles = json.load(polyvore_outfit_titles)

filtered_dataset = outfit_filter(dataset_json, filtered_items_json, outfit_titles)

# ----------------------------------------------------------------------------------------------------------------------

# Modify the filtered dataset in order to put into each outfit itemset an accessory, a bottom and a pair of shoes
# that are not compatible with the top (used to train the compatibility discriminator)

item_ids, item_categories, embeddings = get_embeddings()

filtered_dataset = add_not_compatible_items(dataset_json, item_ids, item_categories, embeddings)

# ----------------------------------------------------------------------------------------------------------------------

# Create the json files containing the train, validation and test data for the learning of the gan architecture

gan_train_set, gan_validation_set, gan_test_set = train_validation_test_split(filtered_dataset, test_ratio=0.2)

with open('.\json\\filtered\\gan_train_set.json', 'w') as gan_train_set_file:
    json.dump(gan_train_set, gan_train_set_file, indent=4)

with open('.\json\\filtered\\gan_validation_set.json', 'w') as gan_validation_set_file:
    json.dump(gan_validation_set, gan_validation_set_file, indent=4)

with open('.\json\\filtered\\gan_test_set.json', 'w') as gan_test_set_file:
    json.dump(gan_test_set, gan_test_set_file, indent=4)

# ----------------------------------------------------------------------------------------------------------------------

# Remove from the images folder all the items that don't belong to the specified categories

# items = [el + '.jpg' for el in list(filtered_items_json.keys())]
# remove_images(folder_path='../images', images_to_keep=items)
