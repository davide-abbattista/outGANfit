import json

from utility.preprocessing import filter_csv, specified_categories_filter, outfit_filter, train_validation_test_split, \
    eliminate_multiple_outfit_instances, remove_images

# Obtain a list containing the wanted category ids

allowed_category_ids = filter_csv('./csv/categories.csv')

# Create a json file with all the items that belongs to the specified wanted categories

with open('.\json\polyvore_item_metadata.json', 'r') as polyvore_item_metadata:
    items_json = json.load(polyvore_item_metadata)

filtered_items_json = specified_categories_filter(items_json, allowed_category_ids)

with open('.\json\\filtered\\filtered_polyvore_item_metadata.json', 'w') as filtered_polyvore_item_metadata_file:
    filtered_polyvore_item_metadata_file.write(filtered_items_json)

# ----------------------------------------------------------------------------------------------------------------------

# Filter the outfit itemsets in the overall data to eliminate the items that don't belong to the wanted categories

with open('.\json\\train.json', 'r') as d1:
    d1 = json.load(d1)
with open('.\json\\test.json', 'r') as d2:
    d2 = json.load(d2)
with open('.\json\\valid.json', 'r') as d3:
    d3 = json.load(d3)

dataset_json = eliminate_multiple_outfit_instances(d1 + d2 + d3)

with open('.\json\\filtered\\filtered_polyvore_item_metadata.json', 'r') as filtered_polyvore_item_metadata:
    filtered_items_json = json.load(filtered_polyvore_item_metadata)

with open('.\json\polyvore_outfit_titles.json', 'r') as polyvore_outfit_titles:
    outfit_titles = json.load(polyvore_outfit_titles)

filtered_dataset = outfit_filter(d1 + d2 + d3, filtered_items_json, outfit_titles)

train_set, validation_set, test_set = train_validation_test_split(filtered_dataset, test_ratio=0.2)

with open('.\json\\filtered\\train_set.json', 'w') as train_set_file:
    json.dump(train_set, train_set_file, indent=4)

with open('.\json\\filtered\\validation_set.json', 'w') as validation_set_file:
    json.dump(validation_set, validation_set_file, indent=4)

with open('.\json\\filtered\\test_set.json', 'w') as test_set_file:
    json.dump(test_set, test_set_file, indent=4)

# ----------------------------------------------------------------------------------------------------------------------

# Remove from the images folder all the items that don't belong to the specified categories

items = [el + '.jpg' for el in list(filtered_items_json.keys())]
remove_images(folder_path='../images', images_to_keep=items)
