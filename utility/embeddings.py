import numpy as np
import torch
from torchvision import transforms

from architecture.ae import AutoEncoder
from utility.custom_image_dataset import CustomImageDatasetAE
from utility.utils import read_json, write_json


def generate_embeddings(items):
    ae_train_set = read_json('../preprocessing/json/filtered/ae_train_set.json')
    ae_validation_set = read_json('../preprocessing/json/filtered/ae_validation_set.json')
    ae_test_set = read_json('../preprocessing/json/filtered/ae_test_set.json')

    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor()
    ])

    dataset = ae_train_set + ae_validation_set + ae_test_set
    totalset = CustomImageDatasetAE(img_dir='../images', data=dataset, transform=transform)
    loader = torch.utils.data.DataLoader(totalset, batch_size=128, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ae = AutoEncoder.to(device)
    checkpoint = torch.load('../checkpoints/trained_ae_128.pth', map_location=torch.device('cpu'))
    ae.load_state_dict(checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings = []
    ae.eval()
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            _, embedds = ae(data)
            embeddings.append(embedds.reshape(embedds.shape[0], -1).cpu().numpy())

    embeddings = np.concatenate(embeddings)
    item_ids = [int(id) for id in list(items.keys())]
    item_categories = list(items.values())

    write_json('../preprocessing/json/embeddings/item_ids.json', item_ids)
    write_json('../preprocessing/json/embeddings/item_categories.json', item_categories)
    with open('../preprocessing/json/embeddings/embeddings.npy', 'wb') as embeddings_file:
        np.save(embeddings_file, embeddings)


def get_embedding(item, item_ids, item_categories, embeddings):
    # return the embedding corresponding to the given item ID
    index = item_ids.index(item)
    return embeddings[index], item_categories[index]


def get_most_similar_item(item, item_ids, item_categories, embeddings):
    # return the ID of the most similar item based on 2-norm distance
    item_embedding, item_category = get_embedding(item, item_ids, item_categories, embeddings)
    distances = torch.cdist(torch.Tensor(item_embedding).unsqueeze(0).unsqueeze(0), torch.Tensor(embeddings))
    ordered_embedding_indexes = torch.argsort(distances).squeeze(0)

    for index in ordered_embedding_indexes[0]:
        if item != item_ids[index]:
            if item_category == item_categories[index]:
                return item_ids[index]


def get_most_different_item(item, items_ids, item_categories, embeddings):
    # return the ID of the most different item based on 2-norm distance
    item_embedding, item_category = get_embedding(item, items_ids, item_categories, embeddings)
    distances = torch.cdist(torch.Tensor(item_embedding).unsqueeze(0).unsqueeze(0), torch.Tensor(embeddings))
    ordered_embedding_indexes = torch.argsort(distances, descending=True).squeeze(0)

    for index in ordered_embedding_indexes[0]:
        if item_category == item_categories[index]:
            return items_ids[index]


def get_embeddings():
    item_ids = read_json('../preprocessing/json/embeddings/item_ids.json')
    item_categories = read_json('../preprocessing/json/embeddings/item_categories.json')

    with open('../preprocessing/json/embeddings/embeddings.npy', 'rb') as embeddings_file:
        embeddings = np.load(embeddings_file)

    return item_ids, item_categories, embeddings
