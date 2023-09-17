import json

import numpy as np
import torch


def generate_embeddings(dataloader, ae, items):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings = []
    ae.eval()
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            _, embedds = ae(data)
            embeddings.append(embedds.reshape(embedds.shape[0], -1).cpu().numpy())

    embeddings = np.concatenate(embeddings)
    item_ids = [int(id) for id in list(items.keys())]
    item_categories = list(items.values())

    with open('.\json\\embeddings\\item_ids.json', 'w') as item_ids_file:
        json.dump(item_ids, item_ids_file, indent=4)

    with open('.\json\\embeddings\\item_categories.json', 'w') as item_categories_file:
        json.dump(item_categories, item_categories_file, indent=4)

    with open('.\json\\embeddings\\embeddings.npy', 'wb') as embeddings_file:
        np.save(embeddings_file, embeddings)


def get_embedding(item, item_ids, item_categories, embeddings):
    index = item_ids.index(item)
    return embeddings[index], item_categories[index]


def get_most_similar_item(item, item_ids, item_categories, embeddings):
    item_embedding, item_category = get_embedding(item, item_ids, item_categories, embeddings)
    distances = torch.cdist(torch.Tensor(item_embedding).unsqueeze(0).unsqueeze(0), torch.Tensor(embeddings))
    ordered_embedding_indexes = torch.argsort(distances).squeeze(0)  # why squeeze(0)
    for index in ordered_embedding_indexes[0]:  # why [0]
        if item != item_ids[index]:
            if item_category == item_categories[index]:
                return item_ids[index]


def get_most_different_item(item, items_ids, item_categories, embeddings):
    item_embedding, item_category = get_embedding(item, items_ids, item_categories, embeddings)
    distances = torch.cdist(torch.Tensor(item_embedding).unsqueeze(0).unsqueeze(0), torch.Tensor(embeddings))
    ordered_embedding_indexes = torch.argsort(distances, descending=True).squeeze(0)  # why squeeze(0)
    for index in ordered_embedding_indexes[0]:  # why [0]
        if item_category == item_categories[index]:
            return items_ids[index]


def get_embeddings():
    with open('..\preprocessing\json\\embeddings\\item_ids.json', 'r') as item_ids_file:
        item_ids = json.load(item_ids_file)
    with open('..\preprocessing\json\\embeddings\\item_categories.json', 'r') as item_categories_file:
        item_categories = json.load(item_categories_file)
    with open('..\preprocessing\json\\embeddings\\embeddings.npy', 'rb') as embeddings_file:
        embeddings = np.load(embeddings_file)
    return item_ids, item_categories, embeddings
