import json
from torch import nn


# custom weights initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def read_json(path):
    with open(path, 'r') as file:
        json_file = json.load(file)
    return json_file


def write(path, json):
    with open(path, 'w') as file:
        file.write(json)


def write_json(path, json_file):
    with open(path, 'w') as file:
        json.dump(json_file, file, indent=4)