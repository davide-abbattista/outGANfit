import json
from torch import nn

from utility.pytorch_msssim import SSIM


def gan_weights_init(m):
    # custom weights initialization for the GAN
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def ae_weights_init(m):
    # custom weights initialization for the Autoencoder
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


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


class SSIM_Loss(SSIM):
    # loss evaluation based of SSIM score
    def forward(self, img1, img2):
        return 100 * (1 - super(SSIM_Loss, self).forward(img1, img2))
