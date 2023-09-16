import torch
from torch import nn

from utility.embeddings import get_embedding, get_embeddings


class Encoder(nn.Module):
    def __init__(self, depth_in, n_filters, filter_size=4, stride=2, padding=1, batchnorm=True):
        super(Encoder, self).__init__()

        if batchnorm:
            self.model = nn.Sequential(nn.Conv2d(depth_in, n_filters, filter_size, stride, padding, bias=False),
                                       nn.BatchNorm2d(n_filters),
                                       nn.LeakyReLU(0.2, inplace=True))
        else:
            self.model = nn.Sequential(nn.Conv2d(depth_in, n_filters, 4, 2, 1, bias=False),
                                       nn.LeakyReLU(0.2, inplace=True))

    def forward(self, input):
        return self.model(input)


class Decoder(nn.Module):
    def __init__(self, depth_in, n_filters, filter_size=4, stride=2, padding=1, dropout=True):
        super(Decoder, self).__init__()

        if dropout:
            self.model = nn.Sequential(nn.ConvTranspose2d(depth_in, n_filters, filter_size, stride, padding, bias=False),
                                       nn.BatchNorm2d(n_filters),
                                       nn.Dropout(0.5, inplace=True),
                                       nn.ReLU(inplace=True))
        else:
            self.model = nn.Sequential(nn.ConvTranspose2d(depth_in, n_filters, filter_size, stride, padding, bias=False),
                                       nn.BatchNorm2d(n_filters),
                                       nn.ReLU(inplace=True))

    def forward(self, input):
        return self.model(input)


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.item_ids, self.item_categories, self.embeddings = get_embeddings()

        # decoder model
        self.d1 = Decoder(128, 128, 2, 1, 0, dropout=False)
        self.d2 = Decoder(128, 64, dropout=False)
        self.d3 = Decoder(64, 32, dropout=False)
        self.d4 = Decoder(32, 16, dropout=False)

        # output
        self.c = nn.ConvTranspose2d(16, 3, 4, 2, 1, bias=False)
        self.t = nn.Tanh()

    def forward(self, cond):
        cond_embedding = get_embedding(cond, self.item_ids, self.item_categories, self.embeddings)
        cond_embedding = cond_embedding.view(-1, 128, 1, 1)
        return self.t(self.c(self.d4(self.d3(self.d2(self.d1(cond_embedding))))))


class Discriminator(nn.Module):
    def __init__(self, depth_in):
        super(Discriminator, self).__init__()

        self.depth_in = depth_in

        # encoder model
        self.e1 = Encoder(self.depth_in, 16, batchnorm=False)  # 16x64x64 (from 3x128x128 or 6x128x128)
        self.e2 = Encoder(16, 32, batchnorm=False)  # 32x32x32
        self.e3 = Encoder(32, 64, batchnorm=False)  # 64x16x16
        self.e4 = Encoder(64, 128, batchnorm=False)  # 128x32x32

        # output
        self.c = nn.Conv2d(128, 1, 2, 1, 0, bias=False)  # 1x1x1
        self.s = nn.Sigmoid()

    def forward(self, img, cond=None):
        if cond is not None:
            img = torch.cat((img, cond), 1)
        output = self.s(self.c(self.e4(self.e3(self.e2(self.e1(img)))))).view(-1, 1)
        return output
