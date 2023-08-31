import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, depth_in, n_filters, batchnorm=True):
        super(Encoder, self).__init__()

        if batchnorm:
            self.model = nn.Sequential(nn.Conv2d(depth_in, n_filters, 4, 2, 1, bias=False),
                                       nn.BatchNorm2d(n_filters),
                                       nn.LeakyReLU(0.2, inplace=True))
        else:
            self.model = nn.Sequential(nn.Conv2d(depth_in, n_filters, 4, 2, 1, bias=False),
                                       nn.LeakyReLU(0.2, inplace=True))

    def forward(self, input):
        return self.model(input)


class ConcatenateModule(nn.Module):
    def __init__(self):
        super(ConcatenateModule, self).__init__()

    def forward(self, x, skip_in):
        concatenated = torch.cat((x, skip_in), dim=1)
        return concatenated


class Decoder(nn.Module):
    def __init__(self, depth_in, n_filters, dim_filter=4, stride=2, padding=1, dropout=True):
        super(Decoder, self).__init__()

        self.dropout = dropout
        self.stride = stride
        self.padding = padding
        self.dim_filter = dim_filter

        self.conv = nn.ConvTranspose2d(depth_in, n_filters, self.dim_filter, self.stride, self.padding, bias=False)
        self.b = nn.BatchNorm2d(n_filters)
        self.d = nn.Dropout(0.5, inplace=True)
        self.conc = ConcatenateModule()
        self.r = nn.ReLU(inplace=True)

    def forward(self, input, skip_in):
        if self.dropout:
            return self.r(self.conc(self.d(self.b(self.conv(input))), skip_in))
        else:
            return self.r(self.conc(self.b(self.conv(input)), skip_in))


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # encoder model
        self.e1 = Encoder(3, 64, batchnorm=False)
        self.e2 = Encoder(64, 128)
        self.e3 = Encoder(128, 256)
        self.e4 = Encoder(256, 512)
        self.e5 = Encoder(512, 512)
        self.e6 = Encoder(512, 512)
        # bottleneck, no batch norm and relu
        self.b = nn.Conv2d(512, 512, 4, 2, 1, bias=False)
        self.r = nn.ReLU(inplace=True)
        # decoder model
        self.d1 = Decoder(512, 512, dim_filter=2, stride=1, padding=0)
        self.d2 = Decoder(1024, 512)
        self.d3 = Decoder(1024, 512, dropout=False)
        self.d4 = Decoder(1024, 256, dropout=False)
        self.d5 = Decoder(512, 128, dropout=False)
        self.d6 = Decoder(256, 64, dropout=False)
        # output
        self.g = nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False)
        self.t = nn.Tanh()

    def forward(self, cond):
        e1 = self.e1(cond)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        out = self.b(e6)
        out = self.r(out)
        out = self.d1(out, e6)
        out = self.d2(out, e5)
        out = self.d3(out, e4)
        out = self.d4(out, e3)
        out = self.d5(out, e2)
        out = self.d6(out, e1)
        out = self.g(out)
        out = self.t(out)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(nn.Conv2d(6, 16, 4, 2, 1, bias=False),  # 16x64x64 (from 6x128x128)
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(16, 32, 4, 2, 1, bias=False),  # 32x32x32
                                   nn.BatchNorm2d(32),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(32, 64, 4, 2, 1, bias=False),  # 64x16x16
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(64 * 1, 64 * 2, 4, 2, 1, bias=False),  # 128x8x8
                                   nn.BatchNorm2d(64 * 2),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),  # 256x4x4
                                   nn.BatchNorm2d(64 * 4),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),  # 512x2x2
                                   nn.BatchNorm2d(64 * 8),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(64 * 8, 1, 2, 1, 0, bias=False),  # 1x1x1
                                   nn.Sigmoid()
                                   )

    def forward(self, img, cond):
        input = torch.cat((img, cond), 1)
        output = self.model(input).view(-1, 1)
        return output
