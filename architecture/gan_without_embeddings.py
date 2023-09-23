import torch
from torch import nn


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
    # def __init__(self):
    #     super(Generator, self).__init__()
    #
    #     # encoder model
    #     self.e1 = Encoder(3, 64, batchnorm=False)
    #     self.e2 = Encoder(64, 128)
    #     self.e3 = Encoder(128, 256)
    #     self.e4 = Encoder(256, 512)
    #     self.e5 = Encoder(512, 512)
    #     self.e6 = Encoder(512, 512)
    #     # bottleneck, no batch norm and relu
    #     self.b = nn.Conv2d(512, 512, 4, 2, 1, bias=False)  # try nn.Conv2d(512, 512, 2, 1, 0, bias=False)
    #     self.r = nn.ReLU(inplace=True)
    #     # decoder model
    #     self.d1 = Decoder(512, 512, dim_filter=2, stride=1, padding=0)
    #     self.d2 = Decoder(512, 512)  # try Decoder(1024, 512)
    #     self.d3 = Decoder(512, 512, dropout=False)
    #     self.d4 = Decoder(512, 256, dropout=False)
    #     self.d5 = Decoder(256, 128, dropout=False)
    #     self.d6 = Decoder(128, 64, dropout=False)
    #     # output
    #     self.g = nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False)
    #     self.t = nn.Tanh()

    def __init__(self):
        super(Generator, self).__init__()

        # encoder model
        self.e1 = Encoder(3, 16, batchnorm=False)
        self.e2 = Encoder(16, 32)
        self.e3 = Encoder(32, 64)
        self.e4 = Encoder(64, 128)
        self.e5 = Encoder(128, 256)
        self.e6 = Encoder(256, 512)
        # bottleneck
        self.b = Encoder(512, 512, 2, 1, 0)
        # decoder model
        self.d1 = Decoder(512, 512, 2, 1, 0, dropout=False)
        self.d2 = Decoder(512, 256, dropout=False)
        self.d3 = Decoder(256, 128, dropout=False)
        self.d4 = Decoder(128, 64, dropout=False)
        self.d5 = Decoder(64, 32, dropout=False)
        self.d6 = Decoder(32, 16, dropout=False)
        # output
        self.g = nn.ConvTranspose2d(16, 3, 4, 2, 1, bias=False)
        self.t = nn.Tanh()

    def forward(self, cond):
        # return self.t(self.g(self.d6(self.d5(self.d4(
        #     self.d3(self.d2(self.d1(self.r(self.b(self.e6(self.e5(self.e4(self.e3(self.e2(self.e1(cond))))))))))))))))
        return self.t(self.g(self.d6(self.d5(self.d4(
            self.d3(self.d2(self.d1(self.b(self.e6(self.e5(self.e4(self.e3(self.e2(self.e1(cond)))))))))))))))


class Discriminator(nn.Module):
    def __init__(self, depth_in):
        super(Discriminator, self).__init__()

        self.depth_in = depth_in

        # self.model = nn.Sequential(nn.Conv2d(depth_in, 16, 4, 2, 1, bias=False),  # 16x64x64 (from 3x128x128 or
        #                            # 6x128x128)
        #                            nn.LeakyReLU(0.2, inplace=True),
        #                            nn.Conv2d(16, 32, 4, 2, 1, bias=False),  # 32x32x32
        #                            nn.BatchNorm2d(32),
        #                            nn.LeakyReLU(0.2, inplace=True),
        #                            nn.Conv2d(32, 64, 4, 2, 1, bias=False),  # 64x16x16
        #                            nn.BatchNorm2d(64),
        #                            nn.LeakyReLU(0.2, inplace=True),
        #                            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # 128x8x8
        #                            nn.BatchNorm2d(128),
        #                            nn.LeakyReLU(0.2, inplace=True),
        #                            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # 256x4x4
        #                            nn.BatchNorm2d(256),
        #                            nn.LeakyReLU(0.2, inplace=True),
        #                            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # 512x2x2
        #                            nn.BatchNorm2d(512),
        #                            nn.LeakyReLU(0.2, inplace=True),
        #                            nn.Conv2d(512, 1, 2, 1, 0, bias=False),  # 1x1x1
        #                            nn.Sigmoid()
        #                            )

        # encoder model
        self.e1 = Encoder(self.depth_in, 16, batchnorm=False)  # 16x64x64 (from 3x128x128 or 6x128x128)
        self.e2 = Encoder(16, 32)  # 32x32x32
        self.e3 = Encoder(32, 64)  # 64x16x16
        self.e4 = Encoder(64, 128)  # 128x32x32
        self.e5 = Encoder(128, 256)  # 256x4x4
        self.e6 = Encoder(256, 512)  # 512x2x2

        # output
        self.c = nn.Conv2d(512, 1, 2, 1, 0, bias=False)  # 1x1x1
        self.s = nn.Sigmoid()

    def forward(self, img, cond=None):
        if cond is not None:
            img = torch.cat((img, cond), 1)
        output = self.s(self.c(self.e6(self.e5(self.e4(self.e3(self.e2(self.e1(img)))))))).view(-1, 1)
        #output = self.model(img).view(-1, 1)
        return output
