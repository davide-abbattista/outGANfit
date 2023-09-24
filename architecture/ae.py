from utility.pytorch_msssim import SSIM  # 'pip install pytorch-msssim' to install fast and differentiable MS-SSIM and SSIM for Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function


class SSIM_Loss(SSIM):
    def forward(self, img1, img2):
        return 100 * (1 - super(SSIM_Loss, self).forward(img1, img2))

# class SSIM_Loss(SSIM):
#     def forward(self, img1, img2):
#         return 1 - super().forward(img1, img2)

# PyTorch's implementation of the GDN non-linearity taken from
# https://github.com/jorge-pessoa/pytorch-gdn/tree/master

class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        ctx.save_for_backward(inputs, inputs.new_ones(1) * bound)
        return inputs.clamp(min=bound)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, bound = ctx.saved_tensors

        pass_through_1 = (inputs >= bound)
        pass_through_2 = (grad_output < 0)

        pass_through = (pass_through_1 | pass_through_2)
        return pass_through.type(grad_output.dtype) * grad_output, None


class GDN(nn.Module):
    """
    Generalized divisive normalization layer.
    """

    def __init__(self, num_features, inverse=False, gamma_init=0.1, beta_bound=1e-6, gamma_bound=0.0,
                 reparam_offset=2 ** -18):
        super(GDN, self).__init__()
        self._inverse = inverse
        self.num_features = num_features
        self.reparam_offset = reparam_offset
        self.pedestal = self.reparam_offset ** 2

        beta_init = torch.sqrt(torch.ones(num_features, dtype=torch.float) + self.pedestal)
        gama_init = torch.sqrt(torch.full((num_features, num_features), fill_value=gamma_init, dtype=torch.float) *
                               torch.eye(num_features, dtype=torch.float) + self.pedestal)

        self.beta = nn.Parameter(beta_init)
        self.gamma = nn.Parameter(gama_init)

        self.beta_bound = (beta_bound + self.pedestal) ** 0.5
        self.gamma_bound = (gamma_bound + self.pedestal) ** 0.5

    def _reparam(self, var, bound):
        var = LowerBound.apply(var, bound)
        return (var ** 2) - self.pedestal

    def forward(self, x):
        gamma = self._reparam(self.gamma, self.gamma_bound).view(self.num_features,
                                                                 self.num_features, 1, 1)  # expand to (C, C, 1, 1)
        beta = self._reparam(self.beta, self.beta_bound)
        norm_pool = F.conv2d(x ** 2, gamma, bias=beta, stride=1, padding=0)
        norm_pool = torch.sqrt(norm_pool)

        if self._inverse:
            norm_pool = x * norm_pool
        else:
            norm_pool = x / norm_pool
        return norm_pool


# AutoEncoder model based on the paper https://arxiv.org/pdf/1611.01704.pdf
# with a few modifications to the architecture

class AutoEncoder(nn.Module):
    def __init__(self, C=512, M=128, in_chan=3, out_chan=3):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(C=C, M=M, in_chan=in_chan)
        self.decoder = Decoder(C=C, M=M, out_chan=out_chan)

    def forward(self, x):
        code = self.encoder(x)
        out = self.decoder(code)
        return out, code


class Encoder(nn.Module):
    def __init__(self, C=512, M=128, in_chan=3):
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(  # input image: 3x128x128
            nn.Conv2d(in_channels=in_chan, out_channels=M, kernel_size=5, stride=2, padding=2, bias=False),  # 128x64x64
            GDN(M),
            nn.Conv2d(in_channels=M, out_channels=M, kernel_size=5, stride=2, padding=2, bias=False),  # 128x32x32
            GDN(M),
            nn.Conv2d(in_channels=M, out_channels=M, kernel_size=5, stride=2, padding=2, bias=False),  # 128x16x16
            GDN(M),
            nn.Conv2d(in_channels=M, out_channels=M, kernel_size=5, stride=2, padding=2, bias=False),  # 128x8x8
            GDN(M),
            nn.Conv2d(in_channels=M, out_channels=2*M, kernel_size=5, stride=2, padding=2, bias=False),  # 256x4x4
            GDN(2*M),
            nn.Conv2d(in_channels=2*M, out_channels=C, kernel_size=5, stride=2, padding=2, bias=False),  # 512X2X2
            nn.Flatten(start_dim=1),
            nn.Linear(512 * 2 * 2, 512)
            # nn.ReLU(),
            # nn.Linear(512, 256)
        )

    def forward(self, x):
        return self.enc(x)


class Decoder(nn.Module):
    def __init__(self, C=512, M=128, out_chan=1):
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            # nn.Linear(256, 512),
            # nn.ReLU(),
            nn.Linear(512, 512 * 2 * 2),
            nn.Unflatten(1, (512, 2, 2)),
            nn.ConvTranspose2d(in_channels=C, out_channels=2*M, kernel_size=5,
                               stride=2, padding=2, output_padding=1, bias=False),
            GDN(2*M, inverse=True),
            nn.ConvTranspose2d(in_channels=2*M, out_channels=M, kernel_size=5,
                               stride=2, padding=2, output_padding=1, bias=False),
            GDN(M, inverse=True),
            nn.ConvTranspose2d(in_channels=M, out_channels=M, kernel_size=5,
                               stride=2, padding=2, output_padding=1, bias=False),
            GDN(M, inverse=True),
            nn.ConvTranspose2d(in_channels=M, out_channels=M, kernel_size=5,
                               stride=2, padding=2, output_padding=1, bias=False),
            GDN(M, inverse=True),
            nn.ConvTranspose2d(in_channels=M, out_channels=M, kernel_size=5,
                               stride=2, padding=2, output_padding=1, bias=False),
            GDN(M, inverse=True),
            nn.ConvTranspose2d(in_channels=M, out_channels=out_chan, kernel_size=5,
                               stride=2, padding=2, output_padding=1, bias=False),
        )

    def forward(self, q):
        return torch.sigmoid(self.dec(q))
        # return self.dec(q)