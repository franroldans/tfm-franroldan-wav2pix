import torch.nn as nn
import torch.nn.functional as F
import torch
from utils import Concat_embed

#https://github.com/anantzoid/BEGAN-pytorch/blob/master/models.py

class Decoder(nn.Module):
    def __init__(self, b_size, h, scale_size, num_channels, disc=False):
        super(Decoder, self).__init__()
        self.num_channel = num_channels
        self.b_size = b_size
        self.h = h
        self.disc = disc
        self.t_act = 1
        self.scale_size = scale_size
        self.embed_dim = 62
        self.projected_embed_dim = 128
        self.latent_dim = 100 + self.projected_embed_dim

        self.l0 = nn.Linear(self.latent_dim, 8 * 8 * self.num_channel)
        self.l1 = nn.Conv2d(self.latent_dim, self.num_channel, 3, 1, 1)
        self.l2 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        self.up1 = nn.Upsample(scale_factor=2)
        self.l3 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        self.l4 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        self.up2 = nn.Upsample(scale_factor=2)
        self.l5 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        self.l6 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        self.up3 = nn.Upsample(scale_factor=2)
        self.l7 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        self.l8 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        if self.scale_size == 128:
            self.up4 = nn.Upsample(scale_factor=2)
            self.l10 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
            self.l11 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        self.l9 = nn.Conv2d(self.num_channel, 3, 3, 1, 1)

        self.projection = nn.Sequential(
            nn.Linear(in_features=self.embed_dim, out_features=self.projected_embed_dim),
            nn.BatchNorm1d(num_features=self.projected_embed_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, noise, embed_vector):
        projected_embed = self.projection(embed_vector).unsqueeze(2).unsqueeze(3)
        print("Dimensions of projected embed {}".format(projected_embed.data.shape))
        print("Dimension of noise {}".format(noise.data.shape))
        latent_vector = torch.cat([projected_embed, noise], 1)
        #x = self.l0(latent_vector)
        #x = x.view(self.b_size, self.num_channel, 8, 8)

        x = F.elu(self.l1(latent_vector), True)
        x = F.elu(self.l2(x), True)
        x = self.up1(x)

        x = F.elu(self.l3(x), True)
        x = F.elu(self.l4(x), True)
        x = self.up2(x)
        x = F.elu(self.l5(x), True)
        x = F.elu(self.l6(x), True)
        x = self.up3(x)
        x = F.elu(self.l7(x), True)
        x = F.elu(self.l8(x), True)
        if self.scale_size == 128:
            x = self.up4(x)
            x = F.elu(self.l10(x))
            x = F.elu(self.l11(x))
        x = self.l9(x)
        # if not self.disc:
        # if self.scale_size != 128:# and self.t_act:
        x = F.tanh(x)
        return x


class Encoder(nn.Module):
    def __init__(self, b_size, h, scale_size, num_channels):
        super(Encoder, self).__init__()
        self.num_channel = num_channels
        self.h = h
        self.b_size = b_size
        self.scale_size = scale_size
        self.embed_dim = 62
        self.projected_embed_dim = 128
        self.l0 = nn.Conv2d(3, self.num_channel, 3, 1, 1)
        self.l1 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        self.l2 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        self.down1 = nn.Conv2d(self.num_channel, self.num_channel, 1, 1, 0)
        self.pool1 = nn.AvgPool2d(2, 2)

        self.l3 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        self.l4 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        self.down2 = nn.Conv2d(self.num_channel, 2 * self.num_channel, 1, 1, 0)
        self.pool2 = nn.AvgPool2d(2, 2)

        self.l5 = nn.Conv2d(2 * self.num_channel, 2 * self.num_channel, 3, 1, 1)
        self.l6 = nn.Conv2d(2 * self.num_channel, 2 * self.num_channel, 3, 1, 1)
        self.down3 = nn.Conv2d(2 * self.num_channel, 3 * self.num_channel, 1, 1, 0)
        self.pool3 = nn.AvgPool2d(2, 2)

        if self.scale_size == 64:
            self.l7 = nn.Conv2d(3 * self.num_channel, 3 * self.num_channel, 3, 1, 1)
            self.l8 = nn.Conv2d(3 * self.num_channel, 3 * self.num_channel, 3, 1, 1)
            self.l9 = nn.Linear(8 * 3 * 8+self.projected_embed_dim, 64)
        elif self.scale_size == 128:
            self.l7 = nn.Conv2d(3 * self.num_channel, 3 * self.num_channel, 3, 1, 1)
            self.l8 = nn.Conv2d(3 * self.num_channel, 3 * self.num_channel, 3, 1, 1)
            self.down4 = nn.Conv2d(3 * self.num_channel, 4 * self.num_channel, 1, 1, 0)
            self.pool4 = nn.AvgPool2d(2, 2)

            self.l9 = nn.Conv2d(4 * self.num_channel, 4 * self.num_channel, 3, 1, 1)
            self.l11 = nn.Conv2d(4 * self.num_channel, 4 * self.num_channel, 3, 1, 1)
            self.l12 = nn.Linear(8 * 8 * 4 * self.num_channel, self.h)

        self.projector = nn.Sequential(
                nn.Linear(in_features=self.embed_dim, out_features=self.projected_embed_dim),
                nn.BatchNorm1d(num_features=self.projected_embed_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )


    def forward(self, input, embed_vector):

        x = F.elu(self.l0(input), True)
        x = F.elu(self.l1(x), True)
        x = F.elu(self.l2(x), True)
        x = self.down1(x)
        x = self.pool1(x)

        x = F.elu(self.l3(x), True)
        x = F.elu(self.l4(x), True)
        x = self.pool2(self.down2(x))

        x = F.elu(self.l5(x), True)
        x = F.elu(self.l6(x), True)
        x = self.pool3(self.down3(x))

        if self.scale_size == 64:
            x = F.elu(self.l7(x), True)
            x_intermediate = F.elu(self.l8(x), True)
            projected_embed= self.projector(embed_vector).unsqueeze(2).unsqueeze(3)
            replicated_embed = projected_embed#.repeat(3, 3, 1, 1).permute(2, 3, 0, 1)
            x = torch.cat([x_intermediate, replicated_embed], 1)
            x = x.view(self.b_size, 3 * 8 * 8 + 128)
            x = self.l9(x)
        else:
            x = F.elu(self.l7(x), True)
            x = F.elu(self.l8(x), True)
            x = self.down4(x)
            x = self.pool4(x)
            x = F.elu(self.l9(x), True)
            x = F.elu(self.l11(x), True)
            x = x.view(self.b_size, 8 * 8 * 4 * self.num_channel)
            x = F.elu(self.l12(x), True)

        return x


class Discriminator(nn.Module):
    def __init__(self, b_size, h, scale_size, num_channels):
        super(Discriminator, self).__init__()
        self.enc = Encoder(b_size, h, scale_size, num_channels)
        self.dec = Decoder(b_size, h, scale_size, num_channels, True)

    def forward(self, input, noise, identity):
        return self.dec(self.enc(noise, identity), identity)


def weights_init(self, m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class _Loss(nn.Module):
    def __init__(self, size_average=True):
        super(_Loss, self).__init__()
        self.size_average = size_average

    def forward(self, input, target):
        # this won't still solve the problem
        # which means gradient will not flow through target
        # _assert_no_grad(target)
        backend_fn = getattr(self._backend, type(self).__name__)
        return backend_fn(self.size_average)(input, target)


class L1Loss(_Loss):
    r"""Creates a criterion that measures the mean absolute value of the
    element-wise difference between input `x` and target `y`:
    :math:`{loss}(x, y)  = 1/n \sum |x_i - y_i|`
    `x` and `y` arbitrary shapes with a total of `n` elements each.
    The sum operation still operates over all the elements, and divides by `n`.
    The division by `n` can be avoided if one sets the constructor argument `sizeAverage=False`
    """
    pass
