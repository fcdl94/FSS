import torch
import torch.nn as nn
from torch.distributions import normal


# PixToPiXResnetBlock
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Taken from PixToPix-HD (Global Generator)
class GlobalGenerator(nn.Module):
    # def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
    #              padding_type='reflect'):
    def __init__(self, z_dim, feat_dim, out_dim, ngf=64, n_downsampling=2, n_blocks=3, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(GlobalGenerator, self).__init__()
        self.Z_dist = normal.Normal(0, 1)
        self.z_dim = z_dim
        input_nc = z_dim + feat_dim
        output_nc = out_dim
        activation = nn.ReLU(True)

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        # downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        # resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        # upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                      norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        self.model = nn.Sequential(*model)

    def forward(self, x, z=None, add_z=True):
        if z is None and add_z:
            z = self.Z_dist.sample((x.shape[0], self.z_dim, x.shape[2], x.shape[3]))
            z = z.to(x.device)
        inp = torch.cat((x, z), dim=1) if add_z else x

        return self.model(inp)


# Taken from PixToPix-HD (Global Generator)
class GlobalGenerator2(nn.Module):
    # def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
    #              padding_type='reflect'):
    def __init__(self, z_dim, feat_dim, out_dim, ngf=1024, n_downsampling=2, n_blocks=3, norm_layer=nn.InstanceNorm2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(GlobalGenerator2, self).__init__()
        self.Z_dist = normal.Normal(0, 1)
        self.z_dim = z_dim
        input_nc = z_dim + feat_dim
        output_nc = out_dim
        activation = nn.ReLU(True)

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]

        curr_ngf = ngf
        # downsample
        for i in range(n_downsampling):
            model += [nn.Conv2d(curr_ngf,  curr_ngf // 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(curr_ngf // 2), activation]
            curr_ngf = curr_ngf // 2

        # resnet blocks
        for i in range(n_blocks):
            model += [ResnetBlock(curr_ngf, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        # upsample
        for i in range(n_downsampling):
            model += [nn.UpsamplingBilinear2d(scale_factor=2),
                      nn.Conv2d(curr_ngf, curr_ngf*2, kernel_size=3, stride=1, padding=1),
                      norm_layer(curr_ngf*2), activation]
            curr_ngf = curr_ngf*2
        model += [nn.ReflectionPad2d(3), nn.Conv2d(curr_ngf, output_nc, kernel_size=7, padding=0)]
        self.model = nn.Sequential(*model)

    def forward(self, x, z=None, add_z=True):
        if z is None and add_z:
            z = self.Z_dist.sample((x.shape[0], self.z_dim, x.shape[2], x.shape[3]))
            z = z.to(x.device)
        inp = torch.cat((x, z), dim=1) if add_z else x

        return self.model(inp)


class FeatGenerator(nn.Module):
    def __init__(self, z_dim, attr_dim, out_dim, dim=256, norm_layer=nn.InstanceNorm2d, n_layer=0):
        super(FeatGenerator, self).__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.Z_dist = normal.Normal(0, 1)
        activation = nn.LeakyReLU(0.2, inplace=True)

        model = []
        model += [nn.Conv2d(z_dim+attr_dim, dim, 3, 1, 1, bias=False),  # this is 4,1,1 instead of 4,2,1
                  norm_layer(dim),
                  activation]

        for _ in range(n_layer):
            model += [ResnetBlock(dim, padding_type="zero", activation=activation, norm_layer=norm_layer), activation]

        model += [
            # # state size. (ndf) x 32 x 32
            nn.Conv2d(dim, dim*2, 3, 1, 1, bias=False),
            norm_layer(dim * 2),
            activation,
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(dim * 2, dim * 4, 3, 1, 1, bias=False),
            norm_layer(dim * 4),
            activation,
            # state size. (ndf*4) x 32 x 32
            nn.Conv2d(dim * 4, dim * 8, 3, 1, 1, bias=False),
            norm_layer(dim * 8),
            activation,
            # state size. (ndf*8) x 32 x 32
            nn.Conv2d(dim * 8, out_dim, 1, 1, 0, bias=False)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x, z=None, add_z=True):
        if z is None and add_z:
            z = self.Z_dist.sample((x.shape[0], self.z_dim, x.shape[2], x.shape[3]))
            z = z.to(x.device)
        inp = torch.cat((x, z), dim=1) if add_z else x

        return self.model(inp)


class DCGANDiscriminator(nn.Module):
    def __init__(self, in_feat=2048, dim=256):
        super(DCGANDiscriminator, self).__init__()
        self.main = nn.Sequential(
            # OUR input is already 32x32 so skip first stride.
            # input is (nc) x 64 x 64
            nn.Conv2d(in_feat, dim, 3, 1, 1, bias=False),  # this is 4,1,1 instead of 4,2,1
            nn.LeakyReLU(0.2, inplace=True),
            # # state size. (ndf) x 32 x 32
            nn.Conv2d(dim, dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(dim * 2, dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(dim * 4, dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(dim * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)
