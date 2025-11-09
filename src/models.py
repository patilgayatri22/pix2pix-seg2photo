import torch
from torch import nn


def conv_block(in_c, out_c, norm=True):
    layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=not norm)]
    if norm:
        layers.append(nn.BatchNorm2d(out_c))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def deconv_block(in_c, out_c, dropout=False):
    layers = [
        nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    ]
    if dropout:
        layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)


class UNetGenerator(nn.Module):
    """
    8 down / 8 up U-Net for 256x256, 3->3
    """
    def __init__(self, in_ch=3, out_ch=3):
        super().__init__()
        self.down1 = nn.Sequential(nn.Conv2d(in_ch, 64, 4, 2, 1), nn.LeakyReLU(0.2, True))
        self.down2 = conv_block(64, 128)
        self.down3 = conv_block(128, 256)
        self.down4 = conv_block(256, 512)
        self.down5 = conv_block(512, 512)
        self.down6 = conv_block(512, 512)
        self.down7 = conv_block(512, 512)
        self.down8 = nn.Sequential(nn.Conv2d(512, 512, 4, 2, 1), nn.ReLU(True))

        self.up1 = deconv_block(512, 512, dropout=True)
        self.up2 = deconv_block(1024, 512, dropout=True)
        self.up3 = deconv_block(1024, 512, dropout=True)
        self.up4 = deconv_block(1024, 512)
        self.up5 = deconv_block(1024, 256)
        self.up6 = deconv_block(512, 128)
        self.up7 = deconv_block(256, 64)
        self.up8 = nn.ConvTranspose2d(128, out_ch, 4, 2, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        d1 = self.down1(x); d2 = self.down2(d1); d3 = self.down3(d2); d4 = self.down4(d3)
        d5 = self.down5(d4); d6 = self.down6(d5); d7 = self.down7(d6); d8 = self.down8(d7)
        u1 = self.up1(d8);  u1 = torch.cat([u1, d7], 1)
        u2 = self.up2(u1);  u2 = torch.cat([u2, d6], 1)
        u3 = self.up3(u2);  u3 = torch.cat([u3, d5], 1)
        u4 = self.up4(u3);  u4 = torch.cat([u4, d4], 1)
        u5 = self.up5(u4);  u5 = torch.cat([u5, d3], 1)
        u6 = self.up6(u5);  u6 = torch.cat([u6, d2], 1)
        u7 = self.up7(u6);  u7 = torch.cat([u7, d1], 1)
        u8 = self.up8(u7)
        return self.tanh(u8)


class PatchDiscriminator(nn.Module):
    """
    70x70 PatchGAN discriminator.
    Input is concatenation of cond (3ch) and img (3ch): 6 channels.
    """
    def __init__(self, in_ch=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 4, 2, 1), nn.LeakyReLU(0.2, True),
            conv_block(64, 128),
            conv_block(128, 256),
            nn.Conv2d(256, 512, 4, 1, 1, bias=False), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1, 4, 1, 1)  # logits
        )

    def forward(self, x):
        return self.net(x)

    def features(self, x):
        feats = []
        for layer in list(self.net)[:-1]:  # exclude final 1x1 conv
            x = layer(x)
            feats.append(x)
        return feats
