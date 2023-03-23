import segmentation_models_pytorch as smp
from torchvision.models import efficientnet_b6
import torch.nn as nn
import torch

import segmentation_models_pytorch as smp
import torch.nn.functional as F

from segmentation_models_pytorch.base import modules as md
from utils.attention_module import C_Module, S_Module, S_Module_wR

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def convblock(in_channels,out_channels,kernel_size=3,stride=1,dilation=1,padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,dilation=dilation,padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class DecoderBlock_wATT(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        s_module = False,
        c_module = False,
        dropout = False
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            convblock(in_channels=in_channels + skip_channels,out_channels=out_channels,kernel_size=1,padding=0),
            convblock(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1)
        )
        self.conv2 = convblock(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1)
        self.s_module = s_module
        self.c_module = c_module
        self.use_dropout = dropout
        self.dropout = nn.Dropout2d(p=0.1)
        if s_module:
            self.s_module_ = S_Module_wR(in_dim=in_channels + skip_channels)
        if c_module:
            self.c_module_ = C_Module(in_dim=in_channels + skip_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        
        if self.use_dropout:
            x = self.dropout(x)
        if self.s_module and self.c_module:
            sa = self.s_module_(x)
            ca = self.c_module_(x)
            x = ca + sa
        elif self.s_module:
            x = self.s_module_(x)
        elif self.c_module:
            x = self.c_module_(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class DecoderBlock_wATT_no_reverse(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        s_module = False,
        c_module = False,
        dropout = False
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            convblock(in_channels=in_channels + skip_channels,out_channels=out_channels,kernel_size=1,padding=0),
            convblock(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1)
        )
        self.conv2 = convblock(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1)
        self.s_module = s_module
        self.c_module = c_module
        self.use_dropout = dropout
        self.dropout = nn.Dropout2d(p=0.1)
        if s_module:
            self.s_module_ = S_Module(in_dim=in_channels + skip_channels)
        if c_module:
            self.c_module_ = C_Module(in_dim=in_channels + skip_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        
        if self.use_dropout:
            x = self.dropout(x)
        if self.s_module and self.c_module:
            sa = self.s_module_(x)
            ca = self.c_module_(x)
            x = ca + sa
        elif self.s_module:
            x = self.s_module_(x)
        elif self.c_module:
            x = self.c_module_(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

#Multi branch gated decoupled attention with reverse attention  
class MBGDRA(nn.Module):
    def __init__(self,c=2,satt = [True, True, True], catt = [True, False, True] ):
        super(MBGDRA, self).__init__()
        ecd = efficientnet_b6(pretrained=True)
        self.l0 = ecd.features[0]  # 56 256
        self.l1 = ecd.features[1]  # 32 256
        self.l2 = ecd.features[2]  # 40 128
        self.l3 = ecd.features[3]  # 72 64
        self.l4 = ecd.features[4]  # 144 32

        self.up2 = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.conv = nn.Conv2d(in_channels=32,out_channels=1,kernel_size=3,padding=1)
        self.conv_cmb = nn.Conv2d(in_channels=2,out_channels=1,kernel_size=3,padding=1)
        self.bn = nn.BatchNorm2d(num_features=2)
        self.relu = nn.ReLU()
        self.headloc = nn.Sequential(
            convblock(in_channels=32,out_channels=32),
            nn.Conv2d(in_channels=32, out_channels=c, kernel_size=1,padding=0),
        )
        self.headcount = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1,padding=0),
        )

        in_channels = [144, 72, 40]
        skip_channels = [72, 40, 32]
        out_channels = [72, 40, 32]
        
        use_dropout = [False,False,False]

        blocks = [
            DecoderBlock_wATT(in_ch, skip_ch, out_ch,True,st,ct,drop)
            for in_ch, skip_ch, out_ch, st, ct, drop in zip(in_channels, skip_channels, out_channels,satt,catt,use_dropout)
        ]
        self.dec = nn.ModuleList(blocks)

    def forward(self, x):
        l0 = self.l0(x)
        l1 = self.l1(l0)
        l2 = self.l2(l1)
        l3 = self.l3(l2)
        l4 = self.l4(l3)

        dec3l = self.dec[0](l4, l3)
        dec2l = self.dec[1](dec3l, l2)
        dec1l = self.dec[2](dec2l, l1)

        dec0l = self.up2(dec1l)
        headl = self.headloc(dec0l)
        headc = self.headcount(dec0l)

        return headc, headl


#Multi branch gated decoupled attention
class MBGDA(nn.Module):
    def __init__(self,c=2,satt = [True, True, True], catt = [True, False, True] ):
        super(MBGDA, self).__init__()
        ecd = efficientnet_b6(pretrained=True)
        self.l0 = ecd.features[0]  # 56 256
        self.l1 = ecd.features[1]  # 32 256
        self.l2 = ecd.features[2]  # 40 128
        self.l3 = ecd.features[3]  # 72 64
        self.l4 = ecd.features[4]  # 144 32

        self.up2 = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.conv = nn.Conv2d(in_channels=32,out_channels=1,kernel_size=3,padding=1)
        self.conv_cmb = nn.Conv2d(in_channels=2,out_channels=1,kernel_size=3,padding=1)
        self.bn = nn.BatchNorm2d(num_features=2)
        self.relu = nn.ReLU()
        self.headloc = nn.Sequential(
            convblock(in_channels=32,out_channels=32),
            nn.Conv2d(in_channels=32, out_channels=c, kernel_size=1,padding=0),
        )
        self.headcount = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1,padding=0),

        )

        in_channels = [144, 72, 40]
        skip_channels = [72, 40, 32]
        out_channels = [72, 40, 32]
        
        use_dropout = [False,False,False]

        blocks = [
            DecoderBlock_wATT_no_reverse(in_ch, skip_ch, out_ch,True,st,ct,drop)
            for in_ch, skip_ch, out_ch, st, ct, drop in zip(in_channels, skip_channels, out_channels,satt,catt,use_dropout)
        ]
        self.dec = nn.ModuleList(blocks)

    def forward(self, x):
        l0 = self.l0(x)
        l1 = self.l1(l0)
        l2 = self.l2(l1)
        l3 = self.l3(l2)
        l4 = self.l4(l3)

        dec3l = self.dec[0](l4, l3)
        dec2l = self.dec[1](dec3l, l2)
        dec1l = self.dec[2](dec2l, l1)

        dec0l = self.up2(dec1l)
        headl = self.headloc(dec0l)
        headc = self.headcount(dec0l)

        return headc, headl

