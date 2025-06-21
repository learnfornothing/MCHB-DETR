import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward
from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerEncoderLayer

######################################## SE start ########################################
class SE(nn.Module):
    def __init__(self, c1, c2, ratio=4):
        super(SE, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c1, c2 // ratio, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(c2 // ratio, c2, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out
######################################## SE end ########################################


######################################## CoordAtt start ########################################
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out
######################################## CoordAtt end ########################################


######################################## HWD start ########################################
class HWD(nn.Module):
    def __init__(self, in_ch, out_ch, k, s, p):
        super(HWD, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv = Conv(in_ch * 4, out_ch, k, s, p)

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv(x)
        return x
######################################## HWD end ########################################


######################################## MELAN start ########################################
class MBlock(nn.Module):

    def __init__(self, c1, c2, shortcut=True, e=2):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c3, k=1, s=1, act=True)
        self.cv2 = Conv(c3, c3, k=3, s=1, act=True)
        self.cv3 = Conv(c3, c2, k=1, s=1, act=True)
        self.channel = SE(c3, c3)
        self.add = shortcut


    def forward(self, x):
        """Forward pass through the MBlock."""
        y = self.cv2(self.cv1(x)) * self.channel(self.cv2(self.cv1(x)))
        return x + self.cv3(y) if self.add else self.cv3(y)


class MELAN(nn.Module):
    """MELAN with multiple MBlocks."""

    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        c3 = int(c1 * e)
        c4 = c3 * (2 + n)
        self.cv2 = Conv(c4, c2, k=3, s=1, act=True)
        self.m = nn.ModuleList(MBlock(c3, c3, shortcut) for _ in range(n))

    def forward(self, x):
        """Forward pass through the MELAN."""
        y = list(x.chunk(2,1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
######################################## MELAN end ########################################


class CAADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.attention = CoordAtt(c1)
        # self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv1 = HWD(c1 // 2, self.c, 3, 1, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        # x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x = self.attention(x)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)

class TF_concat(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv(128, 128, 1, 1, 0)
        self.conv2 = Conv(128, 128, 1, 1, 0)

    def forward(self, x):
        s, m, l = x[0], x[1], x[2]
        # l = F.adaptive_max_pool2d(l, m.shape[2:]) + F.adaptive_avg_pool2d(l, m.shape[2:])
        # s = F.interpolate(s, m.shape[2:], mode='nearest')
        l = self.conv1(l)
        l = torch.nn.functional.adaptive_avg_pool2d(l, m.shape[2:]) + torch.nn.functional.adaptive_max_pool2d(l, m.shape[2:])
        l = self.conv1(l)
        s = self.conv1(F.interpolate(s, m.shape[2:], mode='nearest'))
        out = torch.cat([l, m, s], dim=1)
        return out

class Add(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sum(torch.stack(x, dim=0), dim=0)

class BScaleseqfeafus(nn.Module):
    def __init__(self, c2):
        super(BScaleseqfeafus, self).__init__()
        self.conv3d = nn.Conv3d(c2, c2,kernel_size=(1,1,1))
        self.bn = nn.BatchNorm3d(c2)
        self.act = nn.LeakyReLU(0.1)
        self.pool_3d = nn.MaxPool3d(kernel_size=(3,1,1))

    def forward(self, x):
        p3, p4, p5 = x[0],x[1],x[2]
        p4_2 = F.interpolate(p4, p3.shape[2:], mode='bilinear')
        p5_2 = F.interpolate(p5, p3.shape[2:], mode='bilinear')
        p3_3d = torch.unsqueeze(p3, -3)
        p4_3d = torch.unsqueeze(p4_2, -3)
        p5_3d = torch.unsqueeze(p5_2, -3)
        combine = torch.cat([p3_3d, p4_3d, p5_3d],dim = 2)
        conv_3d = self.conv3d(combine)
        bn = self.bn(conv_3d)
        act = self.act(bn)
        x = self.pool_3d(act)
        x = torch.squeeze(x, 2)
        return x

class channelattention(nn.Module):
    def __init__(self, c1):
        super(channelattention, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)

        self.m = nn.Sequential(
            nn.Conv2d(c1, int(c1/2), kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(c1/2), c1, kernel_size=1, stride=1, padding=0),
        )
        self.sigmoid=nn.Sigmoid()


    def forward(self, x):
        x1 = self.m(self.avgpool(x))
        x2 = self.m(self.maxpool(x))

        return x * self.sigmoid(x1 + x2)


class SpatialAttention(nn.Module):
    def __init__(self, k):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, k, stride=1, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max, _ = torch.max(x, dim=1, keepdim=True)

        out = torch.cat([avg, max], dim=1)
        return x * self.sigmoid(self.conv1(out))

class GLFF(nn.Module):  # Global and Local Feature Fusion
    def __init__(self, inc):
        super(GLFF, self).__init__()

        self.channel = channelattention(inc)
        self.spatial = SpatialAttention(3)


    def forward(self, x):
        f1, f2 = x[0], x[1]

        f2 = self.channel(f2)
        out = f1 + f2

        return self.spatial(out)



