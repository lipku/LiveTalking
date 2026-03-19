import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
_rife_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _rife_root not in sys.path:
    sys.path.insert(0, _rife_root)
from model.warplayer import warp
# from train_log.refine import *

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
          else "cpu")
)

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),        
        nn.LeakyReLU(0.2, True)
    )

def conv_bn(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, True)
    )
    
class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x, feat=False):
        x0 = self.cnn0(x)
        x = self.relu(x0)
        x1 = self.cnn1(x)
        x = self.relu(x1)
        x2 = self.cnn2(x)
        x = self.relu(x2)
        x3 = self.cnn3(x)
        if feat:
            return [x0, x1, x2, x3]
        return x3

class ResConv(nn.Module):
    def __init__(self, c, dilation=1):
        super(ResConv, self).__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1\
)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)

class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
            )
        self.convblock = nn.Sequential(
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
        )
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(c, 4*13, 4, 2, 1),
            nn.PixelShuffle(2)
        )

    def forward(self, x, flow=None, scale=1):
        x = F.interpolate(x, scale_factor= 1. / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = F.interpolate(flow, scale_factor= 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        feat = tmp[:, 5:]
        return flow, mask, feat
        
class IFNet(nn.Module):
    def __init__(self):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(7+8, c=192)
        self.block1 = IFBlock(8+4+8+8, c=128)
        self.block2 = IFBlock(8+4+8+8, c=96)
        self.block3 = IFBlock(8+4+8+8, c=64)
        self.block4 = IFBlock(8+4+8+8, c=24)
        self.encode = Head()

        # not used during inference
        '''
        self.teacher = IFBlock(8+4+8+3+8, c=64)
        self.caltime = nn.Sequential(
            nn.Conv2d(16+9, 8, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        '''

    def forward(self, x, timestep=0.5, scale_list=[8, 4, 2, 1], training=False, fastmode=True, ensemble=False):
        if training == False:
            channel = x.shape[1] // 2
            img0 = x[:, :channel]
            img1 = x[:, channel:]
        if not torch.is_tensor(timestep):
            timestep = (x[:, :1].clone() * 0 + 1) * timestep
        else:
            timestep = timestep.repeat(1, 1, img0.shape[2], img0.shape[3])
        f0 = self.encode(img0[:, :3])
        f1 = self.encode(img1[:, :3])
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        mask = None
        loss_cons = 0
        block = [self.block0, self.block1, self.block2, self.block3, self.block4]
        for i in range(5):
            if flow is None:
                flow, mask, feat = block[i](torch.cat((img0[:, :3], img1[:, :3], f0, f1, timestep), 1), None, scale=scale_list[i])
                if ensemble:
                    print("warning: ensemble is not supported since RIFEv4.21")
            else:
                wf0 = warp(f0, flow[:, :2])
                wf1 = warp(f1, flow[:, 2:4])
                fd, m0, feat = block[i](torch.cat((warped_img0[:, :3], warped_img1[:, :3], wf0, wf1, timestep, mask, feat), 1), flow, scale=scale_list[i])
                if ensemble:
                    print("warning: ensemble is not supported since RIFEv4.21")
                else:
                    mask = m0
                flow = flow + fd
            mask_list.append(mask)
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged.append((warped_img0, warped_img1))
        mask = torch.sigmoid(mask)
        merged[4] = (warped_img0 * mask + warped_img1 * (1 - mask))
        if not fastmode:
            print('contextnet is removed')
            '''
            c0 = self.contextnet(img0, flow[:, :2])
            c1 = self.contextnet(img1, flow[:, 2:4])
            tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
            res = tmp[:, :3] * 2 - 1
            merged[4] = torch.clamp(merged[4] + res, 0, 1)
            '''
        return flow_list, mask_list[4], merged
