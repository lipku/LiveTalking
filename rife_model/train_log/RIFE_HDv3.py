"""
RIFE v4.25 lite — inference-only wrapper (training deps stripped).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_log.IFNet_HDv3 import IFNet

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
          else "cpu")
)

class Model:
    def __init__(self, local_rank=-1):
        self.flownet = IFNet()
        self.device()
        self.version = 4.25

    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(device)

    def load_model(self, path, rank=0):
        def convert(param):
            if rank == -1:
                return {
                    k.replace("module.", ""): v
                    for k, v in param.items()
                    if "module." in k
                }
            else:
                return param
        if rank <= 0:
            if torch.cuda.is_available():
                self.flownet.load_state_dict(
                    convert(torch.load('{}/flownet.pkl'.format(path))), False)
            else:
                self.flownet.load_state_dict(
                    convert(torch.load('{}/flownet.pkl'.format(path),
                                       map_location='cpu')), False)

    @torch.no_grad()
    def inference(self, img0, img1, timestep=0.5, scale=1.0):
        # Pad to multiple of 128 (scale=32 × conv 4x downscale inside IFBlock)
        n, c, h, w = img0.shape
        ph = ((h - 1) // 128 + 1) * 128
        pw = ((w - 1) // 128 + 1) * 128
        pad = (0, pw - w, 0, ph - h)
        if pad != (0, 0, 0, 0):
            img0 = F.pad(img0, pad)
            img1 = F.pad(img1, pad)
        imgs = torch.cat((img0, img1), 1)
        scale_list = [32/scale, 16/scale, 8/scale, 4/scale, 1/scale]
        flow, mask, merged = self.flownet(imgs, timestep, scale_list)
        out = merged[-1]
        if pad != (0, 0, 0, 0):
            out = out[:, :, :h, :w]
        return out
