import torch
import torch.nn as nn

backwarp_tenGrid = {}


def warp(tenInput, tenFlow):
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        _dev = tenFlow.device
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=_dev).view(
            1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=_dev).view(
            1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[k] = torch.cat(
            [tenHorizontal, tenVertical], 1).to(_dev)

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    # MPS doesn't support padding_mode='border', so clamp grid to [-1,1] + use 'zeros'
    # Clamping achieves the same effect as border padding (nearest edge pixel)
    g = g.clamp(-1, 1)
    return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='zeros', align_corners=True)
