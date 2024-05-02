"""This module contains functions for geometry transform and camera projection"""
import torch
import torch.nn as nn
import numpy as np


def euler2rot(euler_angle):
    batch_size = euler_angle.shape[0]
    theta = euler_angle[:, 0].reshape(-1, 1, 1)
    phi = euler_angle[:, 1].reshape(-1, 1, 1)
    psi = euler_angle[:, 2].reshape(-1, 1, 1)
    one = torch.ones((batch_size, 1, 1), dtype=torch.float32, device=euler_angle.device)
    zero = torch.zeros(
        (batch_size, 1, 1), dtype=torch.float32, device=euler_angle.device
    )
    rot_x = torch.cat(
        (
            torch.cat((one, zero, zero), 1),
            torch.cat((zero, theta.cos(), theta.sin()), 1),
            torch.cat((zero, -theta.sin(), theta.cos()), 1),
        ),
        2,
    )
    rot_y = torch.cat(
        (
            torch.cat((phi.cos(), zero, -phi.sin()), 1),
            torch.cat((zero, one, zero), 1),
            torch.cat((phi.sin(), zero, phi.cos()), 1),
        ),
        2,
    )
    rot_z = torch.cat(
        (
            torch.cat((psi.cos(), -psi.sin(), zero), 1),
            torch.cat((psi.sin(), psi.cos(), zero), 1),
            torch.cat((zero, zero, one), 1),
        ),
        2,
    )
    return torch.bmm(rot_x, torch.bmm(rot_y, rot_z))


def rot_trans_geo(geometry, rot, trans):
    rott_geo = torch.bmm(rot, geometry.permute(0, 2, 1)) + trans.view(-1, 3, 1)
    return rott_geo.permute(0, 2, 1)


def euler_trans_geo(geometry, euler, trans):
    rot = euler2rot(euler)
    return rot_trans_geo(geometry, rot, trans)


def proj_geo(rott_geo, camera_para):
    fx = camera_para[:, 0]
    fy = camera_para[:, 0]
    cx = camera_para[:, 1]
    cy = camera_para[:, 2]

    X = rott_geo[:, :, 0]
    Y = rott_geo[:, :, 1]
    Z = rott_geo[:, :, 2]

    fxX = fx[:, None] * X
    fyY = fy[:, None] * Y

    proj_x = -fxX / Z + cx[:, None]
    proj_y = fyY / Z + cy[:, None]

    return torch.cat((proj_x[:, :, None], proj_y[:, :, None], Z[:, :, None]), 2)
