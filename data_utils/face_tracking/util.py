import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_tri_normal(geometry, tris):
    tri_1 = tris[:, 0]
    tri_2 = tris[:, 1]
    tri_3 = tris[:, 2]
    vert_1 = torch.index_select(geometry, 1, tri_1)
    vert_2 = torch.index_select(geometry, 1, tri_2)
    vert_3 = torch.index_select(geometry, 1, tri_3)
    nnorm = torch.cross(vert_2 - vert_1, vert_3 - vert_1, 2)
    normal = nn.functional.normalize(nnorm)
    return normal


def euler2rot(euler_angle):
    batch_size = euler_angle.shape[0]
    theta = euler_angle[:, 0].reshape(-1, 1, 1)
    phi = euler_angle[:, 1].reshape(-1, 1, 1)
    psi = euler_angle[:, 2].reshape(-1, 1, 1)
    one = torch.ones(batch_size, 1, 1).to(euler_angle.device)
    zero = torch.zeros(batch_size, 1, 1).to(euler_angle.device)
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


def rot_trans_pts(geometry, rot, trans):
    rott_geo = torch.bmm(rot, geometry.permute(0, 2, 1)) + trans[:, :, None]
    return rott_geo.permute(0, 2, 1)


def cal_lap_loss(tensor_list, weight_list):
    lap_kernel = (
        torch.Tensor((-0.5, 1.0, -0.5))
        .unsqueeze(0)
        .unsqueeze(0)
        .float()
        .to(tensor_list[0].device)
    )
    loss_lap = 0
    for i in range(len(tensor_list)):
        in_tensor = tensor_list[i]
        in_tensor = in_tensor.view(-1, 1, in_tensor.shape[-1])
        out_tensor = F.conv1d(in_tensor, lap_kernel)
        loss_lap += torch.mean(out_tensor ** 2) * weight_list[i]
    return loss_lap


def proj_pts(rott_geo, focal_length, cxy):
    cx, cy = cxy[0], cxy[1]
    X = rott_geo[:, :, 0]
    Y = rott_geo[:, :, 1]
    Z = rott_geo[:, :, 2]
    fxX = focal_length * X
    fyY = focal_length * Y
    proj_x = -fxX / Z + cx
    proj_y = fyY / Z + cy
    return torch.cat((proj_x[:, :, None], proj_y[:, :, None], Z[:, :, None]), 2)


def forward_rott(geometry, euler_angle, trans):
    rot = euler2rot(euler_angle)
    rott_geo = rot_trans_pts(geometry, rot, trans)
    return rott_geo


def forward_transform(geometry, euler_angle, trans, focal_length, cxy):
    rot = euler2rot(euler_angle)
    rott_geo = rot_trans_pts(geometry, rot, trans)
    proj_geo = proj_pts(rott_geo, focal_length, cxy)
    return proj_geo


def cal_lan_loss(proj_lan, gt_lan):
    return torch.mean((proj_lan - gt_lan) ** 2)


def cal_col_loss(pred_img, gt_img, img_mask):
    pred_img = pred_img.float()
    # loss = torch.sqrt(torch.sum(torch.square(pred_img - gt_img), 3))*img_mask/255
    loss = (torch.sum(torch.square(pred_img - gt_img), 3)) * img_mask / 255
    loss = torch.sum(loss, dim=(1, 2)) / torch.sum(img_mask, dim=(1, 2))
    loss = torch.mean(loss)
    return loss
