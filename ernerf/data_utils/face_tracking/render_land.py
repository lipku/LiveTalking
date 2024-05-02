import torch
import torch.nn as nn
import render_util
import geo_transform
import numpy as np


def compute_tri_normal(geometry, tris):
    geometry = geometry.permute(0, 2, 1)
    tri_1 = tris[:, 0]
    tri_2 = tris[:, 1]
    tri_3 = tris[:, 2]

    vert_1 = torch.index_select(geometry, 2, tri_1)
    vert_2 = torch.index_select(geometry, 2, tri_2)
    vert_3 = torch.index_select(geometry, 2, tri_3)

    nnorm = torch.cross(vert_2 - vert_1, vert_3 - vert_1, 1)
    normal = nn.functional.normalize(nnorm).permute(0, 2, 1)
    return normal


class Compute_normal_base(torch.autograd.Function):
    @staticmethod
    def forward(ctx, normal):
        (normal_b,) = render_util.normal_base_forward(normal)
        ctx.save_for_backward(normal)
        return normal_b

    @staticmethod
    def backward(ctx, grad_normal_b):
        (normal,) = ctx.saved_tensors
        (grad_normal,) = render_util.normal_base_backward(grad_normal_b, normal)
        return grad_normal


class Normal_Base(torch.nn.Module):
    def __init__(self):
        super(Normal_Base, self).__init__()

    def forward(self, normal):
        return Compute_normal_base.apply(normal)


def preprocess_render(geometry, euler, trans, cam, tris, vert_tris, ori_img):
    point_num = geometry.shape[1]
    rott_geo = geo_transform.euler_trans_geo(geometry, euler, trans)
    proj_geo = geo_transform.proj_geo(rott_geo, cam)
    rot_tri_normal = compute_tri_normal(rott_geo, tris)
    rot_vert_normal = torch.index_select(rot_tri_normal, 1, vert_tris)
    is_visible = -torch.bmm(
        rot_vert_normal.reshape(-1, 1, 3),
        nn.functional.normalize(rott_geo.reshape(-1, 3, 1)),
    ).reshape(-1, point_num)
    is_visible[is_visible < 0.01] = -1
    pixel_valid = torch.zeros(
        (ori_img.shape[0], ori_img.shape[1] * ori_img.shape[2]),
        dtype=torch.float32,
        device=ori_img.device,
    )
    return rott_geo, proj_geo, rot_tri_normal, is_visible, pixel_valid


class Render_Face(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, proj_geo, texture, nbl, ori_img, is_visible, tri_inds, pixel_valid
    ):
        batch_size, h, w, _ = ori_img.shape
        ori_img = ori_img.view(batch_size, -1, 3)
        ori_size = torch.cat(
            (
                torch.ones((batch_size, 1), dtype=torch.int32, device=ori_img.device)
                * h,
                torch.ones((batch_size, 1), dtype=torch.int32, device=ori_img.device)
                * w,
            ),
            dim=1,
        ).view(-1)
        tri_index, tri_coord, render, real = render_util.render_face_forward(
            proj_geo, ori_img, ori_size, texture, nbl, is_visible, tri_inds, pixel_valid
        )
        ctx.save_for_backward(
            ori_img, ori_size, proj_geo, texture, nbl, tri_inds, tri_index, tri_coord
        )
        return render, real

    @staticmethod
    def backward(ctx, grad_render, grad_real):
        (
            ori_img,
            ori_size,
            proj_geo,
            texture,
            nbl,
            tri_inds,
            tri_index,
            tri_coord,
        ) = ctx.saved_tensors
        grad_proj_geo, grad_texture, grad_nbl = render_util.render_face_backward(
            grad_render,
            grad_real,
            ori_img,
            ori_size,
            proj_geo,
            texture,
            nbl,
            tri_inds,
            tri_index,
            tri_coord,
        )
        return grad_proj_geo, grad_texture, grad_nbl, None, None, None, None


class Render_RGB(nn.Module):
    def __init__(self):
        super(Render_RGB, self).__init__()

    def forward(
        self, proj_geo, texture, nbl, ori_img, is_visible, tri_inds, pixel_valid
    ):
        return Render_Face.apply(
            proj_geo, texture, nbl, ori_img, is_visible, tri_inds, pixel_valid
        )


def cal_land(proj_geo, is_visible, lands_info, land_num):
    (land_index,) = render_util.update_contour(lands_info, is_visible, land_num)
    proj_land = torch.index_select(proj_geo.reshape(-1, 3), 0, land_index)[
        :, :2
    ].reshape(-1, land_num, 2)
    return proj_land


class Render_Land(nn.Module):
    def __init__(self):
        super(Render_Land, self).__init__()
        lands_info = np.loadtxt("../data/3DMM/lands_info.txt", dtype=np.int32)
        self.lands_info = torch.as_tensor(lands_info).cuda()
        tris = np.loadtxt("../data/3DMM/tris.txt", dtype=np.int64)
        self.tris = torch.as_tensor(tris).cuda() - 1
        vert_tris = np.loadtxt("../data/3DMM/vert_tris.txt", dtype=np.int64)
        self.vert_tris = torch.as_tensor(vert_tris).cuda()
        self.normal_baser = Normal_Base().cuda()
        self.renderer = Render_RGB().cuda()

    def render_mesh(self, geometry, euler, trans, cam, ori_img, light):
        batch_size, h, w, _ = ori_img.shape
        ori_img = ori_img.view(batch_size, -1, 3)
        ori_size = torch.cat(
            (
                torch.ones((batch_size, 1), dtype=torch.int32, device=ori_img.device)
                * h,
                torch.ones((batch_size, 1), dtype=torch.int32, device=ori_img.device)
                * w,
            ),
            dim=1,
        ).view(-1)
        rott_geo, proj_geo, rot_tri_normal, _, _ = preprocess_render(
            geometry, euler, trans, cam, self.tris, self.vert_tris, ori_img
        )
        tri_nb = self.normal_baser(rot_tri_normal.contiguous())
        nbl = torch.bmm(
            tri_nb, (light.reshape(-1, 9, 3))[:, :, 0].unsqueeze(-1).repeat(1, 1, 3)
        )
        texture = torch.ones_like(geometry) * 200
        (render,) = render_util.render_mesh(
            proj_geo, ori_img, ori_size, texture, nbl, self.tris
        )
        return render.view(batch_size, h, w, 3).byte()

    def cal_loss_rgb(self, geometry, euler, trans, cam, ori_img, light, texture, lands):
        rott_geo, proj_geo, rot_tri_normal, is_visible, pixel_valid = preprocess_render(
            geometry, euler, trans, cam, self.tris, self.vert_tris, ori_img
        )
        tri_nb = self.normal_baser(rot_tri_normal.contiguous())
        nbl = torch.bmm(tri_nb, light.reshape(-1, 9, 3))
        render, real = self.renderer(
            proj_geo, texture, nbl, ori_img, is_visible, self.tris, pixel_valid
        )
        proj_land = cal_land(proj_geo, is_visible, self.lands_info, lands.shape[1])
        col_minus = torch.norm((render - real).reshape(-1, 3), dim=1).reshape(
            ori_img.shape[0], -1
        )
        col_dis = torch.mean(col_minus * pixel_valid) / (
            torch.mean(pixel_valid) + 0.00001
        )
        land_dists = torch.norm((proj_land - lands).reshape(-1, 2), dim=1).reshape(
            ori_img.shape[0], -1
        )
        lan_dis = torch.mean(land_dists)
        return col_dis, lan_dis
