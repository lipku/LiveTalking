import math
import trimesh
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import raymarching
from .utils import custom_meshgrid, get_audio_features, euler_angles_to_matrix, convert_poses

def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # bins: [B, T], old_z_vals
    # weights: [B, T - 1], bin weights.
    # return: [B, n_samples], new_z_vals

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples).to(weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(weights.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (B, n_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def plot_pointcloud(pc, color=None):
    # pc: [N, 3]
    # color: [N, 3/4]
    print('[visualize points]', pc.shape, pc.dtype, pc.min(0), pc.max(0))
    pc = trimesh.PointCloud(pc, color)
    # axis
    axes = trimesh.creation.axis(axis_length=4)
    # sphere
    sphere = trimesh.creation.icosphere(radius=1)
    trimesh.Scene([pc, axes, sphere]).show()


class NeRFRenderer(nn.Module):
    def __init__(self, opt):

        super().__init__()

        self.opt = opt
        self.bound = opt.bound
        self.cascade = 1 + math.ceil(math.log2(opt.bound))
        self.grid_size = 128
        self.density_scale = 1

        self.min_near = opt.min_near
        self.density_thresh = opt.density_thresh
        self.density_thresh_torso = opt.density_thresh_torso

        self.exp_eye = opt.exp_eye
        self.test_train = opt.test_train
        self.smooth_lips = opt.smooth_lips

        self.torso = opt.torso
        self.cuda_ray = opt.cuda_ray

        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        # NOTE: aabb (can be rectangular) is only used to generate points, we still rely on bound (always cubic) to calculate density grid and hashing.
        aabb_train = torch.FloatTensor([-opt.bound, -opt.bound/2, -opt.bound, opt.bound, opt.bound/2, opt.bound])
        aabb_infer = aabb_train.clone()
        self.register_buffer('aabb_train', aabb_train)
        self.register_buffer('aabb_infer', aabb_infer)

        # individual codes
        self.individual_num = opt.ind_num

        self.individual_dim = opt.ind_dim
        if self.individual_dim > 0:
            self.individual_codes = nn.Parameter(torch.randn(self.individual_num, self.individual_dim) * 0.1) 
        
        if self.torso:
            self.individual_dim_torso = opt.ind_dim_torso
            if self.individual_dim_torso > 0:
                self.individual_codes_torso = nn.Parameter(torch.randn(self.individual_num, self.individual_dim_torso) * 0.1) 

        # optimize camera pose
        self.train_camera = self.opt.train_camera
        if self.train_camera:
            self.camera_dR = nn.Parameter(torch.zeros(self.individual_num, 3)) # euler angle
            self.camera_dT = nn.Parameter(torch.zeros(self.individual_num, 3)) # xyz offset

        # extra state for cuda raymarching
    
        # 3D head density grid
        density_grid = torch.zeros([self.cascade, self.grid_size ** 3]) # [CAS, H * H * H]
        density_bitfield = torch.zeros(self.cascade * self.grid_size ** 3 // 8, dtype=torch.uint8) # [CAS * H * H * H // 8]
        self.register_buffer('density_grid', density_grid)
        self.register_buffer('density_bitfield', density_bitfield)
        self.mean_density = 0
        self.iter_density = 0

        # 2D torso density grid
        if self.torso:
            density_grid_torso = torch.zeros([self.grid_size ** 2]) # [H * H]
            self.register_buffer('density_grid_torso', density_grid_torso)
        self.mean_density_torso = 0

        # step counter
        step_counter = torch.zeros(16, 2, dtype=torch.int32) # 16 is hardcoded for averaging...
        self.register_buffer('step_counter', step_counter)
        self.mean_count = 0
        self.local_step = 0
        
        # decay for enc_a
        if self.smooth_lips:
            self.enc_a = None
    
    def forward(self, x, d):
        raise NotImplementedError()

    # separated density and color query (can accelerate non-cuda-ray mode.)
    def density(self, x):
        raise NotImplementedError()

    def color(self, x, d, mask=None, **kwargs):
        raise NotImplementedError()

    def reset_extra_state(self):
        if not self.cuda_ray:
            return 
        # density grid
        self.density_grid.zero_()
        self.mean_density = 0
        self.iter_density = 0
        # step counter
        self.step_counter.zero_()
        self.mean_count = 0
        self.local_step = 0


    def run_cuda(self, rays_o, rays_d, auds, bg_coords, poses, eye=None, index=0, dt_gamma=0, bg_color=None, perturb=False, force_all_rays=False, max_steps=1024, T_thresh=1e-4, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # auds: [B, 16]
        # index: [B]
        # return: image: [B, N, 3], depth: [B, N]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)
        bg_coords = bg_coords.contiguous().view(-1, 2)

        # only add camera offset at training!
        if self.train_camera and (self.training or self.test_train):
            dT = self.camera_dT[index] # [1, 3]
            dR = euler_angles_to_matrix(self.camera_dR[index] / 180 * np.pi + 1e-8).squeeze(0) # [1, 3] --> [3, 3]
            
            rays_o = rays_o + dT
            rays_d = rays_d @ dR

        N = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device

        results = {}

        # pre-calculate near far
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, self.aabb_train if self.training else self.aabb_infer, self.min_near)
        nears = nears.detach()
        fars = fars.detach()

        # encode audio
        enc_a = self.encode_audio(auds) # [1, 64]

        if enc_a is not None and self.smooth_lips:
            if self.enc_a is not None:
                _lambda = 0.35
                enc_a = _lambda * self.enc_a + (1 - _lambda) * enc_a
            self.enc_a = enc_a

        
        if self.individual_dim > 0:
            if self.training:
                ind_code = self.individual_codes[index]
            # use a fixed ind code for the unknown test data.
            else:
                ind_code = self.individual_codes[0]
        else:
            ind_code = None

        if self.training:
            # setup counter
            counter = self.step_counter[self.local_step % 16]
            counter.zero_() # set to 0
            self.local_step += 1

            xyzs, dirs, deltas, rays = raymarching.march_rays_train(rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, counter, self.mean_count, perturb, 128, force_all_rays, dt_gamma, max_steps)
            sigmas, rgbs, amb_aud, amb_eye, uncertainty = self(xyzs, dirs, enc_a, ind_code, eye)
            sigmas = self.density_scale * sigmas

            #print(f'valid RGB query ratio: {mask.sum().item() / mask.shape[0]} (total = {mask.sum().item()})')

            # weights_sum, ambient_sum, uncertainty_sum, depth, image = raymarching.composite_rays_train_uncertainty(sigmas, rgbs, ambient.abs().sum(-1), uncertainty, deltas, rays)
            weights_sum, amb_aud_sum, amb_eye_sum, uncertainty_sum, depth, image = raymarching.composite_rays_train_triplane(sigmas, rgbs, amb_aud.abs().sum(-1), amb_eye.abs().sum(-1), uncertainty, deltas, rays)

            # for training only
            results['weights_sum'] = weights_sum
            results['ambient_aud'] = amb_aud_sum
            results['ambient_eye'] = amb_eye_sum
            results['uncertainty'] = uncertainty_sum

            results['rays'] = xyzs, dirs, enc_a, ind_code, eye

        else:
           
            dtype = torch.float32
            
            weights_sum = torch.zeros(N, dtype=dtype, device=device)
            depth = torch.zeros(N, dtype=dtype, device=device)
            image = torch.zeros(N, 3, dtype=dtype, device=device)
            amb_aud_sum = torch.zeros(N, dtype=dtype, device=device)
            amb_eye_sum = torch.zeros(N, dtype=dtype, device=device)
            uncertainty_sum = torch.zeros(N, dtype=dtype, device=device)

            n_alive = N
            rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device) # [N]
            rays_t = nears.clone() # [N]

            step = 0
            
            while step < max_steps:

                # count alive rays 
                n_alive = rays_alive.shape[0]
                
                # exit loop
                if n_alive <= 0:
                    break

                # decide compact_steps
                n_step = max(min(N // n_alive, 8), 1)

                xyzs, dirs, deltas = raymarching.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, 128, perturb if step == 0 else False, dt_gamma, max_steps)

                sigmas, rgbs, ambients_aud, ambients_eye, uncertainties = self(xyzs, dirs, enc_a, ind_code, eye)
                sigmas = self.density_scale * sigmas

                # raymarching.composite_rays_uncertainty(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas, ambients, uncertainties, weights_sum, depth, image, ambient_sum, uncertainty_sum, T_thresh)
                raymarching.composite_rays_triplane(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas, ambients_aud, ambients_eye, uncertainties, weights_sum, depth, image, amb_aud_sum, amb_eye_sum, uncertainty_sum, T_thresh)

                rays_alive = rays_alive[rays_alive >= 0]

                # print(f'step = {step}, n_step = {n_step}, n_alive = {n_alive}, xyzs: {xyzs.shape}')

                step += n_step
            
        torso_results = self.run_torso(rays_o, bg_coords, poses, index, bg_color)
        bg_color = torso_results['bg_color']

        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
        image = image.view(*prefix, 3)
        image = image.clamp(0, 1)

        depth = torch.clamp(depth - nears, min=0) / (fars - nears)
        depth = depth.view(*prefix)

        amb_aud_sum = amb_aud_sum.view(*prefix)
        amb_eye_sum = amb_eye_sum.view(*prefix)

        results['depth'] = depth
        results['image'] = image # head_image if train, else com_image
        results['ambient_aud'] = amb_aud_sum
        results['ambient_eye'] = amb_eye_sum
        results['uncertainty'] = uncertainty_sum

        return results
    

    def run_torso(self, rays_o, bg_coords, poses, index=0, bg_color=None, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # auds: [B, 16]
        # index: [B]
        # return: image: [B, N, 3], depth: [B, N]

        rays_o = rays_o.contiguous().view(-1, 3)
        bg_coords = bg_coords.contiguous().view(-1, 2)

        N = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device

        results = {}

        # background
        if bg_color is None:
            bg_color = 1

        # first mix torso with background
        if self.torso:
            # torso ind code
            if self.individual_dim_torso > 0:
                if self.training:
                    ind_code_torso = self.individual_codes_torso[index]
                # use a fixed ind code for the unknown test data.
                else:
                    ind_code_torso = self.individual_codes_torso[0]
            else:
                ind_code_torso = None
            
            # 2D density grid for acceleration...
            density_thresh_torso = min(self.density_thresh_torso, self.mean_density_torso)
            occupancy = F.grid_sample(self.density_grid_torso.view(1, 1, self.grid_size, self.grid_size), bg_coords.view(1, -1, 1, 2), align_corners=True).view(-1)
            mask = occupancy > density_thresh_torso

            # masked query of torso
            torso_alpha = torch.zeros([N, 1], device=device)
            torso_color = torch.zeros([N, 3], device=device)

            if mask.any():
                torso_alpha_mask, torso_color_mask, deform = self.forward_torso(bg_coords[mask], poses, ind_code_torso)

                torso_alpha[mask] = torso_alpha_mask.float()
                torso_color[mask] = torso_color_mask.float()

                results['deform'] = deform
            
            # first mix torso with background
            
            bg_color = torso_color * torso_alpha + bg_color * (1 - torso_alpha)

            results['torso_alpha'] = torso_alpha
            results['torso_color'] = bg_color

            # print(torso_alpha.shape, torso_alpha.max().item(), torso_alpha.min().item())
        
        results['bg_color'] = bg_color
        
        return results


    @torch.no_grad()
    def mark_untrained_grid(self, poses, intrinsic, S=64):
        # poses: [B, 4, 4]
        # intrinsic: [3, 3]

        if not self.cuda_ray:
            return
        
        if isinstance(poses, np.ndarray):
            poses = torch.from_numpy(poses)

        B = poses.shape[0]
        
        fx, fy, cx, cy = intrinsic
        
        X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)

        count = torch.zeros_like(self.density_grid)
        poses = poses.to(count.device)

        # 5-level loop, forgive me...

        for xs in X:
            for ys in Y:
                for zs in Z:
                    
                    # construct points
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
                    indices = raymarching.morton3D(coords).long() # [N]
                    world_xyzs = (2 * coords.float() / (self.grid_size - 1) - 1).unsqueeze(0) # [1, N, 3] in [-1, 1]

                    # cascading
                    for cas in range(self.cascade):
                        bound = min(2 ** cas, self.bound)
                        half_grid_size = bound / self.grid_size
                        # scale to current cascade's resolution
                        cas_world_xyzs = world_xyzs * (bound - half_grid_size)

                        # split batch to avoid OOM
                        head = 0
                        while head < B:
                            tail = min(head + S, B)

                            # world2cam transform (poses is c2w, so we need to transpose it. Another transpose is needed for batched matmul, so the final form is without transpose.)
                            cam_xyzs = cas_world_xyzs - poses[head:tail, :3, 3].unsqueeze(1)
                            cam_xyzs = cam_xyzs @ poses[head:tail, :3, :3] # [S, N, 3]
                            
                            # query if point is covered by any camera
                            mask_z = cam_xyzs[:, :, 2] > 0 # [S, N]
                            mask_x = torch.abs(cam_xyzs[:, :, 0]) < cx / fx * cam_xyzs[:, :, 2] + half_grid_size * 2
                            mask_y = torch.abs(cam_xyzs[:, :, 1]) < cy / fy * cam_xyzs[:, :, 2] + half_grid_size * 2
                            mask = (mask_z & mask_x & mask_y).sum(0).reshape(-1) # [N]

                            # update count 
                            count[cas, indices] += mask
                            head += S
    
        # mark untrained grid as -1
        self.density_grid[count == 0] = -1

        #print(f'[mark untrained grid] {(count == 0).sum()} from {resolution ** 3 * self.cascade}')

    @torch.no_grad()
    def update_extra_state(self, decay=0.95, S=128):
        # call before each epoch to update extra states.

        if not self.cuda_ray:
            return 
        
        # use random auds (different expressions should have similar density grid...)
        rand_idx = random.randint(0, self.aud_features.shape[0] - 1)
        auds = get_audio_features(self.aud_features, self.att, rand_idx).to(self.density_bitfield.device)

        # encode audio
        enc_a = self.encode_audio(auds)

        ### update density grid
        if not self.torso: # forbid updating head if is training torso...

            tmp_grid = torch.zeros_like(self.density_grid)

            # use a random eye area based on training dataset's statistics...
            if self.exp_eye:
                eye = self.eye_area[[rand_idx]].to(self.density_bitfield.device) # [1, 1]
            else:
                eye = None
            
            # full update
            X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
            Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
            Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)

            for xs in X:
                for ys in Y:
                    for zs in Z:
                        
                        # construct points
                        xx, yy, zz = custom_meshgrid(xs, ys, zs)
                        coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
                        indices = raymarching.morton3D(coords).long() # [N]
                        xyzs = 2 * coords.float() / (self.grid_size - 1) - 1 # [N, 3] in [-1, 1]

                        # cascading
                        for cas in range(self.cascade):
                            bound = min(2 ** cas, self.bound)
                            half_grid_size = bound / self.grid_size
                            # scale to current cascade's resolution
                            cas_xyzs = xyzs * (bound - half_grid_size)
                            # add noise in [-hgs, hgs]
                            cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                            # query density
                            sigmas = self.density(cas_xyzs, enc_a, eye)['sigma'].reshape(-1).detach().to(tmp_grid.dtype)
                            sigmas *= self.density_scale
                            # assign 
                            tmp_grid[cas, indices] = sigmas
            
            # dilate the density_grid (less aggressive culling)
            tmp_grid = raymarching.morton3D_dilation(tmp_grid)

            # ema update
            valid_mask = (self.density_grid >= 0) & (tmp_grid >= 0)
            self.density_grid[valid_mask] = torch.maximum(self.density_grid[valid_mask] * decay, tmp_grid[valid_mask])
            self.mean_density = torch.mean(self.density_grid.clamp(min=0)).item() # -1 non-training regions are viewed as 0 density.
            self.iter_density += 1

            # convert to bitfield
            density_thresh = min(self.mean_density, self.density_thresh)
            self.density_bitfield = raymarching.packbits(self.density_grid, density_thresh, self.density_bitfield)

        ### update torso density grid
        if self.torso:
            tmp_grid_torso = torch.zeros_like(self.density_grid_torso)

            # random pose, random ind_code
            rand_idx = random.randint(0, self.poses.shape[0] - 1)
            # pose = convert_poses(self.poses[[rand_idx]]).to(self.density_bitfield.device)
            pose = self.poses[[rand_idx]].to(self.density_bitfield.device)

            if self.opt.ind_dim_torso > 0:
                ind_code = self.individual_codes_torso[[rand_idx]]
            else:
                ind_code = None

            X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
            Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)

            half_grid_size = 1 / self.grid_size

            for xs in X:
                for ys in Y:
                    xx, yy = custom_meshgrid(xs, ys)
                    coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=-1) # [N, 2], in [0, 128)
                    indices = (coords[:, 1] * self.grid_size + coords[:, 0]).long() # NOTE: xy transposed!
                    xys = 2 * coords.float() / (self.grid_size - 1) - 1 # [N, 2] in [-1, 1]
                    xys = xys * (1 - half_grid_size)
                    # add noise in [-hgs, hgs]
                    xys += (torch.rand_like(xys) * 2 - 1) * half_grid_size
                    # query density
                    alphas, _, _ = self.forward_torso(xys, pose, ind_code) # [N, 1]
                    
                    # assign 
                    tmp_grid_torso[indices] = alphas.squeeze(1).float()

            # dilate
            tmp_grid_torso = tmp_grid_torso.view(1, 1, self.grid_size, self.grid_size)
            # tmp_grid_torso = F.max_pool2d(tmp_grid_torso, kernel_size=3, stride=1, padding=1)
            tmp_grid_torso = F.max_pool2d(tmp_grid_torso, kernel_size=5, stride=1, padding=2)
            tmp_grid_torso = tmp_grid_torso.view(-1)
            
            self.density_grid_torso = torch.maximum(self.density_grid_torso * decay, tmp_grid_torso)
            self.mean_density_torso = torch.mean(self.density_grid_torso).item()

            # density_thresh_torso = min(self.density_thresh_torso, self.mean_density_torso)
            # print(f'[density grid torso] min={self.density_grid_torso.min().item():.4f}, max={self.density_grid_torso.max().item():.4f}, mean={self.mean_density_torso:.4f}, occ_rate={(self.density_grid_torso > density_thresh_torso).sum() / (128**2):.3f}')

        ### update step counter
        total_step = min(16, self.local_step)
        if total_step > 0:
            self.mean_count = int(self.step_counter[:total_step, 0].sum().item() / total_step)
        self.local_step = 0

        #print(f'[density grid] min={self.density_grid.min().item():.4f}, max={self.density_grid.max().item():.4f}, mean={self.mean_density:.4f}, occ_rate={(self.density_grid > 0.01).sum() / (128**3 * self.cascade):.3f} | [step counter] mean={self.mean_count}')


    @torch.no_grad()
    def get_audio_grid(self,  S=128):
        # call before each epoch to update extra states.

        if not self.cuda_ray:
            return 
        
        # use random auds (different expressions should have similar density grid...)
        rand_idx = random.randint(0, self.aud_features.shape[0] - 1)
        auds = get_audio_features(self.aud_features, self.att, rand_idx).to(self.density_bitfield.device)

        # encode audio
        enc_a = self.encode_audio(auds)
        tmp_grid = torch.zeros_like(self.density_grid)

        # use a random eye area based on training dataset's statistics...
        if self.exp_eye:
            eye = self.eye_area[[rand_idx]].to(self.density_bitfield.device) # [1, 1]
        else:
            eye = None
        
        # full update
        X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)

        for xs in X:
            for ys in Y:
                for zs in Z:
                    
                    # construct points
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
                    indices = raymarching.morton3D(coords).long() # [N]
                    xyzs = 2 * coords.float() / (self.grid_size - 1) - 1 # [N, 3] in [-1, 1]

                    # cascading
                    for cas in range(self.cascade):
                        bound = min(2 ** cas, self.bound)
                        half_grid_size = bound / self.grid_size
                        # scale to current cascade's resolution
                        cas_xyzs = xyzs * (bound - half_grid_size)
                        # add noise in [-hgs, hgs]
                        cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                        # query density
                        aud_norms = self.density(cas_xyzs.to(tmp_grid.dtype), enc_a, eye)['ambient_aud'].reshape(-1).detach().to(tmp_grid.dtype)
                        # assign 
                        tmp_grid[cas, indices] = aud_norms
        
        # dilate the density_grid (less aggressive culling)
        tmp_grid = raymarching.morton3D_dilation(tmp_grid)
        return tmp_grid
        # # ema update
        # valid_mask = (self.density_grid >= 0) & (tmp_grid >= 0)
        # self.density_grid[valid_mask] = torch.maximum(self.density_grid[valid_mask] * decay, tmp_grid[valid_mask])


    @torch.no_grad()
    def get_eye_grid(self,  S=128):
        # call before each epoch to update extra states.

        if not self.cuda_ray:
            return 
        
        # use random auds (different expressions should have similar density grid...)
        rand_idx = random.randint(0, self.aud_features.shape[0] - 1)
        auds = get_audio_features(self.aud_features, self.att, rand_idx).to(self.density_bitfield.device)

        # encode audio
        enc_a = self.encode_audio(auds)
        tmp_grid = torch.zeros_like(self.density_grid)

        # use a random eye area based on training dataset's statistics...
        if self.exp_eye:
            eye = self.eye_area[[rand_idx]].to(self.density_bitfield.device) # [1, 1]
        else:
            eye = None
        
        # full update
        X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)

        for xs in X:
            for ys in Y:
                for zs in Z:
                    
                    # construct points
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
                    indices = raymarching.morton3D(coords).long() # [N]
                    xyzs = 2 * coords.float() / (self.grid_size - 1) - 1 # [N, 3] in [-1, 1]

                    # cascading
                    for cas in range(self.cascade):
                        bound = min(2 ** cas, self.bound)
                        half_grid_size = bound / self.grid_size
                        # scale to current cascade's resolution
                        cas_xyzs = xyzs * (bound - half_grid_size)
                        # add noise in [-hgs, hgs]
                        cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                        # query density
                        eye_norms = self.density(cas_xyzs.to(tmp_grid.dtype), enc_a, eye)['ambient_eye'].reshape(-1).detach().to(tmp_grid.dtype)
                        # assign 
                        tmp_grid[cas, indices] = eye_norms
        
        # dilate the density_grid (less aggressive culling)
        tmp_grid = raymarching.morton3D_dilation(tmp_grid)
        return tmp_grid
        # # ema update
        # valid_mask = (self.density_grid >= 0) & (tmp_grid >= 0)
        # self.density_grid[valid_mask] = torch.maximum(self.density_grid[valid_mask] * decay, tmp_grid[valid_mask])



    def render(self, rays_o, rays_d, auds, bg_coords, poses, staged=False, max_ray_batch=4096, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # auds: [B, 29, 16]
        # eye: [B, 1]
        # bg_coords: [1, N, 2]
        # return: pred_rgb: [B, N, 3]

        _run = self.run_cuda
        
        B, N = rays_o.shape[:2]
        device = rays_o.device

        # never stage when cuda_ray
        if staged and not self.cuda_ray:
            # not used
            raise NotImplementedError

        else:
            results = _run(rays_o, rays_d, auds, bg_coords, poses, **kwargs)

        return results
    
    
    def render_torso(self, rays_o, rays_d, auds, bg_coords, poses, staged=False, max_ray_batch=4096, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # auds: [B, 29, 16]
        # eye: [B, 1]
        # bg_coords: [1, N, 2]
        # return: pred_rgb: [B, N, 3]

        _run = self.run_torso
        
        B, N = rays_o.shape[:2]
        device = rays_o.device

        # never stage when cuda_ray
        if staged and not self.cuda_ray:
            # not used
            raise NotImplementedError

        else:
            results = _run(rays_o, bg_coords, poses, **kwargs)

        return results