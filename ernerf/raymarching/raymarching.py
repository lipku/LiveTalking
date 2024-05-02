import numpy as np
import time

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

try:
    import _raymarching_face as _backend
except ImportError:
    from .backend import _backend

# ----------------------------------------
# utils
# ----------------------------------------

class _near_far_from_aabb(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, rays_o, rays_d, aabb, min_near=0.2):
        ''' near_far_from_aabb, CUDA implementation
        Calculate rays' intersection time (near and far) with aabb
        Args:
            rays_o: float, [N, 3]
            rays_d: float, [N, 3]
            aabb: float, [6], (xmin, ymin, zmin, xmax, ymax, zmax)
            min_near: float, scalar
        Returns:
            nears: float, [N]
            fars: float, [N]
        '''
        if not rays_o.is_cuda: rays_o = rays_o.cuda()
        if not rays_d.is_cuda: rays_d = rays_d.cuda()

        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # num rays

        nears = torch.empty(N, dtype=rays_o.dtype, device=rays_o.device)
        fars = torch.empty(N, dtype=rays_o.dtype, device=rays_o.device)

        _backend.near_far_from_aabb(rays_o, rays_d, aabb, N, min_near, nears, fars)

        return nears, fars

near_far_from_aabb = _near_far_from_aabb.apply


class _sph_from_ray(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, rays_o, rays_d, radius):
        ''' sph_from_ray, CUDA implementation
        get spherical coordinate on the background sphere from rays.
        Assume rays_o are inside the Sphere(radius).
        Args:
            rays_o: [N, 3]
            rays_d: [N, 3]
            radius: scalar, float
        Return:
            coords: [N, 2], in [-1, 1], theta and phi on a sphere. (further-surface)
        '''
        if not rays_o.is_cuda: rays_o = rays_o.cuda()
        if not rays_d.is_cuda: rays_d = rays_d.cuda()

        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # num rays

        coords = torch.empty(N, 2, dtype=rays_o.dtype, device=rays_o.device)

        _backend.sph_from_ray(rays_o, rays_d, radius, N, coords)

        return coords

sph_from_ray = _sph_from_ray.apply


class _morton3D(Function):
    @staticmethod
    def forward(ctx, coords):
        ''' morton3D, CUDA implementation
        Args:
            coords: [N, 3], int32, in [0, 128) (for some reason there is no uint32 tensor in torch...) 
            TODO: check if the coord range is valid! (current 128 is safe)
        Returns:
            indices: [N], int32, in [0, 128^3)
            
        '''
        if not coords.is_cuda: coords = coords.cuda()
        
        N = coords.shape[0]

        indices = torch.empty(N, dtype=torch.int32, device=coords.device)
        
        _backend.morton3D(coords.int(), N, indices)

        return indices

morton3D = _morton3D.apply

class _morton3D_invert(Function):
    @staticmethod
    def forward(ctx, indices):
        ''' morton3D_invert, CUDA implementation
        Args:
            indices: [N], int32, in [0, 128^3)
        Returns:
            coords: [N, 3], int32, in [0, 128)
            
        '''
        if not indices.is_cuda: indices = indices.cuda()
        
        N = indices.shape[0]

        coords = torch.empty(N, 3, dtype=torch.int32, device=indices.device)
        
        _backend.morton3D_invert(indices.int(), N, coords)

        return coords

morton3D_invert = _morton3D_invert.apply


class _packbits(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, grid, thresh, bitfield=None):
        ''' packbits, CUDA implementation
        Pack up the density grid into a bit field to accelerate ray marching.
        Args:
            grid: float, [C, H * H * H], assume H % 2 == 0
            thresh: float, threshold
        Returns:
            bitfield: uint8, [C, H * H * H / 8]
        '''
        if not grid.is_cuda: grid = grid.cuda()
        grid = grid.contiguous()

        C = grid.shape[0]
        H3 = grid.shape[1]
        N = C * H3 // 8

        if bitfield is None:
            bitfield = torch.empty(N, dtype=torch.uint8, device=grid.device)

        _backend.packbits(grid, N, thresh, bitfield)

        return bitfield

packbits = _packbits.apply


class _morton3D_dilation(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, grid):
        ''' max pooling with morton coord, CUDA implementation
        or maybe call it dilation... we don't support adjust kernel size.
        Args:
            grid: float, [C, H * H * H], assume H % 2 == 0
        Returns:
            grid_dilate: float, [C, H * H * H], assume H % 2 == 0bitfield: uint8, [C, H * H * H / 8]
        '''
        if not grid.is_cuda: grid = grid.cuda()
        grid = grid.contiguous()

        C = grid.shape[0]
        H3 = grid.shape[1]
        H = int(np.cbrt(H3))
        grid_dilation = torch.empty_like(grid)

        _backend.morton3D_dilation(grid, C, H, grid_dilation)

        return grid_dilation

morton3D_dilation = _morton3D_dilation.apply

# ----------------------------------------
# train functions
# ----------------------------------------

class _march_rays_train(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, rays_o, rays_d, bound, density_bitfield, C, H, nears, fars, step_counter=None, mean_count=-1, perturb=False, align=-1, force_all_rays=False, dt_gamma=0, max_steps=1024):
        ''' march rays to generate points (forward only)
        Args:
            rays_o/d: float, [N, 3]
            bound: float, scalar
            density_bitfield: uint8: [CHHH // 8]
            C: int
            H: int
            nears/fars: float, [N]
            step_counter: int32, (2), used to count the actual number of generated points.
            mean_count: int32, estimated mean steps to accelerate training. (but will randomly drop rays if the actual point count exceeded this threshold.)
            perturb: bool
            align: int, pad output so its size is dividable by align, set to -1 to disable.
            force_all_rays: bool, ignore step_counter and mean_count, always calculate all rays. Useful if rendering the whole image, instead of some rays.
            dt_gamma: float, called cone_angle in instant-ngp, exponentially accelerate ray marching if > 0. (very significant effect, but generally lead to worse performance)
            max_steps: int, max number of sampled points along each ray, also affect min_stepsize.
        Returns:
            xyzs: float, [M, 3], all generated points' coords. (all rays concated, need to use `rays` to extract points belonging to each ray)
            dirs: float, [M, 3], all generated points' view dirs.
            deltas: float, [M, 2], first is delta_t, second is rays_t
            rays: int32, [N, 3], all rays' (index, point_offset, point_count), e.g., xyzs[rays[i, 1]:rays[i, 1] + rays[i, 2]] --> points belonging to rays[i, 0]
        '''

        if not rays_o.is_cuda: rays_o = rays_o.cuda()
        if not rays_d.is_cuda: rays_d = rays_d.cuda()
        if not density_bitfield.is_cuda: density_bitfield = density_bitfield.cuda()
        
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)
        density_bitfield = density_bitfield.contiguous()

        N = rays_o.shape[0] # num rays
        M = N * max_steps # init max points number in total

        # running average based on previous epoch (mimic `measured_batch_size_before_compaction` in instant-ngp)
        # It estimate the max points number to enable faster training, but will lead to random ignored rays if underestimated.
        if not force_all_rays and mean_count > 0:
            if align > 0:
                mean_count += align - mean_count % align
            M = mean_count
        
        xyzs = torch.zeros(M, 3, dtype=rays_o.dtype, device=rays_o.device)
        dirs = torch.zeros(M, 3, dtype=rays_o.dtype, device=rays_o.device)
        deltas = torch.zeros(M, 2, dtype=rays_o.dtype, device=rays_o.device)
        rays = torch.empty(N, 3, dtype=torch.int32, device=rays_o.device) # id, offset, num_steps

        if step_counter is None:
            step_counter = torch.zeros(2, dtype=torch.int32, device=rays_o.device) # point counter, ray counter
        
        if perturb:
            noises = torch.rand(N, dtype=rays_o.dtype, device=rays_o.device)
        else:
            noises = torch.zeros(N, dtype=rays_o.dtype, device=rays_o.device)
        
        _backend.march_rays_train(rays_o, rays_d, density_bitfield, bound, dt_gamma, max_steps, N, C, H, M, nears, fars, xyzs, dirs, deltas, rays, step_counter, noises) # m is the actually used points number

        #print(step_counter, M)

        # only used at the first (few) epochs.
        if force_all_rays or mean_count <= 0:
            m = step_counter[0].item() # D2H copy
            if align > 0:
                m += align - m % align
            xyzs = xyzs[:m]
            dirs = dirs[:m]
            deltas = deltas[:m]

            torch.cuda.empty_cache()
        
        ctx.save_for_backward(rays, deltas)

        return xyzs, dirs, deltas, rays

    # to support optimizing camera poses.
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_xyzs, grad_dirs, grad_deltas, grad_rays):
        # grad_xyzs/dirs: [M, 3]
        
        rays, deltas = ctx.saved_tensors

        N = rays.shape[0]
        M = grad_xyzs.shape[0]

        grad_rays_o = torch.zeros(N, 3, device=rays.device)
        grad_rays_d = torch.zeros(N, 3, device=rays.device)
        
        _backend.march_rays_train_backward(grad_xyzs, grad_dirs, rays, deltas, N, M, grad_rays_o, grad_rays_d)
        
        return grad_rays_o, grad_rays_d, None, None, None, None, None, None, None, None, None, None, None, None, None

march_rays_train = _march_rays_train.apply


class _composite_rays_train(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, sigmas, rgbs, ambient, deltas, rays, T_thresh=1e-4):
        ''' composite rays' rgbs, according to the ray marching formula.
        Args:
            rgbs: float, [M, 3]
            sigmas: float, [M,]
            ambient: float, [M,] (after summing up the last dimension)
            deltas: float, [M, 2]
            rays: int32, [N, 3]
        Returns:
            weights_sum: float, [N,], the alpha channel
            depth: float, [N, ], the Depth
            image: float, [N, 3], the RGB channel (after multiplying alpha!)
        '''
        
        sigmas = sigmas.contiguous()
        rgbs = rgbs.contiguous()
        ambient = ambient.contiguous()

        M = sigmas.shape[0]
        N = rays.shape[0]

        weights_sum = torch.empty(N, dtype=sigmas.dtype, device=sigmas.device)
        ambient_sum = torch.empty(N, dtype=sigmas.dtype, device=sigmas.device)
        depth = torch.empty(N, dtype=sigmas.dtype, device=sigmas.device)
        image = torch.empty(N, 3, dtype=sigmas.dtype, device=sigmas.device)

        _backend.composite_rays_train_forward(sigmas, rgbs, ambient, deltas, rays, M, N, T_thresh, weights_sum, ambient_sum, depth, image)

        ctx.save_for_backward(sigmas, rgbs, ambient, deltas, rays, weights_sum, ambient_sum, depth, image)
        ctx.dims = [M, N, T_thresh]

        return weights_sum, ambient_sum, depth, image
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_weights_sum, grad_ambient_sum, grad_depth, grad_image):

        # NOTE: grad_depth is not used now! It won't be propagated to sigmas.

        grad_weights_sum = grad_weights_sum.contiguous()
        grad_ambient_sum = grad_ambient_sum.contiguous()
        grad_image = grad_image.contiguous()

        sigmas, rgbs, ambient, deltas, rays, weights_sum, ambient_sum, depth, image = ctx.saved_tensors
        M, N, T_thresh = ctx.dims
   
        grad_sigmas = torch.zeros_like(sigmas)
        grad_rgbs = torch.zeros_like(rgbs)
        grad_ambient = torch.zeros_like(ambient)

        _backend.composite_rays_train_backward(grad_weights_sum, grad_ambient_sum, grad_image, sigmas, rgbs, ambient, deltas, rays, weights_sum, ambient_sum, image, M, N, T_thresh, grad_sigmas, grad_rgbs, grad_ambient)

        return grad_sigmas, grad_rgbs, grad_ambient, None, None, None


composite_rays_train = _composite_rays_train.apply

# ----------------------------------------
# infer functions
# ----------------------------------------

class _march_rays(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, bound, density_bitfield, C, H, near, far, align=-1, perturb=False, dt_gamma=0, max_steps=1024):
        ''' march rays to generate points (forward only, for inference)
        Args:
            n_alive: int, number of alive rays
            n_step: int, how many steps we march
            rays_alive: int, [N], the alive rays' IDs in N (N >= n_alive, but we only use first n_alive)
            rays_t: float, [N], the alive rays' time, we only use the first n_alive.
            rays_o/d: float, [N, 3]
            bound: float, scalar
            density_bitfield: uint8: [CHHH // 8]
            C: int
            H: int
            nears/fars: float, [N]
            align: int, pad output so its size is dividable by align, set to -1 to disable.
            perturb: bool/int, int > 0 is used as the random seed.
            dt_gamma: float, called cone_angle in instant-ngp, exponentially accelerate ray marching if > 0. (very significant effect, but generally lead to worse performance)
            max_steps: int, max number of sampled points along each ray, also affect min_stepsize.
        Returns:
            xyzs: float, [n_alive * n_step, 3], all generated points' coords
            dirs: float, [n_alive * n_step, 3], all generated points' view dirs.
            deltas: float, [n_alive * n_step, 2], all generated points' deltas (here we record two deltas, the first is for RGB, the second for depth).
        '''
        
        if not rays_o.is_cuda: rays_o = rays_o.cuda()
        if not rays_d.is_cuda: rays_d = rays_d.cuda()
        
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        M = n_alive * n_step

        if align > 0:
            M += align - (M % align)
        
        xyzs = torch.zeros(M, 3, dtype=rays_o.dtype, device=rays_o.device)
        dirs = torch.zeros(M, 3, dtype=rays_o.dtype, device=rays_o.device)
        deltas = torch.zeros(M, 2, dtype=rays_o.dtype, device=rays_o.device) # 2 vals, one for rgb, one for depth

        if perturb:
            # torch.manual_seed(perturb) # test_gui uses spp index as seed
            noises = torch.rand(n_alive, dtype=rays_o.dtype, device=rays_o.device)
        else:
            noises = torch.zeros(n_alive, dtype=rays_o.dtype, device=rays_o.device)

        _backend.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, bound, dt_gamma, max_steps, C, H, density_bitfield, near, far, xyzs, dirs, deltas, noises)

        return xyzs, dirs, deltas

march_rays = _march_rays.apply


class _composite_rays(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32) # need to cast sigmas & rgbs to float
    def forward(ctx, n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas, weights_sum, depth, image, T_thresh=1e-2):
        ''' composite rays' rgbs, according to the ray marching formula. (for inference)
        Args:
            n_alive: int, number of alive rays
            n_step: int, how many steps we march
            rays_alive: int, [n_alive], the alive rays' IDs in N (N >= n_alive)
            rays_t: float, [N], the alive rays' time
            sigmas: float, [n_alive * n_step,]
            rgbs: float, [n_alive * n_step, 3]
            deltas: float, [n_alive * n_step, 2], all generated points' deltas (here we record two deltas, the first is for RGB, the second for depth).
        In-place Outputs:
            weights_sum: float, [N,], the alpha channel
            depth: float, [N,], the depth value
            image: float, [N, 3], the RGB channel (after multiplying alpha!)
        '''
        _backend.composite_rays(n_alive, n_step, T_thresh, rays_alive, rays_t, sigmas, rgbs, deltas, weights_sum, depth, image)
        return tuple()


composite_rays = _composite_rays.apply


class _composite_rays_ambient(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32) # need to cast sigmas & rgbs to float
    def forward(ctx, n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas, ambients, weights_sum, depth, image, ambient_sum, T_thresh=1e-2):
        _backend.composite_rays_ambient(n_alive, n_step, T_thresh, rays_alive, rays_t, sigmas, rgbs, deltas, ambients, weights_sum, depth, image, ambient_sum)
        return tuple()


composite_rays_ambient = _composite_rays_ambient.apply





# custom

class _composite_rays_train_sigma(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, sigmas, rgbs, ambient, deltas, rays, T_thresh=1e-4):
        ''' composite rays' rgbs, according to the ray marching formula.
        Args:
            rgbs: float, [M, 3]
            sigmas: float, [M,]
            ambient: float, [M,] (after summing up the last dimension)
            deltas: float, [M, 2]
            rays: int32, [N, 3]
        Returns:
            weights_sum: float, [N,], the alpha channel
            depth: float, [N, ], the Depth
            image: float, [N, 3], the RGB channel (after multiplying alpha!)
        '''
        
        sigmas = sigmas.contiguous()
        rgbs = rgbs.contiguous()
        ambient = ambient.contiguous()

        M = sigmas.shape[0]
        N = rays.shape[0]

        weights_sum = torch.empty(N, dtype=sigmas.dtype, device=sigmas.device)
        ambient_sum = torch.empty(N, dtype=sigmas.dtype, device=sigmas.device)
        depth = torch.empty(N, dtype=sigmas.dtype, device=sigmas.device)
        image = torch.empty(N, 3, dtype=sigmas.dtype, device=sigmas.device)

        _backend.composite_rays_train_sigma_forward(sigmas, rgbs, ambient, deltas, rays, M, N, T_thresh, weights_sum, ambient_sum, depth, image)

        ctx.save_for_backward(sigmas, rgbs, ambient, deltas, rays, weights_sum, ambient_sum, depth, image)
        ctx.dims = [M, N, T_thresh]

        return weights_sum, ambient_sum, depth, image
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_weights_sum, grad_ambient_sum, grad_depth, grad_image):

        # NOTE: grad_depth is not used now! It won't be propagated to sigmas.

        grad_weights_sum = grad_weights_sum.contiguous()
        grad_ambient_sum = grad_ambient_sum.contiguous()
        grad_image = grad_image.contiguous()

        sigmas, rgbs, ambient, deltas, rays, weights_sum, ambient_sum, depth, image = ctx.saved_tensors
        M, N, T_thresh = ctx.dims
   
        grad_sigmas = torch.zeros_like(sigmas)
        grad_rgbs = torch.zeros_like(rgbs)
        grad_ambient = torch.zeros_like(ambient)

        _backend.composite_rays_train_sigma_backward(grad_weights_sum, grad_ambient_sum, grad_image, sigmas, rgbs, ambient, deltas, rays, weights_sum, ambient_sum, image, M, N, T_thresh, grad_sigmas, grad_rgbs, grad_ambient)

        return grad_sigmas, grad_rgbs, grad_ambient, None, None, None


composite_rays_train_sigma = _composite_rays_train_sigma.apply


class _composite_rays_ambient_sigma(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32) # need to cast sigmas & rgbs to float
    def forward(ctx, n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas, ambients, weights_sum, depth, image, ambient_sum, T_thresh=1e-2):
        _backend.composite_rays_ambient_sigma(n_alive, n_step, T_thresh, rays_alive, rays_t, sigmas, rgbs, deltas, ambients, weights_sum, depth, image, ambient_sum)
        return tuple()


composite_rays_ambient_sigma = _composite_rays_ambient_sigma.apply



# uncertainty
class _composite_rays_train_uncertainty(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, sigmas, rgbs, ambient, uncertainty, deltas, rays, T_thresh=1e-4):
        ''' composite rays' rgbs, according to the ray marching formula.
        Args:
            rgbs: float, [M, 3]
            sigmas: float, [M,]
            ambient: float, [M,] (after summing up the last dimension)
            deltas: float, [M, 2]
            rays: int32, [N, 3]
        Returns:
            weights_sum: float, [N,], the alpha channel
            depth: float, [N, ], the Depth
            image: float, [N, 3], the RGB channel (after multiplying alpha!)
        '''
        
        sigmas = sigmas.contiguous()
        rgbs = rgbs.contiguous()
        ambient = ambient.contiguous()
        uncertainty = uncertainty.contiguous()

        M = sigmas.shape[0]
        N = rays.shape[0]

        weights_sum = torch.empty(N, dtype=sigmas.dtype, device=sigmas.device)
        ambient_sum = torch.empty(N, dtype=sigmas.dtype, device=sigmas.device)
        uncertainty_sum = torch.empty(N, dtype=sigmas.dtype, device=sigmas.device)
        depth = torch.empty(N, dtype=sigmas.dtype, device=sigmas.device)
        image = torch.empty(N, 3, dtype=sigmas.dtype, device=sigmas.device)

        _backend.composite_rays_train_uncertainty_forward(sigmas, rgbs, ambient, uncertainty, deltas, rays, M, N, T_thresh, weights_sum, ambient_sum, uncertainty_sum, depth, image)

        ctx.save_for_backward(sigmas, rgbs, ambient, uncertainty, deltas, rays, weights_sum, ambient_sum, uncertainty_sum, depth, image)
        ctx.dims = [M, N, T_thresh]

        return weights_sum, ambient_sum, uncertainty_sum, depth, image
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_weights_sum, grad_ambient_sum, grad_uncertainty_sum, grad_depth, grad_image):

        # NOTE: grad_depth is not used now! It won't be propagated to sigmas.

        grad_weights_sum = grad_weights_sum.contiguous()
        grad_ambient_sum = grad_ambient_sum.contiguous()
        grad_uncertainty_sum = grad_uncertainty_sum.contiguous()
        grad_image = grad_image.contiguous()

        sigmas, rgbs, ambient, uncertainty, deltas, rays, weights_sum, ambient_sum, uncertainty_sum, depth, image = ctx.saved_tensors
        M, N, T_thresh = ctx.dims
   
        grad_sigmas = torch.zeros_like(sigmas)
        grad_rgbs = torch.zeros_like(rgbs)
        grad_ambient = torch.zeros_like(ambient)
        grad_uncertainty = torch.zeros_like(uncertainty)

        _backend.composite_rays_train_uncertainty_backward(grad_weights_sum, grad_ambient_sum, grad_uncertainty_sum, grad_image, sigmas, rgbs, ambient, uncertainty, deltas, rays, weights_sum, ambient_sum, uncertainty_sum, image, M, N, T_thresh, grad_sigmas, grad_rgbs, grad_ambient, grad_uncertainty)

        return grad_sigmas, grad_rgbs, grad_ambient, grad_uncertainty, None, None, None


composite_rays_train_uncertainty = _composite_rays_train_uncertainty.apply


class _composite_rays_uncertainty(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32) # need to cast sigmas & rgbs to float
    def forward(ctx, n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas, ambients, uncertainties, weights_sum, depth, image, ambient_sum, uncertainty_sum, T_thresh=1e-2):
        _backend.composite_rays_uncertainty(n_alive, n_step, T_thresh, rays_alive, rays_t, sigmas, rgbs, deltas, ambients, uncertainties, weights_sum, depth, image, ambient_sum, uncertainty_sum)
        return tuple()


composite_rays_uncertainty = _composite_rays_uncertainty.apply



# triplane(eye)
class _composite_rays_train_triplane(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, sigmas, rgbs, amb_aud, amb_eye, uncertainty, deltas, rays, T_thresh=1e-4):
        ''' composite rays' rgbs, according to the ray marching formula.
        Args:
            rgbs: float, [M, 3]
            sigmas: float, [M,]
            ambient: float, [M,] (after summing up the last dimension)
            deltas: float, [M, 2]
            rays: int32, [N, 3]
        Returns:
            weights_sum: float, [N,], the alpha channel
            depth: float, [N, ], the Depth
            image: float, [N, 3], the RGB channel (after multiplying alpha!)
        '''
        
        sigmas = sigmas.contiguous()
        rgbs = rgbs.contiguous()
        amb_aud = amb_aud.contiguous()
        amb_eye = amb_eye.contiguous()
        uncertainty = uncertainty.contiguous()

        M = sigmas.shape[0]
        N = rays.shape[0]

        weights_sum = torch.empty(N, dtype=sigmas.dtype, device=sigmas.device)
        amb_aud_sum = torch.empty(N, dtype=sigmas.dtype, device=sigmas.device)
        amb_eye_sum = torch.empty(N, dtype=sigmas.dtype, device=sigmas.device)
        uncertainty_sum = torch.empty(N, dtype=sigmas.dtype, device=sigmas.device)
        depth = torch.empty(N, dtype=sigmas.dtype, device=sigmas.device)
        image = torch.empty(N, 3, dtype=sigmas.dtype, device=sigmas.device)

        _backend.composite_rays_train_triplane_forward(sigmas, rgbs, amb_aud, amb_eye, uncertainty, deltas, rays, M, N, T_thresh, weights_sum, amb_aud_sum, amb_eye_sum, uncertainty_sum, depth, image)

        ctx.save_for_backward(sigmas, rgbs, amb_aud, amb_eye, uncertainty, deltas, rays, weights_sum, amb_aud_sum, amb_eye_sum, uncertainty_sum, depth, image)
        ctx.dims = [M, N, T_thresh]

        return weights_sum, amb_aud_sum, amb_eye_sum, uncertainty_sum, depth, image
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_weights_sum, grad_amb_aud_sum, grad_amb_eye_sum, grad_uncertainty_sum, grad_depth, grad_image):

        # NOTE: grad_depth is not used now! It won't be propagated to sigmas.

        grad_weights_sum = grad_weights_sum.contiguous()
        grad_amb_aud_sum = grad_amb_aud_sum.contiguous()
        grad_amb_eye_sum = grad_amb_eye_sum.contiguous()
        grad_uncertainty_sum = grad_uncertainty_sum.contiguous()
        grad_image = grad_image.contiguous()

        sigmas, rgbs, amb_aud, amb_eye, uncertainty, deltas, rays, weights_sum, amb_aud_sum, amb_eye_sum, uncertainty_sum, depth, image = ctx.saved_tensors
        M, N, T_thresh = ctx.dims
   
        grad_sigmas = torch.zeros_like(sigmas)
        grad_rgbs = torch.zeros_like(rgbs)
        grad_amb_aud = torch.zeros_like(amb_aud)
        grad_amb_eye = torch.zeros_like(amb_eye)
        grad_uncertainty = torch.zeros_like(uncertainty)

        _backend.composite_rays_train_triplane_backward(grad_weights_sum, grad_amb_aud_sum, grad_amb_eye_sum, grad_uncertainty_sum, grad_image, sigmas, rgbs, amb_aud, amb_eye, uncertainty, deltas, rays, weights_sum, amb_aud_sum, amb_eye_sum, uncertainty_sum, image, M, N, T_thresh, grad_sigmas, grad_rgbs, grad_amb_aud, grad_amb_eye, grad_uncertainty)

        return grad_sigmas, grad_rgbs, grad_amb_aud, grad_amb_eye, grad_uncertainty, None, None, None


composite_rays_train_triplane = _composite_rays_train_triplane.apply


class _composite_rays_triplane(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32) # need to cast sigmas & rgbs to float
    def forward(ctx, n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas, ambs_aud, ambs_eye, uncertainties, weights_sum, depth, image, amb_aud_sum, amb_eye_sum, uncertainty_sum, T_thresh=1e-2):
        _backend.composite_rays_triplane(n_alive, n_step, T_thresh, rays_alive, rays_t, sigmas, rgbs, deltas, ambs_aud, ambs_eye, uncertainties, weights_sum, depth, image, amb_aud_sum, amb_eye_sum, uncertainty_sum)
        return tuple()


composite_rays_triplane = _composite_rays_triplane.apply