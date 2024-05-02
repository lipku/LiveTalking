#pragma once

#include <stdint.h>
#include <torch/torch.h>


void near_far_from_aabb(const at::Tensor rays_o, const at::Tensor rays_d, const at::Tensor aabb, const uint32_t N, const float min_near, at::Tensor nears, at::Tensor fars);
void sph_from_ray(const at::Tensor rays_o, const at::Tensor rays_d, const float radius, const uint32_t N, at::Tensor coords);
void morton3D(const at::Tensor coords, const uint32_t N, at::Tensor indices);
void morton3D_invert(const at::Tensor indices, const uint32_t N, at::Tensor coords);
void packbits(const at::Tensor grid, const uint32_t N, const float density_thresh, at::Tensor bitfield);
void morton3D_dilation(const at::Tensor grid, const uint32_t C, const uint32_t H, at::Tensor grid_dilation);

void march_rays_train(const at::Tensor rays_o, const at::Tensor rays_d, const at::Tensor grid, const float bound, const float dt_gamma, const uint32_t max_steps, const uint32_t N, const uint32_t C, const uint32_t H, const uint32_t M, const at::Tensor nears, const at::Tensor fars, at::Tensor xyzs, at::Tensor dirs, at::Tensor deltas, at::Tensor rays, at::Tensor counter, at::Tensor noises);
void march_rays_train_backward(const at::Tensor grad_xyzs, const at::Tensor grad_dirs, const at::Tensor rays, const at::Tensor deltas, const uint32_t N, const uint32_t M, at::Tensor grad_rays_o, at::Tensor grad_rays_d);
void composite_rays_train_forward(const at::Tensor sigmas, const at::Tensor rgbs, const at::Tensor ambient, const at::Tensor deltas, const at::Tensor rays, const uint32_t M, const uint32_t N, const float T_thresh, at::Tensor weights_sum, at::Tensor ambient_sum, at::Tensor depth, at::Tensor image);
void composite_rays_train_backward(const at::Tensor grad_weights_sum, const at::Tensor grad_ambient_sum, const at::Tensor grad_image, const at::Tensor sigmas, const at::Tensor rgbs, const at::Tensor ambient, const at::Tensor deltas, const at::Tensor rays, const at::Tensor weights_sum, const at::Tensor ambient_sum, const at::Tensor image, const uint32_t M, const uint32_t N, const float T_thresh, at::Tensor grad_sigmas, at::Tensor grad_rgbs, at::Tensor grad_ambient);

void march_rays(const uint32_t n_alive, const uint32_t n_step, const at::Tensor rays_alive, const at::Tensor rays_t, const at::Tensor rays_o, const at::Tensor rays_d, const float bound, const float dt_gamma, const uint32_t max_steps, const uint32_t C, const uint32_t H, const at::Tensor grid, const at::Tensor nears, const at::Tensor fars, at::Tensor xyzs, at::Tensor dirs, at::Tensor deltas, at::Tensor noises);
void composite_rays(const uint32_t n_alive, const uint32_t n_step, const float T_thresh, at::Tensor rays_alive, at::Tensor rays_t, at::Tensor sigmas, at::Tensor rgbs, at::Tensor deltas, at::Tensor weights_sum, at::Tensor depth, at::Tensor image);
void composite_rays_ambient(const uint32_t n_alive, const uint32_t n_step, const float T_thresh, at::Tensor rays_alive, at::Tensor rays_t, at::Tensor sigmas, at::Tensor rgbs, at::Tensor deltas, at::Tensor ambients, at::Tensor weights, at::Tensor depth, at::Tensor image, at::Tensor ambient_sum);


void composite_rays_train_sigma_forward(const at::Tensor sigmas, const at::Tensor rgbs, const at::Tensor ambient, const at::Tensor deltas, const at::Tensor rays, const uint32_t M, const uint32_t N, const float T_thresh, at::Tensor weights_sum, at::Tensor ambient_sum, at::Tensor depth, at::Tensor image);
void composite_rays_train_sigma_backward(const at::Tensor grad_weights_sum, const at::Tensor grad_ambient_sum, const at::Tensor grad_image, const at::Tensor sigmas, const at::Tensor rgbs, const at::Tensor ambient, const at::Tensor deltas, const at::Tensor rays, const at::Tensor weights_sum, const at::Tensor ambient_sum, const at::Tensor image, const uint32_t M, const uint32_t N, const float T_thresh, at::Tensor grad_sigmas, at::Tensor grad_rgbs, at::Tensor grad_ambient);

void composite_rays_ambient_sigma(const uint32_t n_alive, const uint32_t n_step, const float T_thresh, at::Tensor rays_alive, at::Tensor rays_t, at::Tensor sigmas, at::Tensor rgbs, at::Tensor deltas, at::Tensor ambients, at::Tensor weights, at::Tensor depth, at::Tensor image, at::Tensor ambient_sum);


// uncertainty
void composite_rays_train_uncertainty_forward(const at::Tensor sigmas, const at::Tensor rgbs, const at::Tensor ambient, const at::Tensor uncertainty, const at::Tensor deltas, const at::Tensor rays, const uint32_t M, const uint32_t N, const float T_thresh, at::Tensor weights_sum, at::Tensor ambient_sum, at::Tensor uncertainty_sum, at::Tensor depth, at::Tensor image);
void composite_rays_train_uncertainty_backward(const at::Tensor grad_weights_sum, const at::Tensor grad_ambient_sum, const at::Tensor grad_uncertainty_sum, const at::Tensor grad_image, const at::Tensor sigmas, const at::Tensor rgbs, const at::Tensor ambient, const at::Tensor uncertainty, const at::Tensor deltas, const at::Tensor rays, const at::Tensor weights_sum, const at::Tensor ambient_sum, const at::Tensor uncertainty_sum, const at::Tensor image, const uint32_t M, const uint32_t N, const float T_thresh, at::Tensor grad_sigmas, at::Tensor grad_rgbs, at::Tensor grad_ambient, at::Tensor grad_uncertainty);
void composite_rays_uncertainty(const uint32_t n_alive, const uint32_t n_step, const float T_thresh, at::Tensor rays_alive, at::Tensor rays_t, at::Tensor sigmas, at::Tensor rgbs, at::Tensor deltas, at::Tensor ambients, at::Tensor uncertainties, at::Tensor weights, at::Tensor depth, at::Tensor image, at::Tensor ambient_sum, at::Tensor uncertainty_sum);

// triplane
void composite_rays_train_triplane_forward(const at::Tensor sigmas, const at::Tensor rgbs, const at::Tensor amb_aud, const at::Tensor amb_eye, const at::Tensor uncertainty, const at::Tensor deltas, const at::Tensor rays, const uint32_t M, const uint32_t N, const float T_thresh, at::Tensor weights_sum, at::Tensor amb_aud_sum, at::Tensor amb_eye_sum, at::Tensor uncertainty_sum, at::Tensor depth, at::Tensor image);
void composite_rays_train_triplane_backward(const at::Tensor grad_weights_sum, const at::Tensor grad_amb_aud_sum, const at::Tensor grad_amb_eye_sum, const at::Tensor grad_uncertainty_sum, const at::Tensor grad_image, const at::Tensor sigmas, const at::Tensor rgbs, const at::Tensor amb_aud, const at::Tensor amb_eye, const at::Tensor uncertainty, const at::Tensor deltas, const at::Tensor rays, const at::Tensor weights_sum, const at::Tensor amb_aud_sum, const at::Tensor amb_eye_sum, const at::Tensor uncertainty_sum, const at::Tensor image, const uint32_t M, const uint32_t N, const float T_thresh, at::Tensor grad_sigmas, at::Tensor grad_rgbs, at::Tensor grad_amb_aud, at::Tensor grad_amb_eye, at::Tensor grad_uncertainty);
void composite_rays_triplane(const uint32_t n_alive, const uint32_t n_step, const float T_thresh, at::Tensor rays_alive, at::Tensor rays_t, at::Tensor sigmas, at::Tensor rgbs, at::Tensor deltas, at::Tensor ambs_aud, at::Tensor ambs_eye, at::Tensor uncertainties, at::Tensor weights, at::Tensor depth, at::Tensor image, at::Tensor amb_aud_sum, at::Tensor amb_eye_sum, at::Tensor uncertainty_sum);