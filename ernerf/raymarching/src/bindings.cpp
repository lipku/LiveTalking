#include <torch/extension.h>

#include "raymarching.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // utils
    m.def("packbits", &packbits, "packbits (CUDA)");
    m.def("near_far_from_aabb", &near_far_from_aabb, "near_far_from_aabb (CUDA)");
    m.def("sph_from_ray", &sph_from_ray, "sph_from_ray (CUDA)");
    m.def("morton3D", &morton3D, "morton3D (CUDA)");
    m.def("morton3D_invert", &morton3D_invert, "morton3D_invert (CUDA)");
    m.def("morton3D_dilation", &morton3D_dilation, "morton3D_dilation (CUDA)");
    // train
    m.def("march_rays_train", &march_rays_train, "march_rays_train (CUDA)");
    m.def("march_rays_train_backward", &march_rays_train_backward, "march_rays_train_backward (CUDA)");
    m.def("composite_rays_train_forward", &composite_rays_train_forward, "composite_rays_train_forward (CUDA)");
    m.def("composite_rays_train_backward", &composite_rays_train_backward, "composite_rays_train_backward (CUDA)");
    // infer
    m.def("march_rays", &march_rays, "march rays (CUDA)");
    m.def("composite_rays", &composite_rays, "composite rays (CUDA)");
    m.def("composite_rays_ambient", &composite_rays_ambient, "composite rays with ambient (CUDA)");

    // train
    m.def("composite_rays_train_sigma_forward", &composite_rays_train_sigma_forward, "composite_rays_train_forward (CUDA)");
    m.def("composite_rays_train_sigma_backward", &composite_rays_train_sigma_backward, "composite_rays_train_backward (CUDA)");
    // infer
    m.def("composite_rays_ambient_sigma", &composite_rays_ambient_sigma, "composite rays with ambient (CUDA)");

    // uncertainty train
    m.def("composite_rays_train_uncertainty_forward", &composite_rays_train_uncertainty_forward, "composite_rays_train_forward (CUDA)");
    m.def("composite_rays_train_uncertainty_backward", &composite_rays_train_uncertainty_backward, "composite_rays_train_backward (CUDA)");
    m.def("composite_rays_uncertainty", &composite_rays_uncertainty, "composite rays with ambient (CUDA)");

    // triplane
    m.def("composite_rays_train_triplane_forward", &composite_rays_train_triplane_forward, "composite_rays_train_forward (CUDA)");
    m.def("composite_rays_train_triplane_backward", &composite_rays_train_triplane_backward, "composite_rays_train_backward (CUDA)");
    m.def("composite_rays_triplane", &composite_rays_triplane, "composite rays with ambient (CUDA)");

}