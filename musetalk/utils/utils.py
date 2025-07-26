import os
import cv2
import numpy as np
import torch
from typing import Union, List
import torch.nn.functional as F
from einops import rearrange
import shutil
import os.path as osp

from musetalk.models.vae import VAE
from musetalk.models.unet import UNet,PositionalEncoding


def load_all_model(
    unet_model_path=os.path.join("models", "musetalkV15", "unet.pth"),
    vae_type="sd-vae",
    unet_config=os.path.join("models", "musetalkV15", "musetalk.json"),
    device=None,
):
    vae = VAE(
        model_path = os.path.join("models", vae_type),
    )
    print(f"load unet model from {unet_model_path}")
    unet = UNet(
        unet_config=unet_config,
        model_path=unet_model_path,
        device=device
    )
    pe = PositionalEncoding(d_model=384)
    return vae, unet, pe

def get_file_type(video_path):
    _, ext = os.path.splitext(video_path)

    if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
        return 'image'
    elif ext.lower() in ['.avi', '.mp4', '.mov', '.flv', '.mkv']:
        return 'video'
    else:
        return 'unsupported'

def get_video_fps(video_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return fps

def datagen(
    whisper_chunks,
    vae_encode_latents,
    batch_size=8,
    delay_frame=0,
    device="cuda:0",
):
    whisper_batch, latent_batch = [], []
    for i, w in enumerate(whisper_chunks):
        idx = (i+delay_frame)%len(vae_encode_latents)
        latent = vae_encode_latents[idx]
        whisper_batch.append(w)
        latent_batch.append(latent)

        if len(latent_batch) >= batch_size:
            whisper_batch = torch.stack(whisper_batch)
            latent_batch = torch.cat(latent_batch, dim=0)
            yield whisper_batch, latent_batch
            whisper_batch, latent_batch  = [], []

    # the last batch may smaller than batch size
    if len(latent_batch) > 0:
        whisper_batch = torch.stack(whisper_batch)
        latent_batch = torch.cat(latent_batch, dim=0)

        yield whisper_batch.to(device), latent_batch.to(device)

def cast_training_params(
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    dtype=torch.float32,
):
    if not isinstance(model, list):
        model = [model]
    for m in model:
        for param in m.parameters():
            # only upcast trainable parameters into fp32
            if param.requires_grad:
                param.data = param.to(dtype)

def rand_log_normal(
    shape,
    loc=0.,
    scale=1.,
    device='cpu',
    dtype=torch.float32,
    generator=None
):
    """Draws samples from an lognormal distribution."""
    rnd_normal = torch.randn(
        shape, device=device, dtype=dtype, generator=generator)  # N(0, I)
    sigma = (rnd_normal * scale + loc).exp()
    return sigma

def get_mouth_region(frames, image_pred, pixel_values_face_mask):
    # Initialize lists to store the results for each image in the batch
    mouth_real_list = []
    mouth_generated_list = []

    # Process each image in the batch
    for b in range(frames.shape[0]):
        # Find the non-zero area in the face mask
        non_zero_indices = torch.nonzero(pixel_values_face_mask[b])
        # If there are no non-zero indices, skip this image
        if non_zero_indices.numel() == 0:
            continue

        min_y, max_y = torch.min(non_zero_indices[:, 1]), torch.max(
            non_zero_indices[:, 1])
        min_x, max_x = torch.min(non_zero_indices[:, 2]), torch.max(
            non_zero_indices[:, 2])

        # Crop the frames and image_pred according to the non-zero area
        frames_cropped = frames[b, :, min_y:max_y, min_x:max_x]
        image_pred_cropped = image_pred[b, :, min_y:max_y, min_x:max_x]
        # Resize the cropped images to 256*256
        frames_resized = F.interpolate(frames_cropped.unsqueeze(
            0), size=(256, 256), mode='bilinear', align_corners=False)
        image_pred_resized = F.interpolate(image_pred_cropped.unsqueeze(
            0), size=(256, 256), mode='bilinear', align_corners=False)

        # Append the resized images to the result lists
        mouth_real_list.append(frames_resized)
        mouth_generated_list.append(image_pred_resized)

    # Convert the lists to tensors if they are not empty
    mouth_real = torch.cat(mouth_real_list, dim=0) if mouth_real_list else None
    mouth_generated = torch.cat(
        mouth_generated_list, dim=0) if mouth_generated_list else None

    return mouth_real, mouth_generated

def get_image_pred(pixel_values,
                   ref_pixel_values,
                   audio_prompts,
                   vae,
                   net,
                   weight_dtype):
    with torch.no_grad():
        bsz, num_frames, c, h, w = pixel_values.shape

        masked_pixel_values = pixel_values.clone()
        masked_pixel_values[:, :, :, h//2:, :] = -1

        masked_frames = rearrange(
            masked_pixel_values, 'b f c h w -> (b f) c h w')
        masked_latents = vae.encode(masked_frames).latent_dist.mode()
        masked_latents = masked_latents * vae.config.scaling_factor
        masked_latents = masked_latents.float()

        ref_frames = rearrange(ref_pixel_values, 'b f c h w-> (b f) c h w')
        ref_latents = vae.encode(ref_frames).latent_dist.mode()
        ref_latents = ref_latents * vae.config.scaling_factor
        ref_latents = ref_latents.float()

        input_latents = torch.cat([masked_latents, ref_latents], dim=1)
        input_latents = input_latents.to(weight_dtype)
        timesteps = torch.tensor([0], device=input_latents.device)
        latents_pred = net(
            input_latents,
            timesteps,
            audio_prompts,
        )
        latents_pred = (1 / vae.config.scaling_factor) * latents_pred
        image_pred = vae.decode(latents_pred).sample
        image_pred = image_pred.float()

    return image_pred

def process_audio_features(cfg, batch, wav2vec, bsz, num_frames, weight_dtype):
    with torch.no_grad():
        audio_feature_length_per_frame = 2 * \
            (cfg.data.audio_padding_length_left +
             cfg.data.audio_padding_length_right + 1)
        audio_feats = batch['audio_feature'].to(weight_dtype)
        audio_feats = wav2vec.encoder(
            audio_feats, output_hidden_states=True).hidden_states
        audio_feats = torch.stack(audio_feats, dim=2).to(weight_dtype)  # [B, T, 10, 5, 384]

        start_ts = batch['audio_offset']
        step_ts = batch['audio_step']
        audio_feats = torch.cat([torch.zeros_like(audio_feats[:, :2*cfg.data.audio_padding_length_left]),
                                audio_feats,
                                torch.zeros_like(audio_feats[:, :2*cfg.data.audio_padding_length_right])], 1)
        audio_prompts = []
        for bb in range(bsz):
            audio_feats_list = []
            for f in range(num_frames):
                cur_t = (start_ts[bb] + f * step_ts[bb]) * 2
                audio_clip = audio_feats[bb:bb+1,
                                         cur_t: cur_t+audio_feature_length_per_frame]

                audio_feats_list.append(audio_clip)
            audio_feats_list = torch.stack(audio_feats_list, 1)
            audio_prompts.append(audio_feats_list)
        audio_prompts = torch.cat(audio_prompts)  # B, T, 10, 5, 384
    return audio_prompts

def save_checkpoint(model, save_dir, ckpt_num, name="appearance_net", total_limit=None, logger=None):
    save_path = os.path.join(save_dir, f"{name}-{ckpt_num}.pth")

    if total_limit is not None:
        checkpoints = os.listdir(save_dir)
        checkpoints = [d for d in checkpoints if d.endswith(".pth")]
        checkpoints = [d for d in checkpoints if name in d]
        checkpoints = sorted(
            checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0])
        )

        if len(checkpoints) >= total_limit:
            num_to_remove = len(checkpoints) - total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(
                f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(
                    save_dir, removing_checkpoint)
                os.remove(removing_checkpoint)

    state_dict = model.state_dict()
    torch.save(state_dict, save_path)

def save_models(accelerator, net, save_dir, global_step, cfg, logger=None):
    unwarp_net = accelerator.unwrap_model(net)
    save_checkpoint(
        unwarp_net.unet,
        save_dir,
        global_step,
        name="unet",
        total_limit=cfg.total_limit,
        logger=logger
    )

def delete_additional_ckpt(base_path, num_keep):
    dirs = []
    for d in os.listdir(base_path):
        if d.startswith("checkpoint-"):
            dirs.append(d)
    num_tot = len(dirs)
    if num_tot <= num_keep:
        return
    # ensure ckpt is sorted and delete the ealier!
    del_dirs = sorted(dirs, key=lambda x: int(x.split("-")[-1]))[: num_tot - num_keep]
    for d in del_dirs:
        path_to_dir = osp.join(base_path, d)
        if osp.exists(path_to_dir):
            shutil.rmtree(path_to_dir)

def seed_everything(seed):
    import random

    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)

def process_and_save_images(
    batch, 
    image_pred,
    image_pred_infer,
    save_dir,
    global_step,
    accelerator,
    num_images_to_keep=10,
    syncnet_score=1
):
    # Rearrange the tensors
    print("image_pred.shape: ", image_pred.shape)
    pixel_values_ref_img = rearrange(batch['pixel_values_ref_img'], "b f c h w -> (b f) c h w")
    pixel_values = rearrange(batch["pixel_values_vid"], 'b f c h w -> (b f) c h w')
    
    # Create masked pixel values
    masked_pixel_values = batch["pixel_values_vid"].clone()
    _, _, _, h, _ = batch["pixel_values_vid"].shape
    masked_pixel_values[:, :, :, h//2:, :] = -1
    masked_pixel_values = rearrange(masked_pixel_values, 'b f c h w -> (b f) c h w')
    
    # Keep only the specified number of images
    pixel_values = pixel_values[:num_images_to_keep, :, :, :]
    masked_pixel_values = masked_pixel_values[:num_images_to_keep, :, :, :]
    pixel_values_ref_img = pixel_values_ref_img[:num_images_to_keep, :, :, :]
    image_pred = image_pred.detach()[:num_images_to_keep, :, :, :]
    image_pred_infer = image_pred_infer.detach()[:num_images_to_keep, :, :, :]
    
    # Concatenate images
    concat = torch.cat([
        masked_pixel_values * 0.5 + 0.5, 
        pixel_values_ref_img * 0.5 + 0.5,
        image_pred * 0.5 + 0.5,
        pixel_values * 0.5 + 0.5,
        image_pred_infer * 0.5 + 0.5,
    ], dim=2)
    print("concat.shape: ", concat.shape)
    
    # Create the save directory if it doesn't exist
    os.makedirs(f'{save_dir}/samples/', exist_ok=True)

    # Try to save the concatenated image
    try:
        # Concatenate images horizontally and convert to numpy array
        final_image = torch.cat([concat[i] for i in range(concat.shape[0])], dim=-1).permute(1, 2, 0).cpu().numpy()[:, :, [2, 1, 0]] * 255
        # Save the image
        cv2.imwrite(f'{save_dir}/samples/sample_{global_step}_{accelerator.device}_SyncNetScore_{syncnet_score}.jpg', final_image)
        print(f"Image saved successfully: {save_dir}/samples/sample_{global_step}_{accelerator.device}_SyncNetScore_{syncnet_score}.jpg")
    except Exception as e:
        print(f"Failed to save image: {e}")