import os
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import WhisperModel
from diffusers.optimization import get_scheduler
from omegaconf import OmegaConf
from einops import rearrange

from musetalk.models.syncnet import SyncNet
from musetalk.loss.discriminator import MultiScaleDiscriminator, DiscriminatorFullModel
from musetalk.loss.basic_loss import Interpolate
import musetalk.loss.vgg_face as vgg_face
from musetalk.data.dataset import PortraitDataset
from musetalk.utils.utils import (
    get_image_pred,
    process_audio_features,
    process_and_save_images
)

class Net(nn.Module):
    def __init__(
        self,
        unet: UNet2DConditionModel,
    ):
        super().__init__()
        self.unet = unet

    def forward(
        self,
        input_latents,
        timesteps,
        audio_prompts,
    ):
        model_pred = self.unet(
            input_latents,
            timesteps,
            encoder_hidden_states=audio_prompts
        ).sample
        return model_pred

logger = logging.getLogger(__name__)

def initialize_models_and_optimizers(cfg, accelerator, weight_dtype):
    """Initialize models and optimizers"""
    model_dict = {
        'vae': None,
        'unet': None,
        'net': None,
        'wav2vec': None,
        'optimizer': None,
        'lr_scheduler': None,
        'scheduler_max_steps': None,
        'trainable_params': None
    }
    
    model_dict['vae'] = AutoencoderKL.from_pretrained(
        cfg.pretrained_model_name_or_path,
        subfolder=cfg.vae_type,
    )

    unet_config_file = os.path.join(
        cfg.pretrained_model_name_or_path, 
        cfg.unet_sub_folder + "/musetalk.json"
    )
    
    with open(unet_config_file, 'r') as f:
        unet_config = json.load(f)
    model_dict['unet'] = UNet2DConditionModel(**unet_config)
    
    if not cfg.random_init_unet:
        pretrained_unet_path = os.path.join(cfg.pretrained_model_name_or_path, cfg.unet_sub_folder, "pytorch_model.bin")
        print(f"### Loading existing unet weights from {pretrained_unet_path}. ###")
        checkpoint = torch.load(pretrained_unet_path, map_location=accelerator.device)
        model_dict['unet'].load_state_dict(checkpoint)
      
    unet_params = [p.numel() for n, p in model_dict['unet'].named_parameters()]
    logger.info(f"unet {sum(unet_params) / 1e6}M-parameter")
    
    model_dict['vae'].requires_grad_(False)
    model_dict['unet'].requires_grad_(True)

    model_dict['vae'].to(accelerator.device, dtype=weight_dtype)

    model_dict['net'] = Net(model_dict['unet'])

    model_dict['wav2vec'] = WhisperModel.from_pretrained(cfg.whisper_path).to(
        device="cuda", dtype=weight_dtype).eval()
    model_dict['wav2vec'].requires_grad_(False)

    if cfg.solver.gradient_checkpointing:
        model_dict['unet'].enable_gradient_checkpointing()

    if cfg.solver.scale_lr:
        learning_rate = (
            cfg.solver.learning_rate
            * cfg.solver.gradient_accumulation_steps
            * cfg.data.train_bs
            * accelerator.num_processes
        )
    else:
        learning_rate = cfg.solver.learning_rate

    if cfg.solver.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    model_dict['trainable_params'] = list(filter(lambda p: p.requires_grad, model_dict['net'].parameters()))
    if accelerator.is_main_process:
        print('trainable params')
        for n, p in model_dict['net'].named_parameters():
            if p.requires_grad:
                print(n)

    model_dict['optimizer'] = optimizer_cls(
        model_dict['trainable_params'],
        lr=learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )

    model_dict['scheduler_max_steps'] = cfg.solver.max_train_steps * cfg.solver.gradient_accumulation_steps
    model_dict['lr_scheduler'] = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=model_dict['optimizer'],
        num_warmup_steps=cfg.solver.lr_warmup_steps * cfg.solver.gradient_accumulation_steps,
        num_training_steps=model_dict['scheduler_max_steps'],
    )

    return model_dict

def initialize_dataloaders(cfg):
    """Initialize training and validation dataloaders"""
    dataloader_dict = {
        'train_dataset': None,
        'val_dataset': None,
        'train_dataloader': None,
        'val_dataloader': None
    }
    
    dataloader_dict['train_dataset'] = PortraitDataset(cfg={
        'image_size': cfg.data.image_size,
        'T': cfg.data.n_sample_frames,
        "sample_method": cfg.data.sample_method,
        'top_k_ratio': cfg.data.top_k_ratio,
        "contorl_face_min_size": cfg.data.contorl_face_min_size,
        "dataset_key": cfg.data.dataset_key,
        "padding_pixel_mouth": cfg.padding_pixel_mouth,
        "whisper_path": cfg.whisper_path,
        "min_face_size": cfg.data.min_face_size,
        "cropping_jaw2edge_margin_mean": cfg.cropping_jaw2edge_margin_mean,
        "cropping_jaw2edge_margin_std": cfg.cropping_jaw2edge_margin_std,
        "crop_type": cfg.crop_type,
        "random_margin_method": cfg.random_margin_method,
    })

    dataloader_dict['train_dataloader'] = torch.utils.data.DataLoader(
        dataloader_dict['train_dataset'],
        batch_size=cfg.data.train_bs,
        shuffle=True,
        num_workers=cfg.data.num_workers,
    )
    
    dataloader_dict['val_dataset'] = PortraitDataset(cfg={
        'image_size': cfg.data.image_size,
        'T': cfg.data.n_sample_frames,
        "sample_method": cfg.data.sample_method,
        'top_k_ratio': cfg.data.top_k_ratio,
        "contorl_face_min_size": cfg.data.contorl_face_min_size,
        "dataset_key": cfg.data.dataset_key,
        "padding_pixel_mouth": cfg.padding_pixel_mouth,
        "whisper_path": cfg.whisper_path,
        "min_face_size": cfg.data.min_face_size,
        "cropping_jaw2edge_margin_mean": cfg.cropping_jaw2edge_margin_mean,
        "cropping_jaw2edge_margin_std": cfg.cropping_jaw2edge_margin_std,
        "crop_type": cfg.crop_type,
        "random_margin_method": cfg.random_margin_method,
    })

    dataloader_dict['val_dataloader'] = torch.utils.data.DataLoader(
        dataloader_dict['val_dataset'],
        batch_size=cfg.data.train_bs,
        shuffle=True,
        num_workers=1,
    )
    
    return dataloader_dict

def initialize_loss_functions(cfg, accelerator, scheduler_max_steps):
    """Initialize loss functions and discriminators"""
    loss_dict = {
        'L1_loss': nn.L1Loss(reduction='mean'),
        'discriminator': None,
        'mouth_discriminator': None,
        'optimizer_D': None,
        'mouth_optimizer_D': None,
        'scheduler_D': None,
        'mouth_scheduler_D': None,
        'disc_scales': None,
        'discriminator_full': None,
        'mouth_discriminator_full': None
    }
    
    if cfg.loss_params.gan_loss > 0:
        loss_dict['discriminator'] = MultiScaleDiscriminator(
            **cfg.model_params.discriminator_params).to(accelerator.device)
        loss_dict['discriminator_full'] = DiscriminatorFullModel(loss_dict['discriminator'])
        loss_dict['disc_scales'] = cfg.model_params.discriminator_params.scales
        loss_dict['optimizer_D'] = optim.AdamW(
            loss_dict['discriminator'].parameters(),
            lr=cfg.discriminator_train_params.lr,
            weight_decay=cfg.discriminator_train_params.weight_decay,
            betas=cfg.discriminator_train_params.betas,
            eps=cfg.discriminator_train_params.eps)
        loss_dict['scheduler_D'] = CosineAnnealingLR(
            loss_dict['optimizer_D'],
            T_max=scheduler_max_steps,
            eta_min=1e-6
        )

    if cfg.loss_params.mouth_gan_loss > 0:
        loss_dict['mouth_discriminator'] = MultiScaleDiscriminator(
            **cfg.model_params.discriminator_params).to(accelerator.device)
        loss_dict['mouth_discriminator_full'] = DiscriminatorFullModel(loss_dict['mouth_discriminator'])
        loss_dict['mouth_optimizer_D'] = optim.AdamW(
            loss_dict['mouth_discriminator'].parameters(),
            lr=cfg.discriminator_train_params.lr,
            weight_decay=cfg.discriminator_train_params.weight_decay,
            betas=cfg.discriminator_train_params.betas,
            eps=cfg.discriminator_train_params.eps)
        loss_dict['mouth_scheduler_D'] = CosineAnnealingLR(
            loss_dict['mouth_optimizer_D'],
            T_max=scheduler_max_steps,
            eta_min=1e-6
        )
        
    return loss_dict

def initialize_syncnet(cfg, accelerator, weight_dtype):
    """Initialize SyncNet model"""
    if cfg.loss_params.sync_loss > 0 or cfg.use_adapted_weight:
        if cfg.data.n_sample_frames != 16:
            raise ValueError(
                f"Invalid n_sample_frames {cfg.data.n_sample_frames} for sync_loss, it should be 16."
            )
        syncnet_config = OmegaConf.load(cfg.syncnet_config_path)
        syncnet = SyncNet(OmegaConf.to_container(
            syncnet_config.model)).to(accelerator.device)
        print(
            f"Load SyncNet checkpoint from: {syncnet_config.ckpt.inference_ckpt_path}")
        checkpoint = torch.load(
            syncnet_config.ckpt.inference_ckpt_path, map_location=accelerator.device)
        syncnet.load_state_dict(checkpoint["state_dict"])
        syncnet.to(dtype=weight_dtype)
        syncnet.requires_grad_(False)
        syncnet.eval()
        return syncnet
    return None

def initialize_vgg(cfg, accelerator):
    """Initialize VGG model"""
    if cfg.loss_params.vgg_loss > 0:
        vgg_IN = vgg_face.Vgg19().to(accelerator.device,)
        pyramid = vgg_face.ImagePyramide(
            cfg.loss_params.pyramid_scale, 3).to(accelerator.device)
        vgg_IN.eval()
        downsampler = Interpolate(
            size=(224, 224), mode='bilinear', align_corners=False).to(accelerator.device)
        return vgg_IN, pyramid, downsampler
    return None, None, None

def validation(
    cfg,
    val_dataloader,
    net,
    vae,
    wav2vec,
    accelerator,
    save_dir,
    global_step,
    weight_dtype,
    syncnet_score=1,
):
    """Validation function for model evaluation"""
    net.eval()  # Set the model to evaluation mode
    for batch in val_dataloader:
        # The same ref_latents
        ref_pixel_values = batch["pixel_values_ref_img"].to(weight_dtype).to(
            accelerator.device, non_blocking=True
        )
        pixel_values = batch["pixel_values_vid"].to(weight_dtype).to(
            accelerator.device, non_blocking=True
        )
        bsz, num_frames, c, h, w = ref_pixel_values.shape

        audio_prompts = process_audio_features(cfg, batch, wav2vec, bsz, num_frames, weight_dtype)
        # audio feature for unet
        audio_prompts = rearrange(
            audio_prompts, 
            'b f c h w-> (b f) c h w'
        )
        audio_prompts = rearrange(
            audio_prompts, 
            '(b f) c h w -> (b f) (c h) w', 
            b=bsz
        )
        # different masked_latents
        image_pred_train = get_image_pred(
            pixel_values, ref_pixel_values, audio_prompts, vae, net, weight_dtype)
        image_pred_infer = get_image_pred(
            ref_pixel_values, ref_pixel_values, audio_prompts, vae, net, weight_dtype)

        process_and_save_images(
            batch,
            image_pred_train,
            image_pred_infer,
            save_dir,
            global_step,
            accelerator,
            cfg.num_images_to_keep,
            syncnet_score
        )
        # only infer 1 image in validation
        break
    net.train()  # Set the model back to training mode
