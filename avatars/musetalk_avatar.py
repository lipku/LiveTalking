###############################################################################
#  Copyright (C) 2025 四川昱扬科技有限公司
#
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################
#
#  MuseTalk 数字人 — 迁移自 musereal.py + museasr.py
#

import math
import torch
import numpy as np

import subprocess
import os
import time
import torch.nn.functional as F
import cv2
import glob
import pickle
import copy

import queue
from queue import Queue
from threading import Thread, Event
import torch.multiprocessing as mp

from avatars.musetalk.utils.utils import get_file_type,get_video_fps,datagen
from avatars.musetalk.myutil import get_image_blending
from avatars.musetalk.utils.utils import load_all_model
from avatars.musetalk.whisper.audio2feature import Audio2Feature

from avatars.audio_features.whisper import WhisperASR
import asyncio
from av import AudioFrame, VideoFrame
from avatars.base_avatar import BaseAvatar

from tqdm import tqdm
from utils.logger import logger
from utils.image import read_imgs, mirror_index
from utils.device import initialize_device
from registry import register

device = initialize_device()
logger.info('Using {} for inference.'.format(device))

def load_model():
    # load model weights
    vae, unet, pe = load_all_model()
    #device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"))
    timesteps = torch.tensor([0], device=device)
    pe = pe.half().to(device)
    vae.vae = vae.vae.half().to(device)
    unet.model = unet.model.half().to(device)
    # Initialize audio processor and Whisper model
    audio_processor = Audio2Feature(model_path="./models/musetalk/whisper")
    return vae, unet, pe, timesteps, audio_processor


class ONNXMuseTalkWrapper:
    """ONNX 模型包装器，用于 MuseTalk UNet 推理"""
    def __init__(self, onnx_path):
        import onnxruntime as ort
        self.onnx_path = onnx_path
        self.device = device  # 添加 device 属性，兼容 PyTorch 模型接口
        self.dtype = torch.float16  # 使用 float16（与原始 PyTorch 一致）
        self.model = self  # 添加 model 属性，指向自身，兼容 unet.model 调用方式
        logger.info(f"Loading MuseTalk ONNX model from: {onnx_path}")
        
        # 配置 ONNX Runtime — 自动适配 NVIDIA CUDA / 沐曦 MACA / CPU
        from utils.device import get_onnx_providers
        providers = get_onnx_providers()
        logger.info(f"ONNX Runtime available providers: {ort.get_available_providers()}")
        logger.info(f"Trying to use providers: {providers}")
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        
        # 获取实际使用的 provider
        actual_providers = self.session.get_providers()
        logger.info(f"MuseTalk ONNX model actually using providers: {actual_providers}")
        
        # 获取输入输出信息
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        logger.info(f"MuseTalk ONNX model inputs: {self.input_names}")
        logger.info(f"MuseTalk ONNX model outputs: {self.output_names}")
    
    def eval(self):
        """兼容接口，ONNX 模型不需要 eval 模式"""
        return self
    
    def to(self, device):
        """兼容接口，ONNX 模型在 session 创建时已确定设备"""
        return self
    
    def __call__(self, latent_batch, timesteps, encoder_hidden_states):
        """推理接口，兼容 UNet2DConditionModel 调用方式"""
        import numpy as np
        import time
        
        t0 = time.time()
        
        # 将 torch.Tensor 转换为 numpy，并确保为 float16（与原始 PyTorch 一致）
        if isinstance(latent_batch, torch.Tensor):
            latent_batch = latent_batch.cpu().half().numpy()
        if isinstance(timesteps, torch.Tensor):
            timesteps = timesteps.cpu().numpy()
        if isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = encoder_hidden_states.cpu().half().numpy()
        
        # 确保 timestep 是 int64 类型
        timesteps = timesteps.astype(np.int64)
        
        t1 = time.time()
        
        # 运行推理
        outputs = self.session.run(
            None,
            {
                'latents': latent_batch,
                'timestep': timesteps,
                'audio_embedding': encoder_hidden_states
            }
        )
        
        t2 = time.time()
        
        # 将输出转换回 torch.Tensor（保持 float16）
        output_tensor = torch.from_numpy(outputs[0]).to(device=device, dtype=torch.float16)
        
        # MetaX ONNX 兼容: 对 latent 值做裁剪，防止极端值导致 VAE 解码退化（画面模糊）
        # 正常 SD VAE latent 范围约 [-3, 3]，超出 ±3.5 会触发 VAE clamp(0,1) 截断像素细节
        output_tensor = torch.clamp(output_tensor, -3.5, 3.5)
        
        t3 = time.time()
        
        # 每10次推理打印一次性能分析
        if not hasattr(self, '_call_count'):
            self._call_count = 0
        self._call_count += 1
        if self._call_count % 10 == 0:
            logger.info(f"[ONNX Perf] prep: {(t1-t0)*1000:.1f}ms, "
                       f"inference: {(t2-t1)*1000:.1f}ms, "
                       f"post: {(t3-t2)*1000:.1f}ms")
        
        # 包装为与 diffusers 模型输出兼容的格式
        class ModelOutput:
            def __init__(self, sample):
                self.sample = sample
        
        return ModelOutput(output_tensor)


class ONNXVAEDecoderWrapper:
    """ONNX VAE 解码器包装器"""
    def __init__(self, onnx_path):
        import onnxruntime as ort
        self.onnx_path = onnx_path
        self.device = device
        self.dtype = torch.float16
        
        logger.info(f"Loading MuseTalk VAE ONNX model from: {onnx_path}")
        
        # 配置 ONNX Runtime — 自动适配 NVIDIA CUDA / 沐曦 MACA / CPU
        from utils.device import get_onnx_providers
        providers = get_onnx_providers()
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        
        actual_providers = self.session.get_providers()
        logger.info(f"VAE ONNX model using providers: {actual_providers}")
        
        # 获取 scaling_factor（与 PyTorch VAE 保持一致）
        self.scaling_factor = 0.18215  # SD VAE 默认值
    
    def decode(self, latents):
        """解码 latent 为图像"""
        import numpy as np
        
        # 转为 numpy float16
        if isinstance(latents, torch.Tensor):
            latents_np = latents.cpu().half().numpy()
        else:
            latents_np = latents
        
        # 运行推理
        outputs = self.session.run(None, {'latents': latents_np})
        
        # 转回 torch tensor (输出已经是 [0, 1] 范围的图像)
        image = torch.from_numpy(outputs[0]).to(device=device, dtype=torch.float16)
        return type('obj', (object,), {'sample': image})()
    
    def decode_latents(self, latents):
        """兼容接口，与 PyTorch VAE 保持一致"""
        # latents 已经乘以了 scaling_factor，直接解码
        image = self.decode(latents).sample
        
        # 转换为 numpy 格式 [B, H, W, C] BGR
        image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")
        image = image[..., ::-1]  # RGB to BGR
        return image


def load_onnx_model(onnx_path, vae_onnx_path=None):
    """加载 ONNX 格式的 MuseTalk UNet 和 VAE 模型"""
    # 加载 PE，避免加载 PyTorch UNet 和 VAE
    from avatars.musetalk.models.unet import PositionalEncoding
    
    pe = PositionalEncoding(d_model=384)
    pe = pe.half().to(device)  # 使用 float16（与原始 PyTorch 一致）
    
    timesteps = torch.tensor([0], device=device)
    
    # 加载 ONNX UNet
    logger.info(f"Loading ONNX UNet from: {onnx_path}")
    unet = ONNXMuseTalkWrapper(onnx_path)
    
    # 加载 ONNX VAE（如果提供）
    if vae_onnx_path and os.path.exists(vae_onnx_path):
        logger.info(f"Loading ONNX VAE from: {vae_onnx_path}")
        vae = ONNXVAEDecoderWrapper(vae_onnx_path)
    else:
        # 回退到 PyTorch VAE
        logger.info("ONNX VAE not found, using PyTorch VAE")
        from avatars.musetalk.models.vae import VAE
        vae = VAE(model_path=os.path.join("models", "musetalk", "vae"))
        vae.vae = vae.vae.half().to(device)
    
    # Initialize audio processor
    audio_processor = Audio2Feature(model_path="./models/musetalk/whisper")
    
    return vae, unet, pe, timesteps, audio_processor

def load_avatar(avatar_id):
    avatar_path = f"./data/avatars/{avatar_id}"
    full_imgs_path = f"{avatar_path}/full_imgs" 
    coords_path = f"{avatar_path}/coords.pkl"
    latents_out_path= f"{avatar_path}/latents.pt"
    video_out_path = f"{avatar_path}/vid_output/"
    mask_out_path =f"{avatar_path}/mask"
    mask_coords_path =f"{avatar_path}/mask_coords.pkl"
    avatar_info_path = f"{avatar_path}/avator_info.json"

    input_latent_list_cycle = torch.load(latents_out_path)
    with open(coords_path, 'rb') as f:
        coord_list_cycle = pickle.load(f)
    frame_list_cycle = None
    input_img_list = glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    frame_list_cycle = read_imgs(input_img_list)
    with open(mask_coords_path, 'rb') as f:
        mask_coords_list_cycle = pickle.load(f)
    input_mask_list = glob.glob(os.path.join(mask_out_path, '*.[jpJP][pnPN]*[gG]'))
    input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    mask_list_cycle = read_imgs(input_mask_list)
    return frame_list_cycle,mask_list_cycle,coord_list_cycle,mask_coords_list_cycle,input_latent_list_cycle

@torch.no_grad()
def warm_up(batch_size,model):
    # 预热函数
    print('warmup model...')
    vae, unet, pe, timesteps, audio_processor = model
    whisper_batch = np.ones((batch_size, 50, 384), dtype=np.uint8)
    
    # MuseTalk 的 latent 尺寸：VAE 编码后 256x256 图像 -> 32x32 latent
    latent_h, latent_w = 32, 32
    
    latent_batch = torch.ones(batch_size, 8, latent_h, latent_w).to(unet.device)

    audio_feature_batch = torch.from_numpy(whisper_batch)
    audio_feature_batch = audio_feature_batch.to(device=unet.device, dtype=unet.model.dtype)
    audio_feature_batch = pe(audio_feature_batch)
    latent_batch = latent_batch.to(dtype=unet.model.dtype)
    pred_latents = unet.model(latent_batch,
                              timesteps,
                              encoder_hidden_states=audio_feature_batch).sample
    vae.decode_latents(pred_latents)    

@register("avatar", "musetalk")
class MuseReal(BaseAvatar):
    @torch.no_grad()
    def __init__(self, opt, model, avatar):
        super().__init__(opt)

        #self.fps = opt.fps # 20 ms per frame

        # self.batch_size = opt.batch_size
        # self.idx = 0
        # self.res_frame_queue = mp.Queue(self.batch_size*2)

        self.vae, self.unet, self.pe, self.timesteps, self.audio_processor = model

        self.frame_list_cycle,self.mask_list_cycle,self.coord_list_cycle,self.mask_coords_list_cycle, self.input_latent_list_cycle = avatar

        self.asr = WhisperASR(opt,self,self.audio_processor)
        self.asr.warm_up()
    

    def inference_batch(self, index, audiofeat_batch):
        # 这里的 index 是针对当前 avatar 的索引
        # 返回一个 batch 的推理结果，batch 大小由 self.batch_size 决定
        import time
        
        length = len(self.input_latent_list_cycle)
        whisper_batch = np.stack(audiofeat_batch)
        latent_batch = []
        for i in range(self.batch_size):
            idx = mirror_index(length, index + i)
            latent = self.input_latent_list_cycle[idx]
            latent_batch.append(latent)
        latent_batch = torch.cat(latent_batch, dim=0)
        
        # 性能分析
        t0 = time.time()
        audio_feature_batch = torch.from_numpy(whisper_batch)
        audio_feature_batch = audio_feature_batch.to(device=self.unet.device,
                                                        dtype=self.unet.model.dtype)
        t1 = time.time()
        audio_feature_batch = self.pe(audio_feature_batch)
        t2 = time.time()
        latent_batch = latent_batch.to(dtype=self.unet.model.dtype)
        t3 = time.time()

        pred_latents = self.unet.model(latent_batch, 
                                    self.timesteps, 
                                    encoder_hidden_states=audio_feature_batch).sample
        t4 = time.time()
        
        # 安全裁剪: 防止 MetaX GPU 上 latent 异常值导致 VAE 解码画面模糊
        pred_latents = torch.clamp(pred_latents, -3.5, 3.5)
        
        # DEBUG: 检查 pred_latents 的统计信息
        if index % 30 == 0:
            logger.info(f"[DEBUG] pred_latents shape: {pred_latents.shape}, dtype: {pred_latents.dtype}")
            logger.info(f"[DEBUG] pred_latents range: [{pred_latents.min():.4f}, {pred_latents.max():.4f}], mean: {pred_latents.mean():.4f}")
        
        pred = self.vae.decode_latents(pred_latents)
        t5 = time.time()
        
        # 打印性能分析（每10帧打印一次）
        if index % 10 == 0:
            logger.info(f"[Perf] data_prep: {(t1-t0)*1000:.1f}ms, "
                       f"pe: {(t2-t1)*1000:.1f}ms, "
                       f"type_conv: {(t3-t2)*1000:.1f}ms, "
                       f"unet: {(t4-t3)*1000:.1f}ms, "
                       f"vae: {(t5-t4)*1000:.1f}ms, "
                       f"total: {(t5-t0)*1000:.1f}ms")
        
        return pred

    def paste_back_frame(self,pred_frame,idx:int):
        bbox = self.coord_list_cycle[idx]
        ori_frame = copy.deepcopy(self.frame_list_cycle[idx])
        x1, y1, x2, y2 = bbox

        res_frame = cv2.resize(pred_frame.astype(np.uint8),(x2-x1,y2-y1))
        mask = self.mask_list_cycle[idx]
        mask_crop_box = self.mask_coords_list_cycle[idx]

        combine_frame = get_image_blending(ori_frame,res_frame,bbox,mask,mask_crop_box)
        return combine_frame
            
