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
#  Wav2Lip 数字人 — 迁移自 lipreal.py + lipasr.py
#

import math
import torch
import numpy as np

import os
import time
import cv2
import glob
import pickle
import copy

import queue
from queue import Queue
from threading import Thread, Event
import torch.multiprocessing as mp

from avatars.audio_features.mel import MelASR
import asyncio
from av import AudioFrame, VideoFrame
from avatars.wav2lip.models import Wav2Lip
from avatars.base_avatar import BaseAvatar

from tqdm import tqdm
from utils.logger import logger
from utils.image import read_imgs, mirror_index
from utils.device import initialize_device
from registry import register

device = initialize_device()
logger.info('Using {} for inference.'.format(device))

def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_model(path):
    model = Wav2Lip()
    logger.info("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()


class ONNXWav2LipWrapper:
    """ONNX 模型包装器，提供与 PyTorch 模型相同的接口"""
    def __init__(self, onnx_path):
        import onnxruntime as ort
        self.onnx_path = onnx_path
        logger.info(f"Loading ONNX model from: {onnx_path}")
        
        # 配置 ONNX Runtime — 自动适配 NVIDIA CUDA / 沐曦 MACA / CPU
        from utils.device import get_onnx_providers
        providers = get_onnx_providers()
        logger.info(f"ONNX Runtime available providers: {ort.get_available_providers()}")
        logger.info(f"Trying to use providers: {providers}")
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        
        # 获取实际使用的 provider
        actual_providers = self.session.get_providers()
        logger.info(f"ONNX model actually using providers: {actual_providers}")
        
        # 获取输入输出信息
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        logger.info(f"ONNX model inputs: {self.input_names}")
        logger.info(f"ONNX model outputs: {self.output_names}")
    
    def __call__(self, audio_sequences, face_sequences):
        """推理接口，兼容 PyTorch 模型调用方式"""
        import numpy as np
        
        # 将 torch.Tensor 转换为 numpy
        if isinstance(audio_sequences, torch.Tensor):
            audio_np = audio_sequences.cpu().numpy()
        else:
            audio_np = audio_sequences
            
        if isinstance(face_sequences, torch.Tensor):
            face_np = face_sequences.cpu().numpy()
        else:
            face_np = face_sequences
        
        # 运行推理
        outputs = self.session.run(
            None,
            {
                self.input_names[0]: audio_np,
                self.input_names[1]: face_np
            }
        )
        
        # 将输出转换回 torch.Tensor
        return torch.from_numpy(outputs[0]).to(device)
    
    def eval(self):
        """兼容接口，ONNX 模型不需要 eval 模式"""
        return self
    
    def to(self, device):
        """兼容接口，ONNX 模型在 session 创建时已确定设备"""
        return self


def load_onnx_model(onnx_path):
    """加载 ONNX 格式的 Wav2Lip 模型"""
    return ONNXWav2LipWrapper(onnx_path)

def load_avatar(avatar_id):
    avatar_path = f"./data/avatars/{avatar_id}"
    full_imgs_path = f"{avatar_path}/full_imgs" 
    face_imgs_path = f"{avatar_path}/face_imgs" 
    coords_path = f"{avatar_path}/coords.pkl"
    
    with open(coords_path, 'rb') as f:
        coord_list_cycle = pickle.load(f)
    frame_list_cycle = None
    input_img_list = glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    frame_list_cycle = read_imgs(input_img_list)
    input_face_list = glob.glob(os.path.join(face_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_face_list = sorted(input_face_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    face_list_cycle = read_imgs(input_face_list)

    return frame_list_cycle,face_list_cycle,coord_list_cycle

@torch.no_grad()
def warm_up(batch_size,model,modelres):
    # 预热函数
    logger.info('warmup model...')
    img_batch = torch.ones(batch_size, 6, modelres, modelres).to(device)
    mel_batch = torch.ones(batch_size, 1, 80, 16).to(device)
    model(mel_batch, img_batch)

@register("avatar", "wav2lip")
class LipReal(BaseAvatar):
    @torch.no_grad()
    def __init__(self, opt, model, avatar):
        super().__init__(opt)

        #self.fps = opt.fps # 20 ms per frame
        
        # self.batch_size = opt.batch_size
        # self.idx = 0
        # self.res_frame_queue = Queue(self.batch_size*2)
        self.model = model

        self.frame_list_cycle,self.face_list_cycle,self.coord_list_cycle = avatar

        self.asr = MelASR(opt,self)
        self.asr.warm_up()
    
    def inference_batch(self, index, audiofeat_batch):
        # 这里的 index 是针对当前 avatar 的索引
        # 返回一个 batch 的推理结果，batch 大小由 self.batch_size 决定
        length = len(self.face_list_cycle)
        img_batch = []
        for i in range(self.batch_size):
            idx = mirror_index(length, index + i)
            face = self.face_list_cycle[idx]
            img_batch.append(face)
        img_batch, audiofeat_batch = np.asarray(img_batch), np.asarray(audiofeat_batch)

        img_masked = img_batch.copy()
        img_masked[:, face.shape[0]//2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        audiofeat_batch = np.reshape(audiofeat_batch, [len(audiofeat_batch), audiofeat_batch.shape[1], audiofeat_batch.shape[2], 1])
        
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        audiofeat_batch = torch.FloatTensor(np.transpose(audiofeat_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = self.model(audiofeat_batch, img_batch)
        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
        return pred

    def paste_back_frame(self,pred_frame,idx:int):
        bbox = self.coord_list_cycle[idx]
        combine_frame = copy.deepcopy(self.frame_list_cycle[idx])
        y1, y2, x1, x2 = bbox
        res_frame = cv2.resize(pred_frame.astype(np.uint8),(x2-x1,y2-y1))
        combine_frame[y1:y2, x1:x2] = res_frame
        return combine_frame