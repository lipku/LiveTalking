###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku@foxmail.com
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
#  UltraLight 数字人 — 迁移自 lightreal.py
#  使用 HubertASR 音频特征提取（与 wav2lipls 共享）
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

from avatars.audio_features.hubert import HubertASR
import asyncio
from av import AudioFrame, VideoFrame
from avatars.base_avatar import BaseAvatar

from tqdm import tqdm

import torch.nn as nn
from torch import optim
from transformers import Wav2Vec2Processor, HubertModel
from torch.utils.data import DataLoader
from avatars.ultralight.unet import Model
from avatars.ultralight.audio2feature import Audio2Feature
from utils.logger import logger
from utils.image import read_imgs, mirror_index
from utils.device import initialize_device
from registry import register

device = initialize_device()
logger.info('Using {} for inference.'.format(device))

def load_model(opt):
    audio_processor = Audio2Feature()
    model = None
    return audio_processor,model

def load_avatar(avatar_id):
    avatar_path = f"./data/avatars/{avatar_id}"
    full_imgs_path = f"{avatar_path}/full_imgs" 
    face_imgs_path = f"{avatar_path}/face_imgs" 
    coords_path = f"{avatar_path}/coords.pkl" 
    
    model = Model(6, 'hubert').to(device)
    model.load_state_dict(torch.load(f"{avatar_path}/ultralight.pth"))
    
    with open(coords_path, 'rb') as f:
        coord_list_cycle = pickle.load(f)
    input_img_list = glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    frame_list_cycle = read_imgs(input_img_list)
    input_face_list = glob.glob(os.path.join(face_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_face_list = sorted(input_face_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    face_list_cycle = read_imgs(input_face_list)

    return model.eval(),frame_list_cycle,face_list_cycle,coord_list_cycle


@torch.no_grad()
def warm_up(batch_size,avatar,modelres):
    logger.info('warmup model...')
    model,_,_,_ = avatar
    img_batch = torch.ones(batch_size, 6, modelres, modelres).to(device)
    mel_batch = torch.ones(batch_size, 16, 32, 32).to(device)
    model(img_batch, mel_batch)

# def get_audio_features(features, index):
#     left = index - 8
#     right = index + 8
#     pad_left = 0
#     pad_right = 0
#     if left < 0:
#         pad_left = -left
#         left = 0
#     if right > features.shape[0]:
#         pad_right = right - features.shape[0]
#         right = features.shape[0]
#     auds = torch.from_numpy(features[left:right])
#     if pad_left > 0:
#         auds = torch.cat([torch.zeros_like(auds[:pad_left]), auds], dim=0)
#     if pad_right > 0:
#         auds = torch.cat([auds, torch.zeros_like(auds[:pad_right])], dim=0) # [8, 16]
#     return auds


# def read_lms(lms_list):
#     land_marks = []
#     logger.info('reading lms...')
#     for lms_path in tqdm(lms_list):
#         file_landmarks = []
#         with open(lms_path, "r") as f:
#             lines = f.read().splitlines()
#             for line in lines:
#                 arr = list(filter(None, line.split(" ")))
#                 if arr:
#                     arr = np.array(arr, dtype=np.float32)
#                     file_landmarks.append(arr)
#         land_marks.append(file_landmarks)
#     return land_marks
@register("avatar", "ultralight")
class LightReal(BaseAvatar):
    @torch.no_grad()
    def __init__(self, opt, model, avatar):
        super().__init__(opt)

        #self.fps = opt.fps # 20 ms per frame
        
        # self.batch_size = opt.batch_size
        # self.idx = 0
        # self.res_frame_queue = Queue(self.batch_size*2)

        audio_processor,_ = model
        self.model,self.frame_list_cycle,self.face_list_cycle,self.coord_list_cycle = avatar

        self.asr = HubertASR(opt,self,audio_processor,audio_feat_length=[4,4])
        self.asr.warm_up()

    def inference_batch(self, index, audiofeat_batch):
        # 这里的 index 是针对当前 avatar 的索引
        # 返回一个 batch 的推理结果，batch 大小由 self.batch_size 决定
        length = len(self.face_list_cycle)
        img_batch = []

        for i in range(self.batch_size):
            idx = mirror_index(length, index + i)
            crop_img = self.face_list_cycle[idx]
            img_real_ex = crop_img[4:164, 4:164].copy()
            img_real_ex_ori = img_real_ex.copy()
            img_masked = cv2.rectangle(img_real_ex_ori,(5,5,150,145),(0,0,0),-1)

            img_masked = img_masked.transpose(2,0,1).astype(np.float32)
            img_real_ex = img_real_ex.transpose(2,0,1).astype(np.float32)

            img_real_ex_T = torch.from_numpy(img_real_ex / 255.0)
            img_masked_T = torch.from_numpy(img_masked / 255.0)
            img_concat_T = torch.cat([img_real_ex_T, img_masked_T], axis=0)[None]
            img_batch.append(img_concat_T)

        reshaped_audiofeat_batch = [arr.reshape(16, 32, 32) for arr in audiofeat_batch]
        audiofeat_batch = torch.stack([torch.from_numpy(arr) for arr in reshaped_audiofeat_batch])
        img_batch = torch.stack(img_batch).squeeze(1)

        with torch.no_grad():
            pred = self.model(img_batch.cuda(), audiofeat_batch.cuda())
        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
        return pred
    
    def paste_back_frame(self,pred_frame,idx:int):
        bbox = self.coord_list_cycle[idx]
        combine_frame = copy.deepcopy(self.frame_list_cycle[idx])
        x1, y1, x2, y2 = bbox

        crop_img = self.face_list_cycle[idx]
        crop_img_ori = crop_img.copy()

        crop_img_ori[4:164, 4:164] = pred_frame.astype(np.uint8)
        crop_img_ori = cv2.resize(crop_img_ori, (x2-x1,y2-y1))
        combine_frame[y1:y2, x1:x2] = crop_img_ori
        return combine_frame

            
