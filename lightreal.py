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

import math
import torch
import numpy as np

#from .utils import *
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


from hubertasr import HubertASR
import asyncio
from av import AudioFrame, VideoFrame
from basereal import BaseReal

#from imgcache import ImgCache

from tqdm import tqdm

#new
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from transformers import Wav2Vec2Processor, HubertModel
from torch.utils.data import DataLoader
from ultralight.unet import Model
from ultralight.audio2feature import Audio2Feature
from logger import logger

device = "cuda" if torch.cuda.is_available() else ("mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu")
print('Using {} for inference.'.format(device))

def load_model(opt):
    audio_processor = Audio2Feature()
    return audio_processor

def load_avatar(avatar_id):
    avatar_path = f"./data/avatars/{avatar_id}"
    full_imgs_path = f"{avatar_path}/full_imgs" 
    face_imgs_path = f"{avatar_path}/face_imgs" 
    coords_path = f"{avatar_path}/coords.pkl" 
    
    model = Model(6, 'hubert').to(device)  # 假设Model是你自定义的类
    model.load_state_dict(torch.load(f"{avatar_path}/ultralight.pth"))
    
    with open(coords_path, 'rb') as f:
        coord_list_cycle = pickle.load(f)
    input_img_list = glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    frame_list_cycle = read_imgs(input_img_list)
    #self.imagecache = ImgCache(len(self.coord_list_cycle),self.full_imgs_path,1000)
    input_face_list = glob.glob(os.path.join(face_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_face_list = sorted(input_face_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    face_list_cycle = read_imgs(input_face_list)

    return model.eval(),frame_list_cycle,face_list_cycle,coord_list_cycle


@torch.no_grad()
def warm_up(batch_size,avatar,modelres):
    logger.info('warmup model...')
    model,_,_,_ = avatar
    img_batch = torch.ones(batch_size, 6, modelres, modelres).to(device)
    mel_batch = torch.ones(batch_size, 32, 32, 32).to(device)
    model(img_batch, mel_batch)

def read_imgs(img_list):
    frames = []
    logger.info('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

def get_audio_features(features, index):
    left = index - 8
    right = index + 8
    pad_left = 0
    pad_right = 0
    if left < 0:
        pad_left = -left
        left = 0
    if right > features.shape[0]:
        pad_right = right - features.shape[0]
        right = features.shape[0]
    auds = torch.from_numpy(features[left:right])
    if pad_left > 0:
        auds = torch.cat([torch.zeros_like(auds[:pad_left]), auds], dim=0)
    if pad_right > 0:
        auds = torch.cat([auds, torch.zeros_like(auds[:pad_right])], dim=0) # [8, 16]
    return auds


def read_lms(lms_list):
    land_marks = []
    logger.info('reading lms...')
    for lms_path in tqdm(lms_list):
        file_landmarks = []  # Store landmarks for this file
        with open(lms_path, "r") as f:
            lines = f.read().splitlines()
            for line in lines:
                arr = list(filter(None, line.split(" ")))
                if arr:
                    arr = np.array(arr, dtype=np.float32)
                    file_landmarks.append(arr)
        land_marks.append(file_landmarks)  # Add the file's landmarks to the overall list
    return land_marks

def __mirror_index(size, index):
    #size = len(self.coord_list_cycle)
    turn = index // size
    res = index % size
    if turn % 2 == 0:
        return res
    else:
        return size - res - 1 


def inference(quit_event, batch_size, face_list_cycle, audio_feat_queue, audio_out_queue, res_frame_queue, model):
    length = len(face_list_cycle)
    index = 0
    count = 0
    counttime = 0
    logger.info('start inference')

    while not quit_event.is_set():
        starttime=time.perf_counter()
        try:
            mel_batch = audio_feat_queue.get(block=True, timeout=1)
        except queue.Empty:
            continue
        is_all_silence=True
        audio_frames = []
        for _ in range(batch_size*2):
            frame,type_,eventpoint = audio_out_queue.get()
            audio_frames.append((frame,type_,eventpoint))
            if type_==0:
                is_all_silence=False
        if is_all_silence:
            for i in range(batch_size):
                res_frame_queue.put((None,__mirror_index(length,index),audio_frames[i*2:i*2+2]))
                index = index + 1
        else:
            t = time.perf_counter()
            img_batch = []

            for i in range(batch_size):
                idx = __mirror_index(length, index + i)
                #face = face_list_cycle[idx]
                crop_img = face_list_cycle[idx] #face[ymin:ymax, xmin:xmax]
#                h, w = crop_img.shape[:2]
                #crop_img = cv2.resize(crop_img, (168, 168), cv2.INTER_AREA)
                #crop_img_ori = crop_img.copy()
                img_real_ex = crop_img[4:164, 4:164].copy()
                img_real_ex_ori = img_real_ex.copy()
                img_masked = cv2.rectangle(img_real_ex_ori,(5,5,150,145),(0,0,0),-1)
    
                img_masked = img_masked.transpose(2,0,1).astype(np.float32)
                img_real_ex = img_real_ex.transpose(2,0,1).astype(np.float32)
    
                img_real_ex_T = torch.from_numpy(img_real_ex / 255.0)
                img_masked_T = torch.from_numpy(img_masked / 255.0)
                img_concat_T = torch.cat([img_real_ex_T, img_masked_T], axis=0)[None]
                img_batch.append(img_concat_T)

            reshaped_mel_batch = [arr.reshape(32, 32, 32) for arr in mel_batch]
            mel_batch = torch.stack([torch.from_numpy(arr) for arr in reshaped_mel_batch])
            img_batch = torch.stack(img_batch).squeeze(1)


            with torch.no_grad():
                pred = model(img_batch.cuda(),mel_batch.cuda())
            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

            counttime += (time.perf_counter() - t)
            count += batch_size
            if count >= 100:
                logger.info(f"------actual avg infer fps:{count / counttime:.4f}")
                count = 0
                counttime = 0
            for i,res_frame in enumerate(pred):
                #self.__pushmedia(res_frame,loop,audio_track,video_track)
                res_frame_queue.put((res_frame,__mirror_index(length,index),audio_frames[i*2:i*2+2]))
                index = index + 1

#            for i, pred_frame in enumerate(pred):
#                pred_frame_uint8 = np.array(pred_frame, dtype=np.uint8)
#                res_frame_queue.put((pred_frame_uint8, __mirror_index(length, index), audio_frames[i * 2:i * 2 + 2]))
#                index = (index + 1) % length

        #print('total batch time:', time.perf_counter() - starttime)

    logger.info('lightreal inference processor stop')


class LightReal(BaseReal):
    @torch.no_grad()
    def __init__(self, opt, model, avatar):
        super().__init__(opt)
        #self.opt = opt # shared with the trainer's opt to support in-place modification of rendering parameters.
        # self.W = opt.W
        # self.H = opt.H

        self.fps = opt.fps # 20 ms per frame
        
        self.batch_size = opt.batch_size
        self.idx = 0
        self.res_frame_queue = Queue(self.batch_size*2)  #mp.Queue
        #self.__loadavatar()
        audio_processor = model
        self.model,self.frame_list_cycle,self.face_list_cycle,self.coord_list_cycle = avatar

        self.asr = HubertASR(opt,self,audio_processor)
        self.asr.warm_up()
        #self.__warm_up()
        
        self.render_event = mp.Event()
    
    def __del__(self):
        logger.info(f'lightreal({self.sessionid}) delete')

    def paste_back_frame(self,pred_frame,idx:int):
        bbox = self.coord_list_cycle[idx]
        combine_frame = copy.deepcopy(self.frame_list_cycle[idx])
        x1, y1, x2, y2 = bbox

        crop_img = self.face_list_cycle[idx]
        crop_img_ori = crop_img.copy()
        #res_frame = np.array(res_frame, dtype=np.uint8)

        crop_img_ori[4:164, 4:164] = pred_frame.astype(np.uint8)
        crop_img_ori = cv2.resize(crop_img_ori, (x2-x1,y2-y1))
        combine_frame[y1:y2, x1:x2] = crop_img_ori
        return combine_frame
            
    def render(self,quit_event,loop=None,audio_track=None,video_track=None):
        #if self.opt.asr:
        #     self.asr.warm_up()

        self.tts.render(quit_event)
        self.init_customindex()
        process_thread = Thread(target=self.process_frames, args=(quit_event,loop,audio_track,video_track))
        process_thread.start()
        Thread(target=inference, args=(quit_event,self.batch_size,self.face_list_cycle,self.asr.feat_queue,self.asr.output_queue,self.res_frame_queue,
                                           self.model,)).start()  #mp.Process
        

        #self.render_event.set() #start infer process render
        count=0
        totaltime=0
        _starttime=time.perf_counter()
        #_totalframe=0
        while not quit_event.is_set(): 
            # update texture every frame
            # audio stream thread...
            t = time.perf_counter()
            self.asr.run_step()

            # if video_track._queue.qsize()>=2*self.opt.batch_size:
            #     print('sleep qsize=',video_track._queue.qsize())
            #     time.sleep(0.04*video_track._queue.qsize()*0.8)
            if video_track and video_track._queue.qsize()>=5:
                logger.debug('sleep qsize=%d',video_track._queue.qsize())
                time.sleep(0.04*video_track._queue.qsize()*0.8)
                
            # delay = _starttime+_totalframe*0.04-time.perf_counter() #40ms
            # if delay > 0:
            #     time.sleep(delay)
        #self.render_event.clear() #end infer process render
        logger.info('lightreal thread stop')
            

