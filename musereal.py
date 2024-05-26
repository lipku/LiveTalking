import math
import torch
import numpy as np

#from .utils import *
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
from io import BytesIO

from musetalk.utils.utils import get_file_type,get_video_fps,datagen
from musetalk.utils.preprocessing import get_landmark_and_bbox,read_imgs,coord_placeholder
from musetalk.utils.blending import get_image,get_image_prepare_material,get_image_blending
from musetalk.utils.utils import load_all_model

from museasr import MuseASR
import asyncio
from av import AudioFrame, VideoFrame

class MuseReal:
    def __init__(self, opt):
        self.opt = opt # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.W = opt.W
        self.H = opt.H

        self.fps = opt.fps # 20 ms per frame

        #### musetalk
        self.avatar_id = opt.avatar_id
        self.video_path = '' #video_path
        self.bbox_shift = opt.bbox_shift
        self.avatar_path = f"./data/avatars/{self.avatar_id}"
        self.full_imgs_path = f"{self.avatar_path}/full_imgs" 
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.latents_out_path= f"{self.avatar_path}/latents.pt"
        self.video_out_path = f"{self.avatar_path}/vid_output/"
        self.mask_out_path =f"{self.avatar_path}/mask"
        self.mask_coords_path =f"{self.avatar_path}/mask_coords.pkl"
        self.avatar_info_path = f"{self.avatar_path}/avator_info.json"
        self.avatar_info = {
            "avatar_id":self.avatar_id,
            "video_path":self.video_path,
            "bbox_shift":self.bbox_shift   
        }
        self.batch_size = opt.batch_size
        self.idx = 0
        self.res_frame_queue = Queue()
        self.__loadmodels()
        self.__loadavatar()

        self.asr = MuseASR(opt,self.audio_processor)

    def __loadmodels(self):
        # load model weights
        self.audio_processor, self.vae, self.unet, self.pe = load_all_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.timesteps = torch.tensor([0], device=device)
        self.pe = self.pe.half()
        self.vae.vae = self.vae.vae.half()
        self.unet.model = self.unet.model.half()

    def __loadavatar(self):
        self.input_latent_list_cycle = torch.load(self.latents_out_path)
        with open(self.coords_path, 'rb') as f:
            self.coord_list_cycle = pickle.load(f)
        input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
        input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.frame_list_cycle = read_imgs(input_img_list)
        with open(self.mask_coords_path, 'rb') as f:
            self.mask_coords_list_cycle = pickle.load(f)
        input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
        input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.mask_list_cycle = read_imgs(input_mask_list)
        
    
    def push_audio(self,buffer):
        self.asr.push_audio(buffer)

    def __mirror_index(self, index):
        size = len(self.coord_list_cycle)
        turn = index // size
        res = index % size
        if turn % 2 == 0:
            return res
        else:
            return size - res - 1   

    def test_step(self,loop=None,audio_track=None,video_track=None):
        
        # gen = datagen(whisper_chunks,
        #               self.input_latent_list_cycle, 
        #               self.batch_size)
        self.asr.run_step()
        whisper_chunks = self.asr.get_next_feat()
        is_all_silence=True
        audio_frames = []
        for _ in range(self.batch_size*2):
            frame,type = self.asr.get_audio_out()
            audio_frames.append((frame,type))
            if type==0:
                is_all_silence=False
        if is_all_silence:
            for i in range(self.batch_size):
                self.res_frame_queue.put((None,self.__mirror_index(self.idx),audio_frames[i*2:i*2+2]))
                self.idx = self.idx + 1
        else:
            print('infer=======')
            whisper_batch = np.stack(whisper_chunks)
            latent_batch = []
            for i in range(self.batch_size):
                idx = self.__mirror_index(self.idx+i)
                latent = self.input_latent_list_cycle[idx]
                latent_batch.append(latent)
            latent_batch = torch.cat(latent_batch, dim=0)
            
            # for i, (whisper_batch,latent_batch) in enumerate(gen):
            audio_feature_batch = torch.from_numpy(whisper_batch)
            audio_feature_batch = audio_feature_batch.to(device=self.unet.device,
                                                            dtype=self.unet.model.dtype)
            audio_feature_batch = self.pe(audio_feature_batch)
            latent_batch = latent_batch.to(dtype=self.unet.model.dtype)

            pred_latents = self.unet.model(latent_batch, 
                                        self.timesteps, 
                                        encoder_hidden_states=audio_feature_batch).sample
            recon = self.vae.decode_latents(pred_latents)
            #print('diffusion len=',len(recon))
            for i,res_frame in enumerate(recon):
                #self.__pushmedia(res_frame,loop,audio_track,video_track)
                self.res_frame_queue.put((res_frame,self.__mirror_index(self.idx),audio_frames[i*2:i*2+2]))
                self.idx = self.idx + 1
      

    def process_frames(self,quit_event,loop=None,audio_track=None,video_track=None):
        
        while not quit_event.is_set():
            try:
                res_frame,idx,audio_frames = self.res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
            if audio_frames[0][1]==1 and audio_frames[1][1]==1: #全为静音数据，只需要取fullimg
                combine_frame = self.frame_list_cycle[idx]
            else:
                bbox = self.coord_list_cycle[idx]
                ori_frame = copy.deepcopy(self.frame_list_cycle[idx])
                x1, y1, x2, y2 = bbox
                try:
                    res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
                except:
                    continue
                mask = self.mask_list_cycle[idx]
                mask_crop_box = self.mask_coords_list_cycle[idx]
                #combine_frame = get_image(ori_frame,res_frame,bbox)
                combine_frame = get_image_blending(ori_frame,res_frame,bbox,mask,mask_crop_box)

            image = combine_frame #(outputs['image'] * 255).astype(np.uint8)
            new_frame = VideoFrame.from_ndarray(image, format="bgr24")
            asyncio.run_coroutine_threadsafe(video_track._queue.put(new_frame), loop) 

            for audio_frame in audio_frames:
                frame,type = audio_frame
                frame = (frame * 32767).astype(np.int16)
                new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
                new_frame.planes[0].update(frame.tobytes())
                new_frame.sample_rate=16000
                # if audio_track._queue.qsize()>10:
                #     time.sleep(0.1)
                asyncio.run_coroutine_threadsafe(audio_track._queue.put(new_frame), loop) 
            
    def render(self,quit_event,loop=None,audio_track=None,video_track=None):
        #if self.opt.asr:
        #     self.asr.warm_up()

        process_thread = Thread(target=self.process_frames, args=(quit_event,loop,audio_track,video_track))
        process_thread.start()

        count=0
        totaltime=0
        _starttime=time.perf_counter()
        #_totalframe=0
        while not quit_event.is_set(): #todo
            # update texture every frame
            # audio stream thread...
            t = time.perf_counter()
            self.test_step(loop,audio_track,video_track)
            totaltime += (time.perf_counter() - t)
            count += self.opt.batch_size
            #_totalframe += 1
            if count>=100:
                print(f"------actual avg infer fps:{count/totaltime:.4f}")
                count=0
                totaltime=0
            if video_track._queue.qsize()>=2*self.opt.batch_size:
                #print('sleep qsize=',video_track._queue.qsize())
                time.sleep(0.04*self.opt.batch_size*1.5)
                
            # delay = _starttime+_totalframe*0.04-time.perf_counter() #40ms
            # if delay > 0:
            #     time.sleep(delay)
            