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


from lipasr import LipASR
import asyncio
from av import AudioFrame, VideoFrame
from wav2lip.models import Wav2Lip
from basereal import BaseReal

#from imgcache import ImgCache

from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

def _load(checkpoint_path):
	if device == 'cuda':
		checkpoint = torch.load(checkpoint_path) #,weights_only=True
	else:
		checkpoint = torch.load(checkpoint_path,
								map_location=lambda storage, loc: storage)
	return checkpoint

def load_model(path):
	model = Wav2Lip()
	print("Load checkpoint from: {}".format(path))
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	model = model.to(device)
	return model.eval()

def load_avatar(avatar_id):
    avatar_path = f"./data/avatars/{avatar_id}"
    full_imgs_path = f"{avatar_path}/full_imgs" 
    face_imgs_path = f"{avatar_path}/face_imgs" 
    coords_path = f"{avatar_path}/coords.pkl"
    
    with open(coords_path, 'rb') as f:
        coord_list_cycle = pickle.load(f)
    input_img_list = glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    frame_list_cycle = read_imgs(input_img_list)
    #self.imagecache = ImgCache(len(self.coord_list_cycle),self.full_imgs_path,1000)
    input_face_list = glob.glob(os.path.join(face_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_face_list = sorted(input_face_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    face_list_cycle = read_imgs(input_face_list)

    return frame_list_cycle,face_list_cycle,coord_list_cycle

@torch.no_grad()
def warm_up(batch_size,model,modelres):
    # 预热函数
    print('warmup model...')
    img_batch = torch.ones(batch_size, 6, modelres, modelres).to(device)
    mel_batch = torch.ones(batch_size, 1, 80, 16).to(device)
    model(mel_batch, img_batch)

def read_imgs(img_list):
    frames = []
    print('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

def __mirror_index(size, index):
    #size = len(self.coord_list_cycle)
    turn = index // size
    res = index % size
    if turn % 2 == 0:
        return res
    else:
        return size - res - 1 

def inference(quit_event,batch_size,face_list_cycle,audio_feat_queue,audio_out_queue,res_frame_queue,model):
    
    #model = load_model("./models/wav2lip.pth")
    # input_face_list = glob.glob(os.path.join(face_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    # input_face_list = sorted(input_face_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    # face_list_cycle = read_imgs(input_face_list)
    
    #input_latent_list_cycle = torch.load(latents_out_path)
    length = len(face_list_cycle)
    index = 0
    count=0
    counttime=0
    print('start inference')
    while not quit_event.is_set():
        starttime=time.perf_counter()
        mel_batch = []
        try:
            mel_batch = audio_feat_queue.get(block=True, timeout=1)
        except queue.Empty:
            continue
            
        is_all_silence=True
        audio_frames = []
        for _ in range(batch_size*2):
            frame,type = audio_out_queue.get()
            audio_frames.append((frame,type))
            if type==0:
                is_all_silence=False

        if is_all_silence:
            for i in range(batch_size):
                res_frame_queue.put((None,__mirror_index(length,index),audio_frames[i*2:i*2+2]))
                index = index + 1
        else:
            # print('infer=======')
            t=time.perf_counter()
            img_batch = []
            for i in range(batch_size):
                idx = __mirror_index(length,index+i)
                face = face_list_cycle[idx]
                img_batch.append(face)
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, face.shape[0]//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
            
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

            with torch.no_grad():
                pred = model(mel_batch, img_batch)
            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

            counttime += (time.perf_counter() - t)
            count += batch_size
            #_totalframe += 1
            if count>=100:
                print(f"------actual avg infer fps:{count/counttime:.4f}")
                count=0
                counttime=0
            for i,res_frame in enumerate(pred):
                #self.__pushmedia(res_frame,loop,audio_track,video_track)
                res_frame_queue.put((res_frame,__mirror_index(length,index),audio_frames[i*2:i*2+2]))
                index = index + 1
            #print('total batch time:',time.perf_counter()-starttime)            
    print('lipreal inference processor stop')

class LipReal(BaseReal):
    @torch.no_grad()
    def __init__(self, opt, model, avatar):
        super().__init__(opt)
        #self.opt = opt # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.W = opt.W
        self.H = opt.H

        self.fps = opt.fps # 20 ms per frame
        
        self.batch_size = opt.batch_size
        self.idx = 0
        self.res_frame_queue = Queue(self.batch_size*2)  #mp.Queue
        #self.__loadavatar()
        self.model = model
        self.frame_list_cycle,self.face_list_cycle,self.coord_list_cycle = avatar

        self.asr = LipASR(opt,self)
        self.asr.warm_up()
        
        self.render_event = mp.Event()
    
    def __del__(self):
        print(f'lipreal({self.sessionid}) delete')

   
    def process_frames(self,quit_event,loop=None,audio_track=None,video_track=None):
        
        while not quit_event.is_set():
            try:
                res_frame,idx,audio_frames = self.res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
            if audio_frames[0][1]!=0 and audio_frames[1][1]!=0: #全为静音数据，只需要取fullimg
                self.speaking = False
                audiotype = audio_frames[0][1]
                if self.custom_index.get(audiotype) is not None: #有自定义视频
                    mirindex = self.mirror_index(len(self.custom_img_cycle[audiotype]),self.custom_index[audiotype])
                    combine_frame = self.custom_img_cycle[audiotype][mirindex]
                    self.custom_index[audiotype] += 1
                    # if not self.custom_opt[audiotype].loop and self.custom_index[audiotype]>=len(self.custom_img_cycle[audiotype]):
                    #     self.curr_state = 1  #当前视频不循环播放，切换到静音状态
                else:
                    combine_frame = self.frame_list_cycle[idx]
                    #combine_frame = self.imagecache.get_img(idx)
            else:
                self.speaking = True
                bbox = self.coord_list_cycle[idx]
                combine_frame = copy.deepcopy(self.frame_list_cycle[idx])
                #combine_frame = copy.deepcopy(self.imagecache.get_img(idx))
                y1, y2, x1, x2 = bbox
                try:
                    res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
                except:
                    continue
                #combine_frame = get_image(ori_frame,res_frame,bbox)
                #t=time.perf_counter()
                combine_frame[y1:y2, x1:x2] = res_frame
                #print('blending time:',time.perf_counter()-t)

            image = combine_frame #(outputs['image'] * 255).astype(np.uint8)
            new_frame = VideoFrame.from_ndarray(image, format="bgr24")
            asyncio.run_coroutine_threadsafe(video_track._queue.put(new_frame), loop)
            self.record_video_data(image)

            for audio_frame in audio_frames:
                frame,type = audio_frame
                frame = (frame * 32767).astype(np.int16)
                new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
                new_frame.planes[0].update(frame.tobytes())
                new_frame.sample_rate=16000
                # if audio_track._queue.qsize()>10:
                #     time.sleep(0.1)
                asyncio.run_coroutine_threadsafe(audio_track._queue.put(new_frame), loop)
                self.record_audio_data(frame)
        print('lipreal process_frames thread stop') 
            
    def render(self,quit_event,loop=None,audio_track=None,video_track=None):
        #if self.opt.asr:
        #     self.asr.warm_up()

        self.tts.render(quit_event)
        self.init_customindex()
        process_thread = Thread(target=self.process_frames, args=(quit_event,loop,audio_track,video_track))
        process_thread.start()

        Thread(target=inference, args=(quit_event,self.batch_size,self.face_list_cycle,
                                           self.asr.feat_queue,self.asr.output_queue,self.res_frame_queue,
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
            if video_track._queue.qsize()>=5:
                print('sleep qsize=',video_track._queue.qsize())
                time.sleep(0.04*video_track._queue.qsize()*0.8)
                
            # delay = _starttime+_totalframe*0.04-time.perf_counter() #40ms
            # if delay > 0:
            #     time.sleep(delay)
        #self.render_event.clear() #end infer process render
        print('lipreal thread stop')
            