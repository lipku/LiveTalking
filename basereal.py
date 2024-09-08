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
from io import BytesIO
import soundfile as sf

import av
from fractions import Fraction

from ttsreal import EdgeTTS,VoitsTTS,XTTS,CosyVoiceTTS

from tqdm import tqdm
def read_imgs(img_list):
    frames = []
    print('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

class BaseReal:
    def __init__(self, opt):
        self.opt = opt
        self.sample_rate = 16000
        self.chunk = self.sample_rate // opt.fps # 320 samples per chunk (20ms * 16000 / 1000)

        if opt.tts == "edgetts":
            self.tts = EdgeTTS(opt,self)
        elif opt.tts == "gpt-sovits":
            self.tts = VoitsTTS(opt,self)
        elif opt.tts == "xtts":
            self.tts = XTTS(opt,self)
        elif opt.tts == "cosyvoice":
            self.tts = CosyVoiceTTS(opt,self)
        
        self.recording = False
        self.recordq_video = Queue()
        self.recordq_audio = Queue()

        self.curr_state=0
        self.custom_img_cycle = {}
        self.custom_audio_cycle = {}
        self.custom_audio_index = {}
        self.custom_index = {}
        self.custom_opt = {}
        self.__loadcustom()
    
    def __loadcustom(self):
        for item in self.opt.customopt:
            print(item)
            input_img_list = glob.glob(os.path.join(item['imgpath'], '*.[jpJP][pnPN]*[gG]'))
            input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            self.custom_img_cycle[item['audiotype']] = read_imgs(input_img_list)
            self.custom_audio_cycle[item['audiotype']], sample_rate = sf.read(item['audiopath'], dtype='float32')
            self.custom_audio_index[item['audiotype']] = 0
            self.custom_index[item['audiotype']] = 0
            self.custom_opt[item['audiotype']] = item

    def init_customindex(self):
        self.curr_state=0
        for key in self.custom_audio_index:
            self.custom_audio_index[key]=0
        for key in self.custom_index:
            self.custom_index[key]=0

    def start_recording(self,path):
        """开始录制视频"""
        if self.recording:
            return
        self.recording = True
        self.recordq_video.queue.clear()
        self.recordq_audio.queue.clear()
        self.container = av.open(path, mode="w")
    
        process_thread = Thread(target=self.record_frame, args=())
        process_thread.start()
    
    def record_frame(self): 
        videostream = self.container.add_stream("libx264", rate=25)
        videostream.codec_context.time_base = Fraction(1, 25)
        audiostream = self.container.add_stream("aac")
        audiostream.codec_context.time_base = Fraction(1, 16000)
        init = True
        framenum = 0       
        while self.recording:
            try:
                videoframe = self.recordq_video.get(block=True, timeout=1)
                videoframe.pts = framenum #int(round(framenum*0.04 / videostream.codec_context.time_base))
                videoframe.dts = videoframe.pts
                if init:
                    videostream.width = videoframe.width
                    videostream.height = videoframe.height
                    init = False
                for packet in videostream.encode(videoframe):
                    self.container.mux(packet)
                for k in range(2):
                    audioframe = self.recordq_audio.get(block=True, timeout=1)
                    audioframe.pts = int(round((framenum*2+k)*0.02 / audiostream.codec_context.time_base))
                    audioframe.dts = audioframe.pts
                    for packet in audiostream.encode(audioframe):
                        self.container.mux(packet)
                framenum += 1
            except queue.Empty:
                print('record queue empty,')
                continue
            except Exception as e:
                print(e)
                #break
        for packet in videostream.encode(None):
            self.container.mux(packet)
        for packet in audiostream.encode(None):
            self.container.mux(packet)
        self.container.close()
        self.recordq_video.queue.clear()
        self.recordq_audio.queue.clear()
        print('record thread stop')
		
    def stop_recording(self):
        """停止录制视频"""
        if not self.recording:
            return
        self.recording = False        

    def mirror_index(self,size, index):
        #size = len(self.coord_list_cycle)
        turn = index // size
        res = index % size
        if turn % 2 == 0:
            return res
        else:
            return size - res - 1 
    
    def get_audio_stream(self,audiotype):
        idx = self.custom_audio_index[audiotype]
        stream = self.custom_audio_cycle[audiotype][idx:idx+self.chunk]
        self.custom_audio_index[audiotype] += self.chunk
        if self.custom_audio_index[audiotype]>=self.custom_audio_cycle[audiotype].shape[0]:
            self.curr_state = 1  #当前视频不循环播放，切换到静音状态
        return stream
    
    def set_curr_state(self,audiotype, reinit):
        print('set_curr_state:',audiotype)
        self.curr_state = audiotype
        if reinit:
            self.custom_audio_index[audiotype] = 0
            self.custom_index[audiotype] = 0
    
    # def process_custom(self,audiotype:int,idx:int):
    #     if self.curr_state!=audiotype: #从推理切到口播
    #         if idx in self.switch_pos:  #在卡点位置可以切换
    #             self.curr_state=audiotype
    #             self.custom_index=0
    #     else:
    #         self.custom_index+=1