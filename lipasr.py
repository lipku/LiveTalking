import time
import torch
import numpy as np
import soundfile as sf
import resampy

import queue
from queue import Queue
from io import BytesIO
import multiprocessing as mp

from wav2lip import audio

class LipASR:
    def __init__(self, opt):
        self.opt = opt

        self.fps = opt.fps # 20 ms per frame
        self.sample_rate = 16000
        self.chunk = self.sample_rate // self.fps # 320 samples per chunk (20ms * 16000 / 1000)
        self.queue = Queue()
        # self.input_stream = BytesIO()
        self.output_queue = mp.Queue()

        #self.audio_processor = audio_processor
        self.batch_size = opt.batch_size

        self.frames = []
        self.stride_left_size = self.stride_right_size = 10
        self.context_size = 10
        self.audio_feats = []
        self.feat_queue = mp.Queue(5)

        self.warm_up()

    def put_audio_frame(self,audio_chunk): #16khz 20ms pcm
        self.queue.put(audio_chunk)

    def __get_audio_frame(self):        
        try:
            frame = self.queue.get(block=True,timeout=0.018)
            type = 0
            #print(f'[INFO] get frame {frame.shape}')
        except queue.Empty:
            frame = np.zeros(self.chunk, dtype=np.float32)
            type = 1

        return frame,type 

    def get_audio_out(self):  #get origin audio pcm to nerf
        return self.output_queue.get()
    
    def warm_up(self):
        for _ in range(self.stride_left_size + self.stride_right_size):
            audio_frame,type=self.__get_audio_frame()
            self.frames.append(audio_frame)
            self.output_queue.put((audio_frame,type))
        for _ in range(self.stride_left_size):
            self.output_queue.get()

    def run_step(self):
        ############################################## extract audio feature ##############################################
        # get a frame of audio
        for _ in range(self.batch_size*2):
            frame,type = self.__get_audio_frame()
            self.frames.append(frame)
            # put to output
            self.output_queue.put((frame,type))
        # context not enough, do not run network.
        if len(self.frames) < self.stride_left_size + self.context_size + self.stride_right_size:
            return
        
        inputs = np.concatenate(self.frames) # [N * chunk]
        mel = audio.melspectrogram(inputs)
        #print(mel.shape[0],mel.shape,len(mel[0]),len(self.frames))
        # cut off stride
        left = max(0, self.stride_left_size*80/50)
        right = min(len(mel[0]), len(mel[0]) - self.stride_right_size*80/50)
        mel_idx_multiplier = 80.*2/self.fps 
        mel_step_size = 16
        i = 0
        mel_chunks = []
        while i < (len(self.frames)-self.stride_left_size-self.stride_right_size)/2:
            start_idx = int(left + i * mel_idx_multiplier)
            #print(start_idx)
            if start_idx + mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            else:
                mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
            i += 1
        self.feat_queue.put(mel_chunks)
        
        # discard the old part to save memory
        self.frames = self.frames[-(self.stride_left_size + self.stride_right_size):]


    def get_next_feat(self,block,timeout):        
        return self.feat_queue.get(block,timeout)