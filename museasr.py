import time
import torch
import numpy as np
import soundfile as sf
import resampy

import queue
from queue import Queue
from io import BytesIO

from musetalk.whisper.audio2feature import Audio2Feature

class MuseASR:
    def __init__(self, opt, audio_processor:Audio2Feature):
        self.opt = opt

        self.fps = opt.fps # 20 ms per frame
        self.sample_rate = 16000
        self.chunk = self.sample_rate // self.fps # 320 samples per chunk (20ms * 16000 / 1000)
        self.queue = Queue()
        self.input_stream = BytesIO()
        self.output_queue = Queue()

        self.audio_processor = audio_processor
        self.batch_size = opt.batch_size

        self.stride_left_size = self.stride_right_size = 6
        self.audio_feats = []

        self.warm_up()

    def __create_bytes_stream(self,byte_stream):
        #byte_stream=BytesIO(buffer)
        stream, sample_rate = sf.read(byte_stream) # [T*sample_rate,] float64
        print(f'[INFO]tts audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            print(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]
    
        if sample_rate != self.sample_rate and stream.shape[0]>0:
            print(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream

    def push_audio(self,buffer):
        print(f'[INFO] push_audio {len(buffer)}')
        if self.opt.tts == "xtts" or self.opt.tts == "gpt-sovits":
            if len(buffer)>0:            
                stream = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32767
                if self.opt.tts == "xtts":
                    stream = resampy.resample(x=stream, sr_orig=24000, sr_new=self.sample_rate)
                else:
                    stream = resampy.resample(x=stream, sr_orig=32000, sr_new=self.sample_rate)
                #byte_stream=BytesIO(buffer)
                #stream = self.__create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx=0
                while streamlen >= self.chunk:
                    self.queue.put(stream[idx:idx+self.chunk])
                    streamlen -= self.chunk
                    idx += self.chunk
                # if streamlen>0: #skip last frame(not 20ms)
                #     self.queue.put(stream[idx:])
        else: #edge tts
            self.input_stream.write(buffer)
            if len(buffer)<=0:
                self.input_stream.seek(0)
                stream = self.__create_bytes_stream(self.input_stream)
                streamlen = stream.shape[0]
                idx=0
                while streamlen >= self.chunk:
                    self.queue.put(stream[idx:idx+self.chunk])
                    streamlen -= self.chunk
                    idx += self.chunk
                #if streamlen>0:  #skip last frame(not 20ms)
                #    self.queue.put(stream[idx:])
                self.input_stream.seek(0)
                self.input_stream.truncate()  

    def __get_audio_frame(self):        
        try:
            frame = self.queue.get(block=False)
            type = 0
            print(f'[INFO] get frame {frame.shape}')
        except queue.Empty:
            frame = np.zeros(self.chunk, dtype=np.float32)
            type = 1

        return frame,type 

    def get_audio_out(self):  #get origin audio pcm to nerf
        return self.output_queue.get()
    
    def warm_up(self):
        frames = []
        for _ in range(self.stride_left_size + self.stride_right_size):
            audio_frame,type=self.__get_audio_frame()
            frames.append(audio_frame)
            self.output_queue.put((audio_frame,type))
        inputs = np.concatenate(frames) # [N * chunk]
        whisper_feature = self.audio_processor.audio2feat(inputs)
        for feature in whisper_feature:
            self.audio_feats.append(feature)

        for _ in range(self.stride_left_size):
            self.output_queue.get()

    def run_step(self):
        ############################################## extract audio feature ##############################################
        start_time = time.time()
        frames = []
        for _ in range(self.batch_size*2):
            audio_frame,type=self.__get_audio_frame()
            frames.append(audio_frame)
            self.output_queue.put((audio_frame,type))
        inputs = np.concatenate(frames) # [N * chunk]
        whisper_feature = self.audio_processor.audio2feat(inputs)
        for feature in whisper_feature:
            self.audio_feats.append(feature)
        
        #print(f"processing audio costs {(time.time() - start_time) * 1000}ms, inputs shape:{inputs.shape} whisper_feature len:{len(whisper_feature)}")

    def get_next_feat(self):
        whisper_chunks = self.audio_processor.feature2chunks(feature_array=self.audio_feats,fps=self.fps/2,batch_size=self.batch_size,start=self.stride_left_size/2 )
        #print(f"whisper_chunks len:{len(whisper_chunks)},self.audio_feats len:{len(self.audio_feats)},self.output_queue len:{self.output_queue.qsize()}")
        self.audio_feats = self.audio_feats[-(self.stride_left_size + self.stride_right_size):]
        return whisper_chunks