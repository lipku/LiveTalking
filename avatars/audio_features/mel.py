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
#  Mel 频谱音频特征提取 — 用于 Wav2Lip
#  迁移自 lipasr.py
#

import time
import torch
import numpy as np

import queue
from queue import Queue

from avatars.audio_features.base_asr import BaseASR
from avatars.wav2lip import audio

class MelASR(BaseASR):

    def run_step(self):
        ############################################## extract audio feature ##############################################
        # get a frame of audio
        for _ in range(self.batch_size*2):
            audioframe = self.get_audio_frame()
            self.frames.append(audioframe.data)
            # put to output
            self.output_queue.put(audioframe)
        # context not enough, do not run network.
        if len(self.frames) <= self.stride_left_size + self.stride_right_size:
            return
        
        inputs = np.concatenate(self.frames) # [N * chunk]
        mel = audio.melspectrogram(inputs)
        #print(mel.shape[0],mel.shape,len(mel[0]),len(self.frames))
        # cut off stride
        left = max(0, self.stride_left_size*80/50)
        right = min(len(mel[0]), len(mel[0]) - self.stride_right_size*80/50)
        mel_idx_multiplier = 80./self.fps # 80 is the hop length of mel spectrogram, so that each 40ms corresponds to 3.2 mel frames. (80/25=3.2)
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
