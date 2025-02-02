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

import time
import numpy as np

import queue
from queue import Queue
#import multiprocessing as mp
from baseasr import BaseASR
from musetalk.whisper.audio2feature import Audio2Feature

class MuseASR(BaseASR):
    def __init__(self, opt, parent,audio_processor:Audio2Feature):
        super().__init__(opt,parent)
        self.audio_processor = audio_processor

    def run_step(self):
        ############################################## extract audio feature ##############################################
        start_time = time.time()
        for _ in range(self.batch_size*2):
            audio_frame,type,eventpoint = self.get_audio_frame()
            self.frames.append(audio_frame)
            self.output_queue.put((audio_frame,type,eventpoint))
        
        if len(self.frames) <= self.stride_left_size + self.stride_right_size:
            return
        
        inputs = np.concatenate(self.frames) # [N * chunk]
        whisper_feature = self.audio_processor.audio2feat(inputs)
        # for feature in whisper_feature:
        #     self.audio_feats.append(feature)        
        #print(f"processing audio costs {(time.time() - start_time) * 1000}ms, inputs shape:{inputs.shape} whisper_feature len:{len(whisper_feature)}")
        whisper_chunks = self.audio_processor.feature2chunks(feature_array=whisper_feature,fps=self.fps/2,batch_size=self.batch_size,start=self.stride_left_size/2 )
        #print(f"whisper_chunks len:{len(whisper_chunks)},self.audio_feats len:{len(self.audio_feats)},self.output_queue len:{self.output_queue.qsize()}")
        #self.audio_feats = self.audio_feats[-(self.stride_left_size + self.stride_right_size):]
        self.feat_queue.put(whisper_chunks)
        # discard the old part to save memory
        self.frames = self.frames[-(self.stride_left_size + self.stride_right_size):]
