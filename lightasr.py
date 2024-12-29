import time
import torch
import numpy as np
from baseasr import BaseASR


class LightASR(BaseASR):
    def __init__(self, opt, parent, audio_processor):
        super().__init__(opt, parent)
        self.audio_processor = audio_processor
        self.stride_left_size = 32
        self.stride_right_size = 32


    def run_step(self):
        start_time = time.time()
        
        for _ in range(self.batch_size * 2):
            audio_frame, type_ = self.get_audio_frame()
            self.frames.append(audio_frame)
            self.output_queue.put((audio_frame, type_))
        
        if len(self.frames) <= self.stride_left_size + self.stride_right_size:
            return
        
        inputs = np.concatenate(self.frames)  # [N * chunk]

        mel = self.audio_processor.get_hubert_from_16k_speech(inputs)
        mel_chunks=self.audio_processor.feature2chunks(feature_array=mel,fps=self.fps/2,batch_size=self.batch_size,start=self.stride_left_size/2)

        self.feat_queue.put(mel_chunks)
        self.frames = self.frames[-(self.stride_left_size + self.stride_right_size):]
        #print(f"Processing audio costs {(time.time() - start_time) * 1000}ms")

