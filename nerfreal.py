import math
import torch
import numpy as np

#from .utils import *
import subprocess
import os
import time
import torch.nn.functional as F
import cv2

from asrreal import ASR
import asyncio
from rtmp_streaming import StreamerConfig, Streamer
from av import AudioFrame, VideoFrame

class NeRFReal:
    def __init__(self, opt, trainer, data_loader, debug=True):
        self.opt = opt # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.W = opt.W
        self.H = opt.H
        self.debug = debug
        self.training = False
        self.step = 0 # training step 

        self.trainer = trainer
        self.data_loader = data_loader

        # use dataloader's bg
        bg_img = data_loader._data.bg_img #.view(1, -1, 3)
        if self.H != bg_img.shape[0] or self.W != bg_img.shape[1]:
            bg_img = F.interpolate(bg_img.permute(2, 0, 1).unsqueeze(0).contiguous(), (self.H, self.W), mode='bilinear').squeeze(0).permute(1, 2, 0).contiguous()
        self.bg_color = bg_img.view(1, -1, 3)

        # audio features (from dataloader, only used in non-playing mode)
        self.audio_features = data_loader._data.auds # [N, 29, 16]
        self.audio_idx = 0

        #self.frame_total_num = data_loader._data.end_index
        #print("frame_total_num:",self.frame_total_num)

        # control eye
        self.eye_area = None if not self.opt.exp_eye else data_loader._data.eye_area.mean().item()

        # playing seq from dataloader, or pause.
        self.playing = True #False todo
        self.loader = iter(data_loader)

        self.render_buffer = np.zeros((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True # camera moved, should reset accumulation
        self.spp = 1 # sample per pixel
        self.mode = 'image' # choose from ['image', 'depth']

        self.dynamic_resolution = False # assert False!
        self.downscale = 1
        self.train_steps = 16

        self.ind_index = 0
        self.ind_num = trainer.model.individual_codes.shape[0]

        # build asr
        if self.opt.asr:
            self.asr = ASR(opt)
        
        '''
        video_path = 'video_stream'
        if not os.path.exists(video_path):
            os.mkfifo(video_path, mode=0o777)
        audio_path = 'audio_stream'
        if not os.path.exists(audio_path):
            os.mkfifo(audio_path, mode=0o777)
        width=450
        height=450
        command = ['ffmpeg',
                    '-y', #'-an',
                    #'-re',
                    '-f', 'rawvideo',
                    '-vcodec','rawvideo',
                    '-pix_fmt', 'rgb24', #像素格式
                    '-s', "{}x{}".format(width, height),
                    '-r', str(fps),
                    '-i', video_path, 
                    '-f', 's16le',
                    '-acodec','pcm_s16le',
                    '-ac', '1',
                    '-ar', '16000',
                    '-i', audio_path,
                    #'-fflags', '+genpts',
                    '-map', '0:v',
                    '-map', '1:a',
                    #'-copyts', 
                    '-acodec', 'aac',
                    '-pix_fmt', 'yuv420p', #'-vcodec', "h264",
                    #"-rtmp_buffer", "100", 
                    '-f' , 'flv',                  
                    push_url]
        self.pipe = subprocess.Popen(command, shell=False) #, stdin=subprocess.PIPE)
        self.fifo_video = open(video_path, 'wb')
        self.fifo_audio = open(audio_path, 'wb')
        #self.test_step()
        '''
        

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.opt.asr:
            self.asr.stop()

    def push_audio(self,chunk):
        self.asr.push_audio(chunk)   
    
    def before_push_audio(self):
        self.asr.before_push_audio()   

    def prepare_buffer(self, outputs):
        if self.mode == 'image':
            return outputs['image']
        else:
            return np.expand_dims(outputs['depth'], -1).repeat(3, -1)

    def test_step(self,loop=None,audio_track=None,video_track=None):
        
        #starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        #starter.record()

        if self.playing:
            try:
                data = next(self.loader)
            except StopIteration:
                self.loader = iter(self.data_loader)
                data = next(self.loader)
            
            if self.opt.asr:
                # use the live audio stream
                data['auds'] = self.asr.get_next_feat()

            #t = time.time()
            outputs = self.trainer.test_gui_with_data(data, self.W, self.H)
            #print('-------ernerf time: ',time.time()-t)
            #print(f'[INFO] outputs shape ',outputs['image'].shape)
            image = (outputs['image'] * 255).astype(np.uint8)
            if not self.opt.fullbody:
                if self.opt.transport=='rtmp':
                    self.streamer.stream_frame(image)
                else:
                    new_frame = VideoFrame.from_ndarray(image, format="rgb24")
                    asyncio.run_coroutine_threadsafe(video_track._queue.put(new_frame), loop)
            else: #fullbody human
                #print("frame index:",data['index'])
                image_fullbody = cv2.imread(os.path.join(self.opt.fullbody_img, str(data['index'][0])+'.jpg'))
                image_fullbody = cv2.cvtColor(image_fullbody, cv2.COLOR_BGR2RGB)
                start_x = self.opt.fullbody_offset_x  # 合并后小图片的起始x坐标
                start_y = self.opt.fullbody_offset_y  # 合并后小图片的起始y坐标
                image_fullbody[start_y:start_y+image.shape[0], start_x:start_x+image.shape[1]] = image
                if self.opt.transport=='rtmp':
                    self.streamer.stream_frame(image_fullbody)
                else:
                    new_frame = VideoFrame.from_ndarray(image, format="rgb24")
                    asyncio.run_coroutine_threadsafe(video_track._queue.put(new_frame), loop)
            #self.pipe.stdin.write(image.tostring())
            for _ in range(2):
                frame = self.asr.get_audio_out()
                #print(f'[INFO] get_audio_out shape ',frame.shape)
                if self.opt.transport=='rtmp':
                    self.streamer.stream_frame_audio(frame)
                else:
                    frame = (frame * 32767).astype(np.int16)
                    new_frame = AudioFrame(format='s16', layout='mono', samples=320)
                    new_frame.planes[0].update(frame.tobytes())
                    new_frame.sample_rate=16000
                    asyncio.run_coroutine_threadsafe(audio_track._queue.put(new_frame), loop)
            #     frame = (frame * 32767).astype(np.int16).tobytes()
            #     self.fifo_audio.write(frame)           
        else:
            if self.audio_features is not None:
                auds = get_audio_features(self.audio_features, self.opt.att, self.audio_idx)
            else:
                auds = None
            outputs = self.trainer.test_gui(self.cam.pose, self.cam.intrinsics, self.W, self.H, auds, self.eye_area, self.ind_index, self.bg_color, self.spp, self.downscale)

        #ender.record()
        #torch.cuda.synchronize()
        #t = starter.elapsed_time(ender)
            
    def render(self,quit_event,loop=None,audio_track=None,video_track=None):
        if self.opt.asr:
             self.asr.warm_up()
        count=0
        totaltime=0

        if self.opt.transport=='rtmp':
            fps=25
            #push_url='rtmp://localhost/live/livestream' #'data/video/output_0.mp4'
            sc = StreamerConfig()
            sc.source_width = self.W
            sc.source_height = self.H
            sc.stream_width = self.W
            sc.stream_height = self.H
            if self.opt.fullbody:
                sc.source_width = self.opt.fullbody_width
                sc.source_height = self.opt.fullbody_height
                sc.stream_width = self.opt.fullbody_width
                sc.stream_height = self.opt.fullbody_height
            sc.stream_fps = fps
            sc.stream_bitrate = 1000000
            sc.stream_profile = 'baseline' #'high444' # 'main'
            sc.audio_channel = 1
            sc.sample_rate = 16000
            sc.stream_server = self.opt.push_url
            self.streamer = Streamer()
            self.streamer.init(sc)
            #self.streamer.enable_av_debug_log()

        while not quit_event.is_set(): #todo
            # update texture every frame
            # audio stream thread...
            t = time.time()
            if self.opt.asr and self.playing:
                # run 2 ASR steps (audio is at 50FPS, video is at 25FPS)
                for _ in range(2):
                    self.asr.run_step()
            self.test_step(loop,audio_track,video_track)
            totaltime += (time.time() - t)
            count += 1
            if count==100:
                print(f"------actual avg fps:{count/totaltime:.4f}")
                count=0
                totaltime=0
            delay = 0.04 - (time.time() - t) #40ms
            if delay > 0:
                time.sleep(delay)
            