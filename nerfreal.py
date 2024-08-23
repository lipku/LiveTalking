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

from nerfasr import NerfASR
from ttsreal import EdgeTTS,VoitsTTS,XTTS

import asyncio
from av import AudioFrame, VideoFrame
from basereal import BaseReal

from tqdm import tqdm
def read_imgs(img_list):
    frames = []
    print('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

class NeRFReal(BaseReal):
    def __init__(self, opt, trainer, data_loader, debug=True):
        super().__init__(opt)
        #self.opt = opt # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.W = opt.W
        self.H = opt.H

        self.trainer = trainer
        self.data_loader = data_loader

        # use dataloader's bg
        #bg_img = data_loader._data.bg_img #.view(1, -1, 3)
        #if self.H != bg_img.shape[0] or self.W != bg_img.shape[1]:
        #    bg_img = F.interpolate(bg_img.permute(2, 0, 1).unsqueeze(0).contiguous(), (self.H, self.W), mode='bilinear').squeeze(0).permute(1, 2, 0).contiguous()
        #self.bg_color = bg_img.view(1, -1, 3)

        # audio features (from dataloader, only used in non-playing mode)
        #self.audio_features = data_loader._data.auds # [N, 29, 16]
        #self.audio_idx = 0

        #self.frame_total_num = data_loader._data.end_index
        #print("frame_total_num:",self.frame_total_num)

        # control eye
        #self.eye_area = None if not self.opt.exp_eye else data_loader._data.eye_area.mean().item()

        # playing seq from dataloader, or pause.
        self.loader = iter(data_loader)
        frame_total_num = data_loader._data.end_index
        if opt.fullbody:
            input_img_list = glob.glob(os.path.join(self.opt.fullbody_img, '*.[jpJP][pnPN]*[gG]'))
            input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            #print('input_img_list:',input_img_list)
            self.fullbody_list_cycle = read_imgs(input_img_list[:frame_total_num])

        #self.render_buffer = np.zeros((self.W, self.H, 3), dtype=np.float32)
        #self.need_update = True # camera moved, should reset accumulation
        #self.spp = 1 # sample per pixel
        #self.mode = 'image' # choose from ['image', 'depth']

        #self.dynamic_resolution = False # assert False!
        #self.downscale = 1
        #self.train_steps = 16

        #self.ind_index = 0
        #self.ind_num = trainer.model.individual_codes.shape[0]

        #self.customimg_index = 0

        # build asr
        self.asr = NerfASR(opt,self)
        self.asr.warm_up()
        
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

    def put_msg_txt(self,msg):
        self.tts.put_msg_txt(msg)

    def put_audio_frame(self,audio_chunk): #16khz 20ms pcm
        self.asr.put_audio_frame(audio_chunk)

    def pause_talk(self):
        self.tts.pause_talk()
        self.asr.pause_talk()   
    

    # def mirror_index(self, index):
    #     size = self.opt.customvideo_imgnum
    #     turn = index // size
    #     res = index % size
    #     if turn % 2 == 0:
    #         return res
    #     else:
    #         return size - res - 1   

    def test_step(self,loop=None,audio_track=None,video_track=None):
        
        #starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        #starter.record()

        try:
            data = next(self.loader)
        except StopIteration:
            self.loader = iter(self.data_loader)
            data = next(self.loader)
        
        if self.opt.asr:
            # use the live audio stream
            data['auds'] = self.asr.get_next_feat()

        audiotype1 = 0
        audiotype2 = 0
        #send audio
        for i in range(2):
            frame,type = self.asr.get_audio_out()
            if i==0:
                audiotype1 = type
            else:
                audiotype2 = type
            #print(f'[INFO] get_audio_out shape ',frame.shape)
            if self.opt.transport=='rtmp':                
                self.streamer.stream_frame_audio(frame)
            else: #webrtc
                frame = (frame * 32767).astype(np.int16)
                new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
                new_frame.planes[0].update(frame.tobytes())
                new_frame.sample_rate=16000
                asyncio.run_coroutine_threadsafe(audio_track._queue.put(new_frame), loop)

        # if self.opt.transport=='rtmp':
        #     for _ in range(2):
        #         frame,type = self.asr.get_audio_out()
        #         audiotype += type
        #         #print(f'[INFO] get_audio_out shape ',frame.shape)                
        #         self.streamer.stream_frame_audio(frame)
        # else: #webrtc
        #     for _ in range(2):
        #         frame,type = self.asr.get_audio_out()
        #         audiotype += type
        #         frame = (frame * 32767).astype(np.int16)
        #         new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
        #         new_frame.planes[0].update(frame.tobytes())
        #         new_frame.sample_rate=16000
        #         # if audio_track._queue.qsize()>10:
        #         #     time.sleep(0.1)
        #         asyncio.run_coroutine_threadsafe(audio_track._queue.put(new_frame), loop)  
        #t = time.time()
        if audiotype1!=0 and audiotype2!=0 and self.custom_index.get(audiotype1) is not None: #不为推理视频并且有自定义视频
            mirindex = self.mirror_index(len(self.custom_img_cycle[audiotype1]),self.custom_index[audiotype1])
            #imgindex  = self.mirror_index(self.customimg_index)
            #print('custom img index:',imgindex)
            #image = cv2.imread(os.path.join(self.opt.customvideo_img, str(int(imgindex))+'.png'))
            image = self.custom_img_cycle[audiotype1][mirindex]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.custom_index[audiotype1] += 1
            if self.opt.transport=='rtmp':
                self.streamer.stream_frame(image)
            else:
                new_frame = VideoFrame.from_ndarray(image, format="rgb24")
                asyncio.run_coroutine_threadsafe(video_track._queue.put(new_frame), loop)
        else: #推理视频+贴回
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
                #image_fullbody = cv2.imread(os.path.join(self.opt.fullbody_img, str(data['index'][0])+'.jpg'))
                image_fullbody = self.fullbody_list_cycle[data['index'][0]]
                image_fullbody = cv2.cvtColor(image_fullbody, cv2.COLOR_BGR2RGB)
                start_x = self.opt.fullbody_offset_x  # 合并后小图片的起始x坐标
                start_y = self.opt.fullbody_offset_y  # 合并后小图片的起始y坐标
                image_fullbody[start_y:start_y+image.shape[0], start_x:start_x+image.shape[1]] = image
                if self.opt.transport=='rtmp':
                    self.streamer.stream_frame(image_fullbody)
                else:
                    new_frame = VideoFrame.from_ndarray(image_fullbody, format="rgb24")
                    asyncio.run_coroutine_threadsafe(video_track._queue.put(new_frame), loop)
            #self.pipe.stdin.write(image.tostring())        
       
        #ender.record()
        #torch.cuda.synchronize()
        #t = starter.elapsed_time(ender)
            
    def render(self,quit_event,loop=None,audio_track=None,video_track=None):
        #if self.opt.asr:
        #     self.asr.warm_up()
        
        self.init_customindex()

        if self.opt.transport=='rtmp':
            from rtmp_streaming import StreamerConfig, Streamer
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

        count=0
        totaltime=0
        _starttime=time.perf_counter()
        _totalframe=0

        self.tts.render(quit_event)
        while not quit_event.is_set(): #todo
            # update texture every frame
            # audio stream thread...
            t = time.perf_counter()
            # run 2 ASR steps (audio is at 50FPS, video is at 25FPS)
            for _ in range(2):
                self.asr.run_step()
            self.test_step(loop,audio_track,video_track)
            totaltime += (time.perf_counter() - t)
            count += 1
            _totalframe += 1
            if count==100:
                print(f"------actual avg infer fps:{count/totaltime:.4f}")
                count=0
                totaltime=0
            if self.opt.transport=='rtmp':
                delay = _starttime+_totalframe*0.04-time.perf_counter() #40ms
                if delay > 0:
                    time.sleep(delay)
            else:
                if video_track._queue.qsize()>=5:
                    #print('sleep qsize=',video_track._queue.qsize())
                    time.sleep(0.04*video_track._queue.qsize()*0.8)
        print('nerfreal thread stop')
            
            