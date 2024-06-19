import math
import torch
import numpy as np

# from .utils import *
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
import multiprocessing as mp

from musetalk.utils.utils import get_file_type, get_video_fps, datagen
# from musetalk.utils.preprocessing import get_landmark_and_bbox,read_imgs,coord_placeholder
from musetalk.utils.blending import get_image, get_image_prepare_material, get_image_blending
from musetalk.utils.utils import load_all_model, load_diffusion_model, load_audio_model
from ttsreal import EdgeTTS, VoitsTTS, XTTS

from museasr import MuseASR
import asyncio
from av import AudioFrame, VideoFrame

from tqdm import tqdm


def read_imgs(img_list):
    frames = []
    print('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames


def __mirror_index(size, index):
    # size = len(self.coord_list_cycle)
    turn = index // size
    res = index % size
    if turn % 2 == 0:
        return res
    else:
        return size - res - 1


def inference(render_event, batch_size, latents_out_path, audio_feat_queue, audio_out_queue, res_frame_queue,
              ):  # vae, unet, pe,timesteps

    vae, unet, pe = load_diffusion_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timesteps = torch.tensor([0], device=device)
    pe = pe.half()
    vae.vae = vae.vae.half()
    unet.model = unet.model.half()

    input_latent_list_cycle = torch.load(latents_out_path)
    length = len(input_latent_list_cycle)
    index = 0
    count = 0
    counttime = 0
    print('start inference')
    while True:
        if render_event.is_set():
            starttime = time.perf_counter()
            try:
                whisper_chunks = audio_feat_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
            is_all_silence = True
            audio_frames = []
            for _ in range(batch_size * 2):
                frame, type = audio_out_queue.get()
                audio_frames.append((frame, type))
                if type == 0:
                    is_all_silence = False
            if is_all_silence:
                for i in range(batch_size):
                    res_frame_queue.put((None, __mirror_index(length, index), audio_frames[i * 2:i * 2 + 2]))
                    index = index + 1
            else:
                # print('infer=======')
                t = time.perf_counter()
                whisper_batch = np.stack(whisper_chunks)
                latent_batch = []
                for i in range(batch_size):
                    idx = __mirror_index(length, index + i)
                    latent = input_latent_list_cycle[idx]
                    latent_batch.append(latent)
                latent_batch = torch.cat(latent_batch, dim=0)

                # for i, (whisper_batch,latent_batch) in enumerate(gen):
                audio_feature_batch = torch.from_numpy(whisper_batch)
                audio_feature_batch = audio_feature_batch.to(device=unet.device,
                                                             dtype=unet.model.dtype)
                audio_feature_batch = pe(audio_feature_batch)
                latent_batch = latent_batch.to(dtype=unet.model.dtype)
                # print('prepare time:',time.perf_counter()-t)
                # t=time.perf_counter()

                pred_latents = unet.model(latent_batch,
                                          timesteps,
                                          encoder_hidden_states=audio_feature_batch).sample
                # print('unet time:',time.perf_counter()-t)
                # t=time.perf_counter()
                recon = vae.decode_latents(pred_latents)
                # print('vae time:',time.perf_counter()-t)
                # print('diffusion len=',len(recon))
                counttime += (time.perf_counter() - t)
                count += batch_size
                # _totalframe += 1
                if count >= 100:
                    print(f"------actual avg infer fps:{count / counttime:.4f}")
                    count = 0
                    counttime = 0
                for i, res_frame in enumerate(recon):
                    # self.__pushmedia(res_frame,loop,audio_track,video_track)
                    res_frame_queue.put((res_frame, __mirror_index(length, index), audio_frames[i * 2:i * 2 + 2]))
                    index = index + 1
                # print('total batch time:',time.perf_counter()-starttime)
        else:
            time.sleep(1)
    print('musereal inference processor stop')


@torch.no_grad()
class MuseReal:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.W = opt.W
        self.H = opt.H

        self.fps = opt.fps  # 20 ms per frame

        #### musetalk
        self.avatar_id = opt.avatar_id
        self.static_img = opt.static_img
        self.video_path = ''  # video_path
        self.bbox_shift = opt.bbox_shift
        self.avatar_path = f"./data/avatars/{self.avatar_id}"
        self.full_imgs_path = f"{self.avatar_path}/full_imgs"
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.latents_out_path = f"{self.avatar_path}/latents.pt"
        self.video_out_path = f"{self.avatar_path}/vid_output/"
        self.mask_out_path = f"{self.avatar_path}/mask"
        self.mask_coords_path = f"{self.avatar_path}/mask_coords.pkl"
        self.avatar_info_path = f"{self.avatar_path}/avator_info.json"
        self.avatar_info = {
            "avatar_id": self.avatar_id,
            "video_path": self.video_path,
            "bbox_shift": self.bbox_shift
        }
        self.batch_size = opt.batch_size
        self.idx = 0
        self.res_frame_queue = mp.Queue(self.batch_size * 2)
        self.__loadmodels()
        self.__loadavatar()

        self.asr = MuseASR(opt, self.audio_processor)
        if opt.tts == "edgetts":
            self.tts = EdgeTTS(opt, self)
        elif opt.tts == "gpt-sovits":
            self.tts = VoitsTTS(opt, self)
        elif opt.tts == "xtts":
            self.tts = XTTS(opt, self)
        # self.__warm_up()

        self.render_event = mp.Event()
        mp.Process(target=inference, args=(self.render_event, self.batch_size, self.latents_out_path,
                                           self.asr.feat_queue, self.asr.output_queue, self.res_frame_queue,
                                           )).start()  # self.vae, self.unet, self.pe,self.timesteps

    def __loadmodels(self):
        # load model weights
        self.audio_processor = load_audio_model()
        # self.audio_processor, self.vae, self.unet, self.pe = load_all_model()
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.timesteps = torch.tensor([0], device=device)
        # self.pe = self.pe.half()
        # self.vae.vae = self.vae.vae.half()
        # self.unet.model = self.unet.model.half()

    def __loadavatar(self):
        # self.input_latent_list_cycle = torch.load(self.latents_out_path)
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

    def put_msg_txt(self, msg):
        self.tts.put_msg_txt(msg)

    def put_audio_frame(self, audio_chunk):  # 16khz 20ms pcm
        self.asr.put_audio_frame(audio_chunk)

    def __mirror_index(self, index):
        size = len(self.coord_list_cycle)
        turn = index // size
        res = index % size
        if turn % 2 == 0:
            return res
        else:
            return size - res - 1

    def __warm_up(self):
        self.asr.run_step()
        whisper_chunks = self.asr.get_next_feat()
        whisper_batch = np.stack(whisper_chunks)
        latent_batch = []
        for i in range(self.batch_size):
            idx = self.__mirror_index(self.idx + i)
            latent = self.input_latent_list_cycle[idx]
            latent_batch.append(latent)
        latent_batch = torch.cat(latent_batch, dim=0)
        print('infer=======')
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

    def process_frames(self, quit_event, loop=None, audio_track=None, video_track=None):

        while not quit_event.is_set():
            try:
                res_frame, idx, audio_frames = self.res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
            if audio_frames[0][1] == 1 and audio_frames[1][1] == 1:  # 全为静音数据，只需要取fullimg
                if self.static_img:
                    combine_frame = self.frame_list_cycle[0]
                else:
                    combine_frame = self.frame_list_cycle[idx]
            else:
                bbox = self.coord_list_cycle[idx]
                ori_frame = copy.deepcopy(self.frame_list_cycle[idx])
                x1, y1, x2, y2 = bbox
                try:
                    res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                except:
                    continue
                mask = self.mask_list_cycle[idx]
                mask_crop_box = self.mask_coords_list_cycle[idx]
                # combine_frame = get_image(ori_frame,res_frame,bbox)
                # t=time.perf_counter()
                combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)
                # print('blending time:',time.perf_counter()-t)

            image = combine_frame  # (outputs['image'] * 255).astype(np.uint8)
            new_frame = VideoFrame.from_ndarray(image, format="bgr24")
            asyncio.run_coroutine_threadsafe(video_track._queue.put(new_frame), loop)

            for audio_frame in audio_frames:
                frame, type = audio_frame
                frame = (frame * 32767).astype(np.int16)
                new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
                new_frame.planes[0].update(frame.tobytes())
                new_frame.sample_rate = 16000
                # if audio_track._queue.qsize()>10:
                #     time.sleep(0.1)
                asyncio.run_coroutine_threadsafe(audio_track._queue.put(new_frame), loop)
        print('musereal process_frames thread stop')

    def render(self, quit_event, loop=None, audio_track=None, video_track=None):
        # if self.opt.asr:
        #     self.asr.warm_up()

        self.tts.render(quit_event)
        process_thread = Thread(target=self.process_frames, args=(quit_event, loop, audio_track, video_track))
        process_thread.start()

        self.render_event.set()  # start infer process render
        count = 0
        totaltime = 0
        _starttime = time.perf_counter()
        # _totalframe=0
        while not quit_event.is_set():  # todo
            # update texture every frame
            # audio stream thread...
            t = time.perf_counter()
            self.asr.run_step()
            # self.test_step(loop,audio_track,video_track)
            # totaltime += (time.perf_counter() - t)
            # count += self.opt.batch_size
            # if count>=100:
            #     print(f"------actual avg infer fps:{count/totaltime:.4f}")
            #     count=0
            #     totaltime=0
            if video_track._queue.qsize() >= 2 * self.opt.batch_size:
                print('sleep qsize=', video_track._queue.qsize())
                time.sleep(0.04 * self.opt.batch_size * 1.5)

            # delay = _starttime+_totalframe*0.04-time.perf_counter() #40ms
            # if delay > 0:
            #     time.sleep(delay)
        self.render_event.clear()  # end infer process render
        print('musereal thread stop')
