import time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCTC, AutoProcessor

import pyaudio
import soundfile as sf
import resampy

from queue import Queue
from threading import Thread, Event


def _read_frame(stream, exit_event, queue, chunk):

    while True:
        if exit_event.is_set():
            print(f'[INFO] read frame thread ends')
            break
        frame = stream.read(chunk, exception_on_overflow=False)
        frame = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32767 # [chunk]
        queue.put(frame)

def _play_frame(stream, exit_event, queue, chunk):

    while True:
        if exit_event.is_set():
            print(f'[INFO] play frame thread ends')
            break
        frame = queue.get()
        frame = (frame * 32767).astype(np.int16).tobytes()
        stream.write(frame, chunk)

class ASR:
    def __init__(self, opt):

        self.opt = opt

        self.play = opt.asr_play

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.fps = opt.fps # 20 ms per frame
        self.sample_rate = 16000
        self.chunk = self.sample_rate // self.fps # 320 samples per chunk (20ms * 16000 / 1000)
        self.mode = 'live' if opt.asr_wav == '' else 'file'

        if 'esperanto' in self.opt.asr_model:
            self.audio_dim = 44
        elif 'deepspeech' in self.opt.asr_model:
            self.audio_dim = 29
        else:
            self.audio_dim = 32

        # prepare context cache
        # each segment is (stride_left + ctx + stride_right) * 20ms, latency should be (ctx + stride_right) * 20ms
        self.context_size = opt.m
        self.stride_left_size = opt.l
        self.stride_right_size = opt.r
        self.text = '[START]\n'
        self.terminated = False
        self.frames = []

        # pad left frames
        if self.stride_left_size > 0:
            self.frames.extend([np.zeros(self.chunk, dtype=np.float32)] * self.stride_left_size)


        self.exit_event = Event()
        self.audio_instance = pyaudio.PyAudio()

        # create input stream
        if self.mode == 'file':
            self.file_stream = self.create_file_stream()
        else:
            # start a background process to read frames
            self.input_stream = self.audio_instance.open(format=pyaudio.paInt16, channels=1, rate=self.sample_rate, input=True, output=False, frames_per_buffer=self.chunk)
            self.queue = Queue()
            self.process_read_frame = Thread(target=_read_frame, args=(self.input_stream, self.exit_event, self.queue, self.chunk))
        
        # play out the audio too...?
        if self.play:
            self.output_stream = self.audio_instance.open(format=pyaudio.paInt16, channels=1, rate=self.sample_rate, input=False, output=True, frames_per_buffer=self.chunk)
            self.output_queue = Queue()
            self.process_play_frame = Thread(target=_play_frame, args=(self.output_stream, self.exit_event, self.output_queue, self.chunk))

        # current location of audio
        self.idx = 0

        # create wav2vec model
        print(f'[INFO] loading ASR model {self.opt.asr_model}...')
        self.processor = AutoProcessor.from_pretrained(opt.asr_model)
        self.model = AutoModelForCTC.from_pretrained(opt.asr_model).to(self.device)

        # prepare to save logits
        if self.opt.asr_save_feats:
            self.all_feats = []

        # the extracted features 
        # use a loop queue to efficiently record endless features: [f--t---][-------][-------]
        self.feat_buffer_size = 4
        self.feat_buffer_idx = 0
        self.feat_queue = torch.zeros(self.feat_buffer_size * self.context_size, self.audio_dim, dtype=torch.float32, device=self.device)

        # TODO: hard coded 16 and 8 window size...
        self.front = self.feat_buffer_size * self.context_size - 8 # fake padding
        self.tail = 8
        # attention window...
        self.att_feats = [torch.zeros(self.audio_dim, 16, dtype=torch.float32, device=self.device)] * 4 # 4 zero padding...

        # warm up steps needed: mid + right + window_size + attention_size
        self.warm_up_steps = self.context_size + self.stride_right_size + 8 + 2 * 3

        self.listening = False
        self.playing = False

    def listen(self):
        # start
        if self.mode == 'live' and not self.listening:
            print(f'[INFO] starting read frame thread...')
            self.process_read_frame.start()
            self.listening = True
        
        if self.play and not self.playing:
            print(f'[INFO] starting play frame thread...')
            self.process_play_frame.start()
            self.playing = True

    def stop(self):

        self.exit_event.set()

        if self.play:
            self.output_stream.stop_stream()
            self.output_stream.close()
            if self.playing:
                self.process_play_frame.join()
                self.playing = False

        if self.mode == 'live':
            self.input_stream.stop_stream()
            self.input_stream.close()
            if self.listening:
                self.process_read_frame.join()
                self.listening = False


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        
        self.stop()

        if self.mode == 'live':
            # live mode: also print the result text.        
            self.text += '\n[END]'
            print(self.text)

    def get_next_feat(self):
        # return a [1/8, 16] window, for the next input to nerf side.
        
        while len(self.att_feats) < 8:
            # [------f+++t-----]
            if self.front < self.tail:
                feat = self.feat_queue[self.front:self.tail]
            # [++t-----------f+]
            else:
                feat = torch.cat([self.feat_queue[self.front:], self.feat_queue[:self.tail]], dim=0)

            self.front = (self.front + 2) % self.feat_queue.shape[0]
            self.tail = (self.tail + 2) % self.feat_queue.shape[0]

            # print(self.front, self.tail, feat.shape)

            self.att_feats.append(feat.permute(1, 0))
        
        att_feat = torch.stack(self.att_feats, dim=0) # [8, 44, 16]

        # discard old
        self.att_feats = self.att_feats[1:]

        return att_feat

    def run_step(self):

        if self.terminated:
            return

        # get a frame of audio
        frame = self.get_audio_frame()
        
        # the last frame
        if frame is None:
            # terminate, but always run the network for the left frames
            self.terminated = True
        else:
            self.frames.append(frame)
            # put to output
            if self.play:
                self.output_queue.put(frame)
            # context not enough, do not run network.
            if len(self.frames) < self.stride_left_size + self.context_size + self.stride_right_size:
                return
        
        inputs = np.concatenate(self.frames) # [N * chunk]

        # discard the old part to save memory
        if not self.terminated:
            self.frames = self.frames[-(self.stride_left_size + self.stride_right_size):]

        logits, labels, text = self.frame_to_text(inputs)
        feats = logits # better lips-sync than labels

        # save feats
        if self.opt.asr_save_feats:
            self.all_feats.append(feats)

        # record the feats efficiently.. (no concat, constant memory)
        start = self.feat_buffer_idx * self.context_size
        end = start + feats.shape[0]
        self.feat_queue[start:end] = feats
        self.feat_buffer_idx = (self.feat_buffer_idx + 1) % self.feat_buffer_size

        # very naive, just concat the text output.
        if text != '':
            self.text = self.text + ' ' + text

        # will only run once at ternimation
        if self.terminated:
            self.text += '\n[END]'
            print(self.text)
            if self.opt.asr_save_feats:
                print(f'[INFO] save all feats for training purpose... ')
                feats = torch.cat(self.all_feats, dim=0) # [N, C]
                # print('[INFO] before unfold', feats.shape)
                window_size = 16
                padding = window_size // 2
                feats = feats.view(-1, self.audio_dim).permute(1, 0).contiguous() # [C, M]
                feats = feats.view(1, self.audio_dim, -1, 1) # [1, C, M, 1]
                unfold_feats = F.unfold(feats, kernel_size=(window_size, 1), padding=(padding, 0), stride=(2, 1)) # [1, C * window_size, M / 2 + 1]
                unfold_feats = unfold_feats.view(self.audio_dim, window_size, -1).permute(2, 1, 0).contiguous() # [C, window_size, M / 2 + 1] --> [M / 2 + 1, window_size, C]
                # print('[INFO] after unfold', unfold_feats.shape)
                # save to a npy file
                if 'esperanto' in self.opt.asr_model:
                    output_path = self.opt.asr_wav.replace('.wav', '_eo.npy')
                else:
                    output_path = self.opt.asr_wav.replace('.wav', '.npy')
                np.save(output_path, unfold_feats.cpu().numpy())
                print(f"[INFO] saved logits to {output_path}")
    
    def create_file_stream(self):
    
        stream, sample_rate = sf.read(self.opt.asr_wav) # [T*sample_rate,] float64
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            print(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]
    
        if sample_rate != self.sample_rate:
            print(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        print(f'[INFO] loaded audio stream {self.opt.asr_wav}: {stream.shape}')

        return stream


    def create_pyaudio_stream(self):

        import pyaudio

        print(f'[INFO] creating live audio stream ...')

        audio = pyaudio.PyAudio()
        
        # get devices
        info = audio.get_host_api_info_by_index(0)
        n_devices = info.get('deviceCount')

        for i in range(0, n_devices):
            if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                name = audio.get_device_info_by_host_api_device_index(0, i).get('name')
                print(f'[INFO] choose audio device {name}, id {i}')
                break
        
        # get stream
        stream = audio.open(input_device_index=i,
                            format=pyaudio.paInt16,
                            channels=1,
                            rate=self.sample_rate,
                            input=True,
                            frames_per_buffer=self.chunk)
        
        return audio, stream

    
    def get_audio_frame(self):

        if self.mode == 'file':

            if self.idx < self.file_stream.shape[0]:
                frame = self.file_stream[self.idx: self.idx + self.chunk]
                self.idx = self.idx + self.chunk
                return frame
            else:
                return None
        
        else:

            frame = self.queue.get()
            # print(f'[INFO] get frame {frame.shape}')

            self.idx = self.idx + self.chunk

            return frame

        
    def frame_to_text(self, frame):
        # frame: [N * 320], N = (context_size + 2 * stride_size)
        
        inputs = self.processor(frame, sampling_rate=self.sample_rate, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            result = self.model(inputs.input_values.to(self.device))
            logits = result.logits # [1, N - 1, 32]
        
        # cut off stride
        left = max(0, self.stride_left_size)
        right = min(logits.shape[1], logits.shape[1] - self.stride_right_size + 1) # +1 to make sure output is the same length as input.

        # do not cut right if terminated.
        if self.terminated:
            right = logits.shape[1]

        logits = logits[:, left:right]

        # print(frame.shape, inputs.input_values.shape, logits.shape)
    
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0].lower()

        
        # for esperanto
        # labels = np.array(['ŭ', '»', 'c', 'ĵ', 'ñ', '”', '„', '“', 'ǔ', 'o', 'ĝ', 'm', 'k', 'd', 'a', 'ŝ', 'z', 'i', '«', '—', '‘', 'ĥ', 'f', 'y', 'h', 'j', '|', 'r', 'u', 'ĉ', 's', '–', 'ﬁ', 'l', 'p', '’', 'g', 'v', 't', 'b', 'n', 'e', '[UNK]', '[PAD]'])

        # labels = np.array([' ', ' ', ' ', '-', '|', 'E', 'T', 'A', 'O', 'N', 'I', 'H', 'S', 'R', 'D', 'L', 'U', 'M', 'W', 'C', 'F', 'G', 'Y', 'P', 'B', 'V', 'K', "'", 'X', 'J', 'Q', 'Z'])
        # print(''.join(labels[predicted_ids[0].detach().cpu().long().numpy()]))
        # print(predicted_ids[0])
        # print(transcription)

        return logits[0], predicted_ids[0], transcription # [N,]


    def run(self):

        self.listen()

        while not self.terminated:
            self.run_step()

    def clear_queue(self):
        # clear the queue, to reduce potential latency...
        print(f'[INFO] clear queue')
        if self.mode == 'live':
            self.queue.queue.clear()
        if self.play:
            self.output_queue.queue.clear()

    def warm_up(self):

        self.listen()
        
        print(f'[INFO] warm up ASR live model, expected latency = {self.warm_up_steps / self.fps:.6f}s')
        t = time.time()
        for _ in range(self.warm_up_steps):
            self.run_step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.time() - t
        print(f'[INFO] warm-up done, actual latency = {t:.6f}s')

        self.clear_queue()

            


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--wav', type=str, default='')
    parser.add_argument('--play', action='store_true', help="play out the audio")
    
    parser.add_argument('--model', type=str, default='cpierse/wav2vec2-large-xlsr-53-esperanto')
    # parser.add_argument('--model', type=str, default='facebook/wav2vec2-large-960h-lv60-self')

    parser.add_argument('--save_feats', action='store_true')
    # audio FPS
    parser.add_argument('--fps', type=int, default=50)
    # sliding window left-middle-right length.
    parser.add_argument('-l', type=int, default=10)
    parser.add_argument('-m', type=int, default=50)
    parser.add_argument('-r', type=int, default=10)
    
    opt = parser.parse_args()

    # fix
    opt.asr_wav = opt.wav
    opt.asr_play = opt.play
    opt.asr_model = opt.model
    opt.asr_save_feats = opt.save_feats

    if 'deepspeech' in opt.asr_model:
        raise ValueError("DeepSpeech features should not use this code to extract...")

    with ASR(opt) as asr:
        asr.run()