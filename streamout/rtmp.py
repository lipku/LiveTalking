###############################################################################
#  Output — RTMP 推流输出
###############################################################################

import subprocess
import time
import numpy as np
from streamout.base_output import BaseOutput
from registry import register
from utils.logger import logger
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from avatars.base_avatar import BaseAvatar


@register("streamout", "rtmp")
class RTMPOutput(BaseOutput):
    """RTMP 推流输出模式 — 基于 python_rtmpstream 库推送音视频"""

    def __init__(self, opt=None, parent: Optional['BaseAvatar'] = None, **kwargs):
        super().__init__(opt, parent)
        self.push_url = getattr(opt, 'push_url', 'rtmp://localhost/live/livestream')
        self.width = getattr(opt, 'W', 450)
        self.height = getattr(opt, 'H', 450)
        self.fps = getattr(opt, 'fps', 25)
        self.bitrate = getattr(opt, 'bitrate', 1000000)
        self._streamer = None

        #统计视频帧率用
        self.framecount = 0
        self.lasttime = time.perf_counter()
        self.totaltime = 0

    def start(self) -> None:
        """Streamer 延迟到第一帧视频到达时再根据实际宽高初始化"""
        import queue
        self._audio_queue = queue.Queue()
        self._quit_event = False

    def _init_streamer(self, frame_height, frame_width):
        try:
            from rtmp_streaming import StreamerConfig, Streamer
        except ImportError:
            logger.error("rtmp_streaming is not installed. Please install python_rtmpstream.")
            raise

        sc = StreamerConfig()
        sc.source_width = frame_width
        sc.source_height = frame_height
        sc.stream_width = frame_width
        sc.stream_height = frame_height
        sc.stream_fps = self.fps
        sc.stream_bitrate = self.bitrate
        sc.stream_profile = 'main'
        sc.audio_channel = 1
        
        sc.sample_rate = getattr(self.opt, 'sample_rate', 16000)
        if self.parent:
            sc.sample_rate = self.parent.sample_rate
            
        sc.stream_server = self.push_url

        self._streamer = Streamer()
        self._streamer.init(sc)

        self._starttime=time.perf_counter()
        self._totalframe=0
        logger.info(f"RTMP output started via python_rtmpstream: {self.push_url} with resolution {frame_width}x{frame_height}")

    def push_video_frame(self, frame) -> None:
        if isinstance(frame, np.ndarray):
            if self._streamer is None:
                self.height, self.width = frame.shape[:2]
                self._init_streamer(self.height, self.width)
                
                # Consume any buffered audio that arrived before the first video frame
                while not self._audio_queue.empty():
                    buffered_audio = self._audio_queue.get()
                    self._streamer.stream_frame_audio(buffered_audio)
                    
            import cv2
            # Convert BGR (OpenCV) to RGB since Streamer expects RGB memory layout
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._streamer.stream_frame(rgb_frame)

            delay = self._starttime+self._totalframe*0.04-time.perf_counter() #40ms
            if delay > 0:
                time.sleep(delay)
            self._totalframe += 1

            self.totaltime += (time.perf_counter() - self.lasttime)
            self.framecount += 1
            self.lasttime = time.perf_counter()
            if self.framecount==100:
                logger.info(f"------actual avg final fps:{self.framecount/self.totaltime:.4f}")
                self.framecount = 0
                self.totaltime=0

    def push_audio_frame(self, frame, eventpoint=None) -> None:
        if isinstance(frame, np.ndarray):
            # The upstream pipeline typically passes np.int16 (after multiplying by 32767).
            # The python_rtmpstream bindings expect np.float32 for AV_SAMPLE_FMT_FLTP.
            if frame.dtype == np.int16:
                frame = frame.astype(np.float32) / 32767.0
            
            if self._streamer:
                self._streamer.stream_frame_audio(frame)
                self.parent.notify(eventpoint)
            else:
                self._audio_queue.put(frame)

    def stop(self) -> None:
        self._quit_event = True
        self._streamer = None
        logger.info("RTMP output stopped")
