###############################################################################
#  Output — 虚拟摄像头输出
###############################################################################

import numpy as np
from streamout.base_output import BaseOutput
from registry import register
from utils.logger import logger
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from avatars.base_avatar import BaseAvatar


@register("streamout", "virtualcam")
class VirtualCamOutput(BaseOutput):
    """虚拟摄像头输出模式 — 通过 pyvirtualcam 输出到虚拟摄像头"""

    def __init__(self, opt=None, parent: Optional['BaseAvatar'] = None, **kwargs):
        super().__init__(opt, parent)
        self.width = getattr(opt, 'W', 450)
        self.height = getattr(opt, 'H', 450)
        self.fps = getattr(opt, 'fps', 25)
        self.audio_output_device = getattr(opt, 'audio_output_device', None)
        self._cam = None
        self._audio_queue = None
        self._audio_thread = None
        self._quit_event = None

    def _play_audio_loop(self):
        import pyaudio
        p = pyaudio.PyAudio()

        # 使用用户指定的设备索引，或自动选择默认设备
        if self.audio_output_device is not None:
            output_device_index = self.audio_output_device
            try:
                device_info = p.get_device_info_by_index(output_device_index)
                logger.info(f"[VirtualCam Audio] Using specified device: {device_info['name']} (index {output_device_index})")
            except:
                logger.warning(f"[VirtualCam Audio] Device index {output_device_index} not found, using default")
                output_device_index = None
        else:
            # 获取默认输出设备
            try:
                default_output_info = p.get_default_output_device_info()
                output_device_index = default_output_info['index']
                logger.info(f"[VirtualCam Audio] Using default output device: {default_output_info['name']} (index {output_device_index})")
            except:
                # 如果无法获取默认设备，使用 None 让 PyAudio 自动选择
                output_device_index = None
                logger.warning("[VirtualCam Audio] Cannot get default output device, using system default")

        stream = p.open(
            rate=16000,
            channels=1,
            format=8,
            output=True,
            output_device_index=output_device_index,
        )
        stream.start_stream()
        import queue
        while not self._quit_event.is_set():
            try:
                data = self._audio_queue.get(block=True, timeout=1)
                stream.write(data)
            except queue.Empty:
                continue
        stream.close()
        p.terminate()

    def start(self) -> None:
        """启动虚拟摄像头音频线程，视频流延迟到第一帧接收时初始化"""
        try:
            import pyvirtualcam
            # 仅仅验证包是否安装，延迟实例化 Camera
            
            # Start PyAudio playback thread
            import queue
            from threading import Thread, Event
            self._audio_queue = queue.Queue(maxsize=3000)
            self._quit_event = Event()
            self._audio_thread = Thread(target=self._play_audio_loop, daemon=True, name="pyaudio_stream")
            self._audio_thread.start()
        except ImportError:
            logger.error("pyvirtualcam not installed. pip install pyvirtualcam")
            raise

    def push_video_frame(self, frame) -> None:
        if isinstance(frame, np.ndarray):
            if self._cam is None:
                import pyvirtualcam
                self.height, self.width = frame.shape[:2]
                self._cam = pyvirtualcam.Camera(
                    width=self.width,
                    height=self.height,
                    fps=self.fps,
                )
                logger.info(f"VirtualCam output started: {self._cam.device} with resolution {self.width}x{self.height}")

            # pyvirtualcam 需要 RGB 格式，而 cv2 渲染的是 BGR 格式，需要转换
            import cv2
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._cam.send(frame_rgb)
            self._cam.sleep_until_next_frame()

    def push_audio_frame(self, frame, eventpoint=None) -> None:
        if self._audio_queue:
            self._audio_queue.put(frame.tobytes())
            self.parent.notify(eventpoint)

    def stop(self) -> None:
        if self._quit_event:
            self._quit_event.set()
        if self._audio_thread:
            self._audio_thread.join()
        if self._cam:
            self._cam.close()
            self._cam = None
            logger.info("VirtualCam output stopped")
