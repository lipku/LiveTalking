###############################################################################
#  Output — 虚拟摄像头输出
###############################################################################

import queue
import threading
import time

import numpy as np
from streamout.base_output import BaseOutput
from registry import register
from utils.logger import logger
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from avatars.base_avatar import BaseAvatar


@register("streamout", "virtualcam")
class VirtualCamOutput(BaseOutput):
    """虚拟摄像头输出模式 — 通过 pyvirtualcam 输出到虚拟摄像头

    视频发送由独立线程按固定节拍驱动：
    - push_video_frame 只入队，不阻塞推理/渲染线程；
    - 发送线程每 1/fps 秒出一帧，没有新帧时重复上一帧，保证消费端
      （OBS/Zoom/微信等）拿到稳定的帧间隔，避免帧忽快忽慢造成卡顿；
    - 落后时重新对齐节拍而不是连续补帧，避免突发。
    """

    MAX_VIDEO_QUEUE = 50  # 视频帧队列上限，正常运行时靠 get_buffer_size 限流，不会积压

    def __init__(self, opt=None, parent: Optional['BaseAvatar'] = None, **kwargs):
        super().__init__(opt, parent)
        self.width = getattr(opt, 'W', 450)
        self.height = getattr(opt, 'H', 450)
        self.fps = getattr(opt, 'fps', 25)
        self.audio_output_device = getattr(opt, 'audio_output_device', None)
        self._cam = None
        self._video_queue = None
        self._video_thread = None
        self._last_frame_rgb = None
        self._audio_queue = None
        self._audio_thread = None
        self._quit_event = None

        #统计视频帧率用（发送端实际出帧率）
        self.framecount = 0
        self.lasttime = time.perf_counter()
        self.totaltime = 0
        #统计管线真实产帧率（进入输出端的帧）
        self.src_framecount = 0
        self.src_lasttime = time.perf_counter()
        self.src_totaltime = 0

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
        while not self._quit_event.is_set():
            try:
                data = self._audio_queue.get(block=True, timeout=1)
                stream.write(data)
            except queue.Empty:
                continue
        stream.close()
        p.terminate()

    def _init_cam(self, frame):
        import pyvirtualcam
        self.height, self.width = frame.shape[:2]
        self._cam = pyvirtualcam.Camera(
            width=self.width,
            height=self.height,
            fps=self.fps,
        )
        logger.info(f"VirtualCam output started: {self._cam.device} with resolution {self.width}x{self.height}")

    def _video_loop(self):
        interval = 1.0 / self.fps if self.fps > 0 else 0.04
        next_t = time.perf_counter()
        self.lasttime = time.perf_counter()
        while not self._quit_event.is_set():
            # 取下一帧；没有新帧时沿用上一帧，保证输出节奏不中断
            try:
                frame = self._video_queue.get_nowait()
            except queue.Empty:
                frame = None

            if frame is not None:
                if self._cam is None:
                    try:
                        self._init_cam(frame)
                    except Exception as e:
                        logger.error(f"[VirtualCam] init failed: {e}")
                        break
                # pyvirtualcam 需要 RGB 格式，而 cv2 渲染的是 BGR 格式，需要转换
                self._last_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self._last_frame_rgb is None:  # 首帧还没到
                if self._quit_event.wait(0.005):
                    break
                continue

            while time.perf_counter() < next_t:
                if self._quit_event.wait(next_t - time.perf_counter()):
                    break
            if self._quit_event.is_set():
                break
            try:
                self._cam.send(self._last_frame_rgb)
            except Exception as e:
                logger.error(f"[VirtualCam] send failed: {e}")
                break

            # 重复帧也计入，反映虚拟摄像头实际出帧率
            self.totaltime += (time.perf_counter() - self.lasttime)
            self.framecount += 1
            self.lasttime = time.perf_counter()
            if self.framecount == 100:
                logger.info(f"------actual avg final fps:{self.framecount/self.totaltime:.4f}")
                self.framecount = 0
                self.totaltime = 0

            next_t += interval
            # 处理线程被卡住导致落后时重新对齐节拍，避免连续补帧造成突发
            if time.perf_counter() - next_t > interval:
                next_t = time.perf_counter() + interval
        if self._cam is not None:
            self._cam.close()
            self._cam = None

    def start(self) -> None:
        """启动虚拟摄像头视频/音频发送线程，摄像头在收到首帧后初始化"""
        try:
            import pyvirtualcam
        except ImportError:
            logger.error("pyvirtualcam not installed. pip install pyvirtualcam")
            raise

        self._quit_event = threading.Event()
        self._video_queue = queue.Queue(maxsize=self.MAX_VIDEO_QUEUE)
        self._last_frame_rgb = None
        self.src_lasttime = time.perf_counter()
        self._video_thread = threading.Thread(target=self._video_loop, daemon=True, name="virtualcam_video")
        self._video_thread.start()

        # Start PyAudio playback thread
        self._audio_queue = queue.Queue(maxsize=3000)
        self._audio_thread = threading.Thread(target=self._play_audio_loop, daemon=True, name="pyaudio_stream")
        self._audio_thread.start()

    def push_video_frame(self, frame) -> None:
        if not isinstance(frame, np.ndarray) or self._video_queue is None:
            return

        # 统计管线真实产帧率（与发送端的实际出帧率区分，产帧不足时发送端会重复上一帧）
        self.src_totaltime += (time.perf_counter() - self.src_lasttime)
        self.src_framecount += 1
        self.src_lasttime = time.perf_counter()
        if self.src_framecount == 100:
            logger.info(f"------actual avg pipeline fps:{self.src_framecount/self.src_totaltime:.4f}")
            self.src_framecount = 0
            self.src_totaltime = 0

        try:
            self._video_queue.put_nowait(frame)
        except queue.Full:
            # 输出端长时间不消费（如虚拟摄像头后端卡顿）：丢弃最旧帧，保证引擎不被阻塞
            try:
                self._video_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._video_queue.put_nowait(frame)
            except queue.Full:
                pass

    def push_audio_frame(self, frame, eventpoint=None) -> None:
        if self._audio_queue:
            self._audio_queue.put(frame.tobytes())
            self.parent.notify(eventpoint)

    def get_buffer_size(self) -> int:
        # 供渲染线程限流，避免管线跑在播放前面导致音视频延迟持续增大
        return self._video_queue.qsize() if self._video_queue else 0

    def stop(self) -> None:
        if self._quit_event:
            self._quit_event.set()
        if self._video_thread:
            self._video_thread.join(timeout=2)
            self._video_thread = None
        if self._audio_thread:
            self._audio_thread.join(timeout=2)
            self._audio_thread = None
        if self._cam is not None:  # 发送线程异常退出时兜底关闭
            try:
                self._cam.close()
            except Exception:
                pass
            self._cam = None
        logger.info("VirtualCam output stopped")
