###############################################################################
#  输出模式基类 — 视频/音频输出接口
###############################################################################

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional
from numpy.typing import NDArray
import numpy as np

if TYPE_CHECKING:
    from avatars.base_avatar import BaseAvatar


class BaseOutput(ABC):
    """
    输出传输模式抽象基类。
    
    实现者需要：
    1. start(): 启动输出
    2. push_video_frame(): 推送视频帧
    3. push_audio_frame(): 推送音频帧
    4. stop(): 关闭输出
    """

    def __init__(self, opt=None, parent: Optional['BaseAvatar'] = None, **kwargs):
        self.opt = opt
        self.parent = parent

    @abstractmethod
    def start(self) -> None:
        """启动输出通道"""
        ...

    @abstractmethod
    def push_video_frame(self, frame) -> None:
        """推送视频帧"""
        ...

    @abstractmethod
    def push_audio_frame(self, frame:NDArray[np.int16], eventpoint=None) -> None:
        """推送音频帧"""
        ...


    def get_buffer_size(self) -> int:
        """获取底层发送队列的积压帧数，用于引擎降速限流"""
        return 0

    @abstractmethod
    def stop(self) -> None:
        """关闭输出通道"""
        ...
