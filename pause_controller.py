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

"""
改进的暂停控制模块
提供线程安全的暂停/恢复功能，支持多种暂停模式和状态管理
"""

import threading
import time
import queue
import numpy as np
from enum import Enum
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass, field
from logger import logger


class PauseError(Exception):
    """暂停控制相关错误的基类"""
    pass


class DeadlockError(PauseError):
    """死锁错误"""
    pass


class BufferOverflowError(PauseError):
    """缓冲区溢出错误"""
    pass


class StateInconsistencyError(PauseError):
    """状态不一致错误"""
    pass


class PauseState(Enum):
    """暂停状态枚举"""
    RUNNING = "running"  # 正常运行
    PAUSED = "paused"    # 已暂停
    RESUMING = "resuming"  # 恢复中


class PauseMode(Enum):
    """暂停模式枚举"""
    IMMEDIATE = "immediate"  # 立即暂停
    GRACEFUL = "graceful"    # 优雅暂停（完成当前任务）


@dataclass
class PauseMetrics:
    """暂停统计指标"""
    pause_count: int = 0
    total_pause_duration: float = 0.0
    last_pause_time: Optional[float] = None
    last_resume_time: Optional[float] = None


@dataclass
class BufferedAudioFrame:
    """
    缓冲的音频帧数据模型
    
    用于在暂停期间缓存音频帧，确保数据不丢失
    """
    frame_data: np.ndarray  # 音频帧数据
    timestamp: float  # 时间戳
    event_info: Dict[str, Any]  # 事件信息
    session_id: Optional[int] = None  # 会话ID
    sequence_number: int = 0  # 序列号，用于保证顺序
    
    def is_expired(self, max_age_seconds: float) -> bool:
        """
        检查数据是否过期
        
        Args:
            max_age_seconds: 最大年龄（秒）
            
        Returns:
            bool: True表示已过期
        """
        return (time.time() - self.timestamp) > max_age_seconds


@dataclass
class BufferedInferenceData:
    """
    缓冲的推理数据模型
    
    用于在暂停期间缓存推理相关数据
    """
    audio_features: np.ndarray  # 音频特征
    latent_data: Optional[Any] = None  # 潜在数据（可能是torch.Tensor等）
    timestamp: float = field(default_factory=time.time)  # 时间戳
    session_id: Optional[int] = None  # 会话ID
    batch_index: int = 0  # 批次索引
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    
    def is_expired(self, max_age_seconds: float) -> bool:
        """
        检查数据是否过期
        
        Args:
            max_age_seconds: 最大年龄（秒）
            
        Returns:
            bool: True表示已过期
        """
        return (time.time() - self.timestamp) > max_age_seconds


class ImprovedPauseController:
    """
    改进的暂停控制器
    
    提供线程安全的暂停/恢复功能，支持：
    - 立即暂停和优雅暂停模式
    - 状态查询和监控
    - 暂停统计指标
    - 回调通知机制
    """
    
    def __init__(self, mode: PauseMode = PauseMode.IMMEDIATE, 
                 buffer_size: int = 1000,
                 max_buffer_age: float = 30.0):
        """
        初始化暂停控制器
        
        Args:
            mode: 暂停模式（立即或优雅）
            buffer_size: 缓冲区最大大小
            max_buffer_age: 缓冲数据最大年龄（秒）
        """
        # 核心同步机制
        self._pause_event = threading.Event()
        self._pause_event.set()  # 初始状态为运行
        
        # 状态管理
        self._state = PauseState.RUNNING
        self._state_lock = threading.RLock()
        
        # 暂停模式
        self._mode = mode
        
        # 统计指标
        self._metrics = PauseMetrics()
        self._metrics_lock = threading.Lock()
        
        # 回调函数
        self._on_pause_callback: Optional[Callable] = None
        self._on_resume_callback: Optional[Callable] = None
        
        # 数据缓冲区
        self._audio_buffer: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._inference_buffer: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._buffer_lock = threading.Lock()
        self._max_buffer_age = max_buffer_age
        self._buffer_size = buffer_size
        
        # 缓冲区统计
        self._audio_frames_buffered = 0
        self._inference_data_buffered = 0
        self._buffer_overflow_count = 0
        self._sequence_counter = 0
        
        logger.info(f"ImprovedPauseController initialized with mode: {mode.value}, "
                   f"buffer_size: {buffer_size}, max_buffer_age: {max_buffer_age}s")
    
    def pause(self, mode: Optional[PauseMode] = None) -> bool:
        """
        暂停操作
        
        Args:
            mode: 可选的暂停模式，如果不指定则使用初始化时的模式
            
        Returns:
            bool: 是否成功暂停
        """
        with self._state_lock:
            if self._state == PauseState.PAUSED:
                logger.warning("Already paused, ignoring pause request")
                return False
            
            # 使用指定模式或默认模式
            effective_mode = mode if mode is not None else self._mode
            
            logger.info(f"Pausing with mode: {effective_mode.value}")
            
            # 清除事件，阻塞等待的线程
            self._pause_event.clear()
            
            # 更新状态
            self._state = PauseState.PAUSED
            
            # 更新统计指标
            with self._metrics_lock:
                self._metrics.pause_count += 1
                self._metrics.last_pause_time = time.time()
            
            # 触发回调
            if self._on_pause_callback:
                try:
                    self._on_pause_callback()
                except Exception as e:
                    logger.error(f"Error in pause callback: {e}")
            
            return True

    
    def resume(self, clear_buffers: bool = True) -> bool:
        """
        恢复操作
        
        Args:
            clear_buffers: 是否在恢复时清空缓冲区（默认True，避免音画不同步）
        
        Returns:
            bool: 是否成功恢复
        """
        with self._state_lock:
            if self._state == PauseState.RUNNING:
                logger.warning("Already running, ignoring resume request")
                return False
            
            logger.info("Resuming operation")
            
            # 更新状态为恢复中
            self._state = PauseState.RESUMING
            
            # 清空缓冲区以避免音画不同步
            if clear_buffers:
                logger.info("Clearing buffers to prevent audio-video desync")
                self.clear_buffers()
            
            # 设置事件，释放等待的线程
            self._pause_event.set()
            
            # 更新状态为运行
            self._state = PauseState.RUNNING
            
            # 更新统计指标
            with self._metrics_lock:
                current_time = time.time()
                self._metrics.last_resume_time = current_time
                
                # 计算本次暂停时长
                if self._metrics.last_pause_time:
                    pause_duration = current_time - self._metrics.last_pause_time
                    self._metrics.total_pause_duration += pause_duration
                    logger.info(f"Pause duration: {pause_duration:.2f}s")
            
            # 触发回调
            if self._on_resume_callback:
                try:
                    self._on_resume_callback()
                except Exception as e:
                    logger.error(f"Error in resume callback: {e}")
            
            return True
    
    def is_paused(self) -> bool:
        """
        检查是否处于暂停状态
        
        Returns:
            bool: True表示已暂停，False表示运行中
        """
        with self._state_lock:
            return self._state == PauseState.PAUSED
    
    def get_state(self) -> PauseState:
        """
        获取当前状态
        
        Returns:
            PauseState: 当前暂停状态
        """
        with self._state_lock:
            return self._state

    
    def wait_if_paused(self, timeout: Optional[float] = None) -> bool:
        """
        如果处于暂停状态，则等待直到恢复
        
        这是工作线程应该调用的主要方法
        
        Args:
            timeout: 可选的超时时间（秒），None表示无限等待
            
        Returns:
            bool: True表示已恢复或未暂停，False表示超时
        """
        # 如果未暂停，立即返回True
        if not self.is_paused():
            return True
        
        logger.debug("Thread waiting due to pause state")
        result = self._pause_event.wait(timeout)
        
        if result:
            logger.debug("Thread resumed after pause")
        else:
            logger.warning(f"Thread wait timed out after {timeout}s")
        
        return result
    
    def check_and_wait(self, timeout: Optional[float] = 0.5) -> bool:
        """
        检查暂停状态并等待（如果需要）
        
        这是一个便捷的检查点方法，用于在关键处理点检查暂停状态。
        如果处于暂停状态，会阻塞当前线程直到恢复。
        
        Args:
            timeout: 超时时间（秒），默认0.5秒。None表示无限等待
        
        Returns:
            bool: True表示可以继续执行，False表示超时
        """
        if self.is_paused():
            logger.debug("Checkpoint: paused, waiting for resume")
            result = self._pause_event.wait(timeout)
            if result:
                logger.debug("Checkpoint: resumed, continuing execution")
            else:
                # 使用debug级别而不是warning，避免日志过多
                logger.debug(f"Checkpoint: wait timeout after {timeout}s, will retry")
            return result
        
        return True
    
    def set_mode(self, mode: PauseMode):
        """
        设置暂停模式
        
        Args:
            mode: 新的暂停模式
        """
        with self._state_lock:
            self._mode = mode
            logger.info(f"Pause mode changed to: {mode.value}")
    
    def get_mode(self) -> PauseMode:
        """
        获取当前暂停模式
        
        Returns:
            PauseMode: 当前暂停模式
        """
        with self._state_lock:
            return self._mode
    
    def get_metrics(self) -> PauseMetrics:
        """
        获取暂停统计指标
        
        Returns:
            PauseMetrics: 暂停统计数据的副本
        """
        with self._metrics_lock:
            # 返回副本以避免外部修改
            return PauseMetrics(
                pause_count=self._metrics.pause_count,
                total_pause_duration=self._metrics.total_pause_duration,
                last_pause_time=self._metrics.last_pause_time,
                last_resume_time=self._metrics.last_resume_time
            )
    
    def reset_metrics(self):
        """重置统计指标"""
        with self._metrics_lock:
            self._metrics = PauseMetrics()
            logger.info("Pause metrics reset")

    
    def set_on_pause_callback(self, callback: Callable):
        """
        设置暂停时的回调函数
        
        Args:
            callback: 暂停时调用的函数
        """
        self._on_pause_callback = callback
        logger.info("Pause callback registered")
    
    def set_on_resume_callback(self, callback: Callable):
        """
        设置恢复时的回调函数
        
        Args:
            callback: 恢复时调用的函数
        """
        self._on_resume_callback = callback
        logger.info("Resume callback registered")
    
    def buffer_audio_frame(self, frame_data: np.ndarray, 
                          event_info: Dict[str, Any],
                          session_id: Optional[int] = None) -> bool:
        """
        缓存音频帧数据
        
        在暂停期间调用此方法来缓存音频帧，避免数据丢失
        
        Args:
            frame_data: 音频帧数据（numpy数组）
            event_info: 事件信息字典
            session_id: 可选的会话ID
            
        Returns:
            bool: True表示成功缓存，False表示缓冲区已满
        """
        try:
            with self._buffer_lock:
                self._sequence_counter += 1
                buffered_frame = BufferedAudioFrame(
                    frame_data=frame_data,
                    timestamp=time.time(),
                    event_info=event_info,
                    session_id=session_id,
                    sequence_number=self._sequence_counter
                )
                
                # 尝试放入缓冲区（非阻塞）
                self._audio_buffer.put_nowait(buffered_frame)
                self._audio_frames_buffered += 1
                
                logger.debug(f"Audio frame buffered (seq: {self._sequence_counter}, "
                           f"buffer size: {self._audio_buffer.qsize()})")
                return True
                
        except queue.Full:
            self._buffer_overflow_count += 1
            logger.warning(f"Audio buffer full (size: {self._buffer_size}), "
                         f"frame dropped. Overflow count: {self._buffer_overflow_count}")
            
            # 尝试清理过期数据后重试
            self._handle_buffer_overflow('audio')
            
            try:
                self._audio_buffer.put_nowait(buffered_frame)
                self._audio_frames_buffered += 1
                logger.info("Audio frame buffered after overflow cleanup")
                return True
            except queue.Full:
                logger.error("Audio buffer still full after cleanup, frame lost")
                return False
    
    def buffer_inference_data(self, audio_features: np.ndarray,
                             latent_data: Optional[Any] = None,
                             session_id: Optional[int] = None,
                             batch_index: int = 0,
                             metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        缓存推理数据
        
        在暂停期间调用此方法来缓存推理相关数据
        
        Args:
            audio_features: 音频特征数据
            latent_data: 可选的潜在数据
            session_id: 可选的会话ID
            batch_index: 批次索引
            metadata: 可选的元数据字典
            
        Returns:
            bool: True表示成功缓存，False表示缓冲区已满
        """
        try:
            with self._buffer_lock:
                buffered_data = BufferedInferenceData(
                    audio_features=audio_features,
                    latent_data=latent_data,
                    timestamp=time.time(),
                    session_id=session_id,
                    batch_index=batch_index,
                    metadata=metadata or {}
                )
                
                # 尝试放入缓冲区（非阻塞）
                self._inference_buffer.put_nowait(buffered_data)
                self._inference_data_buffered += 1
                
                logger.debug(f"Inference data buffered (batch: {batch_index}, "
                           f"buffer size: {self._inference_buffer.qsize()})")
                return True
                
        except queue.Full:
            self._buffer_overflow_count += 1
            logger.warning(f"Inference buffer full (size: {self._buffer_size}), "
                         f"data dropped. Overflow count: {self._buffer_overflow_count}")
            
            # 尝试清理过期数据后重试
            self._handle_buffer_overflow('inference')
            
            try:
                self._inference_buffer.put_nowait(buffered_data)
                self._inference_data_buffered += 1
                logger.info("Inference data buffered after overflow cleanup")
                return True
            except queue.Full:
                logger.error("Inference buffer still full after cleanup, data lost")
                return False
    
    def get_buffered_audio_frame(self, timeout: Optional[float] = None) -> Optional[BufferedAudioFrame]:
        """
        获取缓冲的音频帧
        
        Args:
            timeout: 可选的超时时间（秒）
            
        Returns:
            BufferedAudioFrame: 缓冲的音频帧，如果队列为空则返回None
        """
        try:
            frame = self._audio_buffer.get(block=True, timeout=timeout)
            logger.debug(f"Retrieved buffered audio frame (seq: {frame.sequence_number})")
            return frame
        except queue.Empty:
            return None
    
    def get_buffered_inference_data(self, timeout: Optional[float] = None) -> Optional[BufferedInferenceData]:
        """
        获取缓冲的推理数据
        
        Args:
            timeout: 可选的超时时间（秒）
            
        Returns:
            BufferedInferenceData: 缓冲的推理数据，如果队列为空则返回None
        """
        try:
            data = self._inference_buffer.get(block=True, timeout=timeout)
            logger.debug(f"Retrieved buffered inference data (batch: {data.batch_index})")
            return data
        except queue.Empty:
            return None
    
    def flush_expired_buffers(self, max_age_seconds: Optional[float] = None) -> Dict[str, int]:
        """
        清理过期的缓冲数据
        
        Args:
            max_age_seconds: 可选的最大年龄（秒），如果不指定则使用初始化时的值
            
        Returns:
            Dict[str, int]: 清理统计信息，包含清理的音频帧和推理数据数量
        """
        max_age = max_age_seconds if max_age_seconds is not None else self._max_buffer_age
        
        audio_cleaned = 0
        inference_cleaned = 0
        
        with self._buffer_lock:
            # 清理音频缓冲区
            audio_cleaned = self._flush_queue(self._audio_buffer, max_age)
            
            # 清理推理缓冲区
            inference_cleaned = self._flush_queue(self._inference_buffer, max_age)
        
        if audio_cleaned > 0 or inference_cleaned > 0:
            logger.info(f"Flushed expired buffers: {audio_cleaned} audio frames, "
                       f"{inference_cleaned} inference data items")
        
        return {
            'audio_frames_cleaned': audio_cleaned,
            'inference_data_cleaned': inference_cleaned
        }
    
    def _flush_queue(self, buffer_queue: queue.Queue, max_age: float) -> int:
        """
        清理队列中的过期数据
        
        Args:
            buffer_queue: 要清理的队列
            max_age: 最大年龄（秒）
            
        Returns:
            int: 清理的数据项数量
        """
        temp_buffer = []
        cleaned_count = 0
        current_time = time.time()
        
        # 取出所有数据
        while not buffer_queue.empty():
            try:
                item = buffer_queue.get_nowait()
                
                # 检查是否过期
                if hasattr(item, 'is_expired') and item.is_expired(max_age):
                    cleaned_count += 1
                    logger.debug(f"Expired buffer item removed (age: {current_time - item.timestamp:.2f}s)")
                else:
                    temp_buffer.append(item)
                    
            except queue.Empty:
                break
        
        # 将未过期的数据放回队列
        for item in temp_buffer:
            try:
                buffer_queue.put_nowait(item)
            except queue.Full:
                logger.warning("Queue full while restoring items after flush")
                break
        
        return cleaned_count
    
    def _handle_buffer_overflow(self, buffer_type: str):
        """
        处理缓冲区溢出
        
        采用LRU策略：清理最旧的数据
        
        Args:
            buffer_type: 缓冲区类型（'audio' 或 'inference'）
        """
        logger.warning(f"Handling {buffer_type} buffer overflow")
        
        # 首先尝试清理过期数据
        cleaned = self.flush_expired_buffers()
        
        if buffer_type == 'audio':
            if cleaned['audio_frames_cleaned'] == 0:
                # 如果没有过期数据，移除最旧的项
                try:
                    old_frame = self._audio_buffer.get_nowait()
                    logger.warning(f"Removed oldest audio frame (seq: {old_frame.sequence_number}) "
                                 f"due to buffer overflow")
                except queue.Empty:
                    pass
        else:
            if cleaned['inference_data_cleaned'] == 0:
                # 如果没有过期数据，移除最旧的项
                try:
                    old_data = self._inference_buffer.get_nowait()
                    logger.warning(f"Removed oldest inference data (batch: {old_data.batch_index}) "
                                 f"due to buffer overflow")
                except queue.Empty:
                    pass
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """
        获取缓冲区统计信息
        
        Returns:
            Dict[str, Any]: 缓冲区统计数据
        """
        with self._buffer_lock:
            return {
                'audio_buffer_size': self._audio_buffer.qsize(),
                'inference_buffer_size': self._inference_buffer.qsize(),
                'audio_frames_buffered': self._audio_frames_buffered,
                'inference_data_buffered': self._inference_data_buffered,
                'buffer_overflow_count': self._buffer_overflow_count,
                'max_buffer_size': self._buffer_size,
                'max_buffer_age': self._max_buffer_age
            }
    
    def clear_buffers(self):
        """清空所有缓冲区"""
        with self._buffer_lock:
            # 清空音频缓冲区
            audio_count = 0
            while not self._audio_buffer.empty():
                try:
                    self._audio_buffer.get_nowait()
                    audio_count += 1
                except queue.Empty:
                    break
            
            # 清空推理缓冲区
            inference_count = 0
            while not self._inference_buffer.empty():
                try:
                    self._inference_buffer.get_nowait()
                    inference_count += 1
                except queue.Empty:
                    break
            
            logger.info(f"Cleared buffers: {audio_count} audio frames, "
                       f"{inference_count} inference data items")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出，确保恢复运行状态"""
        if self.is_paused():
            self.resume()
        return False
    
    def __repr__(self) -> str:
        """字符串表示"""
        with self._state_lock:
            return (f"ImprovedPauseController(state={self._state.value}, "
                   f"mode={self._mode.value})")


class PauseControllerErrorHandler:
    """
    暂停控制器错误处理器
    
    提供死锁检测、缓冲区溢出处理、状态不一致修复和紧急重置功能
    """
    
    def __init__(self, controller: ImprovedPauseController, 
                 deadlock_timeout: float = 30.0,
                 max_recovery_attempts: int = 3):
        """
        初始化错误处理器
        
        Args:
            controller: 要管理的暂停控制器实例
            deadlock_timeout: 死锁检测超时时间（秒）
            max_recovery_attempts: 最大恢复尝试次数
        """
        self._controller = controller
        self._deadlock_timeout = deadlock_timeout
        self._max_recovery_attempts = max_recovery_attempts
        
        # 错误统计
        self._deadlock_count = 0
        self._buffer_overflow_handled = 0
        self._state_inconsistency_count = 0
        self._emergency_reset_count = 0
        self._recovery_success_count = 0
        self._recovery_failure_count = 0
        
        # 死锁检测
        self._last_state_change_time = time.time()
        self._deadlock_detection_lock = threading.Lock()
        
        logger.info(f"PauseControllerErrorHandler initialized with "
                   f"deadlock_timeout={deadlock_timeout}s, "
                   f"max_recovery_attempts={max_recovery_attempts}")

    
    def handle_deadlock(self, thread_info: Optional[Dict[str, Any]] = None) -> bool:
        """
        处理死锁情况
        
        检测并尝试解决暂停控制器中的死锁问题。死锁可能发生在：
        - 多个线程同时等待暂停事件
        - 状态锁被长时间持有
        - 缓冲区操作导致的循环等待
        
        Args:
            thread_info: 可选的线程信息字典，用于诊断
            
        Returns:
            bool: True表示成功处理死锁，False表示处理失败
        """
        with self._deadlock_detection_lock:
            self._deadlock_count += 1
            
            logger.error(f"Deadlock detected (count: {self._deadlock_count})")
            
            if thread_info:
                logger.error(f"Thread info: {thread_info}")
            
            # 记录当前状态
            current_state = self._controller.get_state()
            buffer_stats = self._controller.get_buffer_stats()
            
            logger.error(f"Current state: {current_state.value}")
            logger.error(f"Buffer stats: {buffer_stats}")
            
            # 尝试恢复策略
            for attempt in range(1, self._max_recovery_attempts + 1):
                logger.warning(f"Deadlock recovery attempt {attempt}/{self._max_recovery_attempts}")
                
                try:
                    # 策略1: 强制恢复运行状态
                    if current_state == PauseState.PAUSED:
                        logger.warning("Attempting to force resume from paused state")
                        success = self._controller.resume()
                        if success:
                            logger.info("Successfully resumed from deadlock")
                            self._recovery_success_count += 1
                            self._last_state_change_time = time.time()
                            return True
                    
                    # 策略2: 清空缓冲区以释放可能的阻塞
                    logger.warning("Clearing buffers to release potential blocks")
                    self._controller.clear_buffers()
                    
                    # 策略3: 重置暂停事件
                    logger.warning("Resetting pause event")
                    self._controller._pause_event.set()
                    
                    # 短暂等待，观察是否恢复
                    time.sleep(0.5)
                    
                    # 检查状态是否恢复
                    new_state = self._controller.get_state()
                    if new_state == PauseState.RUNNING:
                        logger.info(f"Deadlock resolved after attempt {attempt}")
                        self._recovery_success_count += 1
                        self._last_state_change_time = time.time()
                        return True
                    
                except Exception as e:
                    logger.error(f"Error during deadlock recovery attempt {attempt}: {e}")
            
            # 所有恢复尝试失败
            logger.error(f"Failed to resolve deadlock after {self._max_recovery_attempts} attempts")
            self._recovery_failure_count += 1
            
            # 建议紧急重置
            logger.critical("Deadlock recovery failed. Consider calling emergency_reset()")
            
            return False

    
    def handle_buffer_overflow(self, buffer_type: str = 'all', 
                              aggressive: bool = False) -> Dict[str, int]:
        """
        处理缓冲区溢出
        
        当缓冲区达到容量限制时，采取清理策略以恢复正常运行
        
        Args:
            buffer_type: 要处理的缓冲区类型 ('audio', 'inference', 'all')
            aggressive: 是否采用激进清理策略（清理更多数据）
            
        Returns:
            Dict[str, int]: 清理统计信息
        """
        self._buffer_overflow_handled += 1
        
        logger.warning(f"Handling buffer overflow (type: {buffer_type}, "
                      f"aggressive: {aggressive}, count: {self._buffer_overflow_handled})")
        
        stats = {
            'audio_frames_removed': 0,
            'inference_data_removed': 0,
            'expired_items_removed': 0
        }
        
        try:
            # 首先清理过期数据
            logger.info("Flushing expired buffers")
            expired_stats = self._controller.flush_expired_buffers()
            stats['expired_items_removed'] = (
                expired_stats['audio_frames_cleaned'] + 
                expired_stats['inference_data_cleaned']
            )
            
            # 如果采用激进策略，清理更多数据
            if aggressive:
                logger.warning("Using aggressive buffer cleanup strategy")
                
                if buffer_type in ['audio', 'all']:
                    # 清理一半的音频缓冲区
                    audio_size = self._controller._audio_buffer.qsize()
                    target_remove = audio_size // 2
                    
                    for _ in range(target_remove):
                        try:
                            self._controller._audio_buffer.get_nowait()
                            stats['audio_frames_removed'] += 1
                        except queue.Empty:
                            break
                    
                    logger.warning(f"Aggressively removed {stats['audio_frames_removed']} "
                                 f"audio frames")
                
                if buffer_type in ['inference', 'all']:
                    # 清理一半的推理缓冲区
                    inference_size = self._controller._inference_buffer.qsize()
                    target_remove = inference_size // 2
                    
                    for _ in range(target_remove):
                        try:
                            self._controller._inference_buffer.get_nowait()
                            stats['inference_data_removed'] += 1
                        except queue.Empty:
                            break
                    
                    logger.warning(f"Aggressively removed {stats['inference_data_removed']} "
                                 f"inference data items")
            
            # 记录最终缓冲区状态
            final_stats = self._controller.get_buffer_stats()
            logger.info(f"Buffer overflow handled. Final buffer sizes - "
                       f"audio: {final_stats['audio_buffer_size']}, "
                       f"inference: {final_stats['inference_buffer_size']}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error handling buffer overflow: {e}")
            raise BufferOverflowError(f"Failed to handle buffer overflow: {e}")

    
    def handle_state_inconsistency(self, expected_state: Optional[PauseState] = None,
                                   force_state: Optional[PauseState] = None) -> bool:
        """
        处理状态不一致
        
        检测并修复暂停控制器的状态不一致问题，例如：
        - 状态标志与实际行为不匹配
        - 事件状态与状态枚举不一致
        - 缓冲区状态与暂停状态不匹配
        
        Args:
            expected_state: 期望的状态，用于验证
            force_state: 强制设置的状态（谨慎使用）
            
        Returns:
            bool: True表示成功修复，False表示修复失败
        """
        self._state_inconsistency_count += 1
        
        logger.error(f"State inconsistency detected (count: {self._state_inconsistency_count})")
        
        try:
            # 获取当前状态信息
            current_state = self._controller.get_state()
            event_is_set = self._controller._pause_event.is_set()
            buffer_stats = self._controller.get_buffer_stats()
            
            logger.error(f"Current state: {current_state.value}")
            logger.error(f"Pause event is set: {event_is_set}")
            logger.error(f"Buffer stats: {buffer_stats}")
            
            # 检测不一致情况
            inconsistencies = []
            
            # 不一致1: 状态为RUNNING但事件未设置
            if current_state == PauseState.RUNNING and not event_is_set:
                inconsistencies.append("State is RUNNING but pause event is not set")
                logger.error("Inconsistency: State is RUNNING but pause event is not set")
            
            # 不一致2: 状态为PAUSED但事件已设置
            if current_state == PauseState.PAUSED and event_is_set:
                inconsistencies.append("State is PAUSED but pause event is set")
                logger.error("Inconsistency: State is PAUSED but pause event is set")
            
            # 不一致3: 期望状态不匹配
            if expected_state and current_state != expected_state:
                inconsistencies.append(f"Expected {expected_state.value} but got {current_state.value}")
                logger.error(f"Inconsistency: Expected {expected_state.value} but got {current_state.value}")
            
            if not inconsistencies and not force_state:
                logger.info("No state inconsistencies detected")
                return True
            
            # 修复策略
            logger.warning("Attempting to fix state inconsistencies")
            
            if force_state:
                # 强制设置状态
                logger.warning(f"Forcing state to {force_state.value}")
                
                with self._controller._state_lock:
                    self._controller._state = force_state
                    
                    if force_state == PauseState.RUNNING:
                        self._controller._pause_event.set()
                    elif force_state == PauseState.PAUSED:
                        self._controller._pause_event.clear()
                
                logger.info(f"State forced to {force_state.value}")
                self._recovery_success_count += 1
                with self._deadlock_detection_lock:
                    self._last_state_change_time = time.time()
                return True
            
            else:
                # 自动修复：同步状态和事件
                with self._controller._state_lock:
                    if current_state == PauseState.RUNNING and not event_is_set:
                        logger.warning("Fixing: Setting pause event to match RUNNING state")
                        self._controller._pause_event.set()
                    
                    elif current_state == PauseState.PAUSED and event_is_set:
                        logger.warning("Fixing: Clearing pause event to match PAUSED state")
                        self._controller._pause_event.clear()
                    
                    elif expected_state:
                        logger.warning(f"Fixing: Aligning state to expected {expected_state.value}")
                        self._controller._state = expected_state
                        
                        if expected_state == PauseState.RUNNING:
                            self._controller._pause_event.set()
                        elif expected_state == PauseState.PAUSED:
                            self._controller._pause_event.clear()
                
                # 验证修复
                time.sleep(0.1)
                new_state = self._controller.get_state()
                new_event_is_set = self._controller._pause_event.is_set()
                
                is_consistent = (
                    (new_state == PauseState.RUNNING and new_event_is_set) or
                    (new_state == PauseState.PAUSED and not new_event_is_set) or
                    (new_state == PauseState.RESUMING)
                )
                
                if is_consistent:
                    logger.info("State inconsistency successfully fixed")
                    self._recovery_success_count += 1
                    with self._deadlock_detection_lock:
                        self._last_state_change_time = time.time()
                    return True
                else:
                    logger.error("Failed to fix state inconsistency")
                    self._recovery_failure_count += 1
                    return False
        
        except Exception as e:
            logger.error(f"Error handling state inconsistency: {e}")
            self._recovery_failure_count += 1
            raise StateInconsistencyError(f"Failed to handle state inconsistency: {e}")

    
    def emergency_reset(self, clear_metrics: bool = False) -> bool:
        """
        紧急重置
        
        当所有其他恢复策略失败时，执行完全重置。这将：
        - 强制设置为RUNNING状态
        - 清空所有缓冲区
        - 重置所有事件和锁
        - 可选地清除统计指标
        
        警告：这是一个破坏性操作，应该只在紧急情况下使用
        
        Args:
            clear_metrics: 是否清除统计指标
            
        Returns:
            bool: True表示重置成功
        """
        self._emergency_reset_count += 1
        
        logger.critical(f"EMERGENCY RESET initiated (count: {self._emergency_reset_count})")
        logger.critical("This is a destructive operation that will reset all state")
        
        try:
            # 1. 强制释放所有锁（通过重新创建）
            logger.warning("Resetting locks")
            self._controller._state_lock = threading.RLock()
            self._controller._metrics_lock = threading.Lock()
            self._controller._buffer_lock = threading.Lock()
            
            # 2. 重置暂停事件
            logger.warning("Resetting pause event")
            self._controller._pause_event = threading.Event()
            self._controller._pause_event.set()  # 设置为运行状态
            
            # 3. 强制设置状态为RUNNING
            logger.warning("Forcing state to RUNNING")
            self._controller._state = PauseState.RUNNING
            
            # 4. 清空所有缓冲区
            logger.warning("Clearing all buffers")
            self._controller.clear_buffers()
            
            # 5. 重置缓冲区统计
            logger.warning("Resetting buffer statistics")
            self._controller._audio_frames_buffered = 0
            self._controller._inference_data_buffered = 0
            self._controller._buffer_overflow_count = 0
            self._controller._sequence_counter = 0
            
            # 6. 可选：清除暂停统计指标
            if clear_metrics:
                logger.warning("Clearing pause metrics")
                self._controller.reset_metrics()
            
            # 7. 重置错误处理器的状态
            logger.warning("Resetting error handler state")
            with self._deadlock_detection_lock:
                self._last_state_change_time = time.time()
            
            # 8. 验证重置结果
            final_state = self._controller.get_state()
            event_is_set = self._controller._pause_event.is_set()
            buffer_stats = self._controller.get_buffer_stats()
            
            logger.info(f"Emergency reset completed. Final state: {final_state.value}")
            logger.info(f"Pause event is set: {event_is_set}")
            logger.info(f"Buffer stats: {buffer_stats}")
            
            if final_state == PauseState.RUNNING and event_is_set:
                logger.info("Emergency reset successful - controller is in clean RUNNING state")
                self._recovery_success_count += 1
                return True
            else:
                logger.error("Emergency reset completed but state may still be inconsistent")
                self._recovery_failure_count += 1
                return False
        
        except Exception as e:
            logger.critical(f"CRITICAL: Emergency reset failed with error: {e}")
            self._recovery_failure_count += 1
            raise PauseError(f"Emergency reset failed: {e}")

    
    def detect_deadlock(self, timeout: Optional[float] = None) -> bool:
        """
        检测是否存在死锁
        
        通过监控状态变化时间来检测潜在的死锁情况
        
        Args:
            timeout: 可选的超时时间（秒），如果不指定则使用初始化时的值
            
        Returns:
            bool: True表示检测到死锁，False表示正常
        """
        check_timeout = timeout if timeout is not None else self._deadlock_timeout
        
        with self._deadlock_detection_lock:
            time_since_last_change = time.time() - self._last_state_change_time
            
            # 如果控制器处于暂停状态且时间过长，可能是死锁
            if self._controller.is_paused() and time_since_last_change > check_timeout:
                logger.warning(f"Potential deadlock detected: paused for {time_since_last_change:.2f}s "
                             f"(threshold: {check_timeout}s)")
                return True
            
            return False
    
    def _update_state_change_time(self):
        """更新最后状态变化时间"""
        with self._deadlock_detection_lock:
            self._last_state_change_time = time.time()
    
    def get_error_stats(self) -> Dict[str, int]:
        """
        获取错误处理统计信息
        
        Returns:
            Dict[str, int]: 错误统计数据
        """
        return {
            'deadlock_count': self._deadlock_count,
            'buffer_overflow_handled': self._buffer_overflow_handled,
            'state_inconsistency_count': self._state_inconsistency_count,
            'emergency_reset_count': self._emergency_reset_count,
            'recovery_success_count': self._recovery_success_count,
            'recovery_failure_count': self._recovery_failure_count,
            'recovery_success_rate': (
                self._recovery_success_count / 
                max(1, self._recovery_success_count + self._recovery_failure_count)
            )
        }
    
    def reset_error_stats(self):
        """重置错误统计信息"""
        self._deadlock_count = 0
        self._buffer_overflow_handled = 0
        self._state_inconsistency_count = 0
        self._emergency_reset_count = 0
        self._recovery_success_count = 0
        self._recovery_failure_count = 0
        logger.info("Error handler statistics reset")
    
    def __repr__(self) -> str:
        """字符串表示"""
        stats = self.get_error_stats()
        return (f"PauseControllerErrorHandler("
               f"deadlocks={stats['deadlock_count']}, "
               f"overflows={stats['buffer_overflow_handled']}, "
               f"inconsistencies={stats['state_inconsistency_count']}, "
               f"emergency_resets={stats['emergency_reset_count']}, "
               f"success_rate={stats['recovery_success_rate']:.2%})")
