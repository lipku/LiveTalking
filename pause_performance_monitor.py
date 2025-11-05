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
暂停控制性能监控模块
提供暂停响应时间、缓冲区使用率、CPU和内存监控功能
"""

import time
import threading
import psutil
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import deque
from logger import logger


@dataclass
class PerformanceSnapshot:
    """性能快照数据模型"""
    timestamp: float
    pause_response_time_ms: Optional[float] = None
    resume_response_time_ms: Optional[float] = None
    audio_buffer_usage_percent: float = 0.0
    inference_buffer_usage_percent: float = 0.0
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    memory_percent: float = 0.0
    thread_count: int = 0
    lock_contention_count: int = 0


@dataclass
class PerformanceMetrics:
    """性能指标统计"""
    avg_pause_response_ms: float = 0.0
    max_pause_response_ms: float = 0.0
    min_pause_response_ms: float = float('inf')
    avg_resume_response_ms: float = 0.0
    max_resume_response_ms: float = 0.0
    min_resume_response_ms: float = float('inf')
    avg_buffer_usage_percent: float = 0.0
    max_buffer_usage_percent: float = 0.0
    avg_cpu_percent: float = 0.0
    max_cpu_percent: float = 0.0
    avg_memory_mb: float = 0.0
    max_memory_mb: float = 0.0
    total_samples: int = 0
    performance_impact_percent: float = 0.0


class PausePerformanceMonitor:
    """
    暂停控制性能监控器
    
    监控暂停控制系统的性能指标，包括：
    - 暂停/恢复响应时间
    - 缓冲区使用率
    - CPU和内存使用
    - 线程和锁竞争情况
    """
    
    def __init__(self, 
                 history_size: int = 1000,
                 sampling_interval: float = 1.0,
                 enable_continuous_monitoring: bool = False):
        """
        初始化性能监控器
        
        Args:
            history_size: 保留的历史快照数量
            sampling_interval: 采样间隔（秒）
            enable_continuous_monitoring: 是否启用持续监控
        """
        self._history_size = history_size
        self._sampling_interval = sampling_interval
        self._enable_continuous_monitoring = enable_continuous_monitoring
        
        # 性能快照历史
        self._snapshots: deque = deque(maxlen=history_size)
        self._snapshots_lock = threading.Lock()
        
        # 响应时间追踪
        self._pause_start_times: Dict[int, float] = {}
        self._resume_start_times: Dict[int, float] = {}
        self._response_times_lock = threading.Lock()
        
        # 基线性能（用于计算性能影响）
        self._baseline_cpu: Optional[float] = None
        self._baseline_memory: Optional[float] = None
        self._baseline_established = False
        
        # 锁竞争计数
        self._lock_contention_count = 0
        self._lock_contention_lock = threading.Lock()
        
        # 持续监控线程
        self._monitoring_thread: Optional[threading.Thread] = None
        self._monitoring_active = False
        self._monitoring_stop_event = threading.Event()
        
        # 进程信息
        self._process = psutil.Process(os.getpid())
        
        # 性能警告阈值
        self._warning_thresholds = {
            'pause_response_ms': 100.0,
            'resume_response_ms': 100.0,
            'buffer_usage_percent': 80.0,
            'cpu_percent': 50.0,
            'memory_mb': 1000.0,
            'performance_impact_percent': 5.0
        }
        
        logger.info(f"PausePerformanceMonitor initialized with history_size={history_size}, "
                   f"sampling_interval={sampling_interval}s")
        
        if enable_continuous_monitoring:
            self.start_continuous_monitoring()
