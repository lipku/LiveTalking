import os
import base64
import time
import threading
import numpy as np
import resampy

from utils.logger import logger
from .base_tts import BaseTTS, State
from registry import register

try:
    import dashscope
    from dashscope.audio.qwen_tts_realtime import (
        QwenTtsRealtime,
        QwenTtsRealtimeCallback,
        AudioFormat,
    )
except ImportError:
    logger.error("QwenTTS 需要安装 dashscope SDK: pip install dashscope>=1.25.11")
    raise


SRC_SR = 24000   # Qwen TTS 只支持 24kHz 输出
DST_SR = 16000   # 项目标准采样率


@register("tts", "qwentts")
class QwenTTS(BaseTTS):
    """
    阿里云通义千问实时语音合成 (Qwen TTS Realtime)
    基于 DashScope Python SDK (dashscope >= 1.25.11)
    使用 commit 模式：只建立一次 WebSocket 连接，每次合成通过 append_text + commit 触发。

    需要设置环境变量 DASHSCOPE_API_KEY。
    用法:
        python app.py --tts qwentts --REF_FILE Cherry
    其中 REF_FILE 用作音色名称 (voice)，如 Cherry / Ethan 等系统音色。
    """

    def __init__(self, opt, parent):
        super().__init__(opt, parent)

        # 音色名, 复用 REF_FILE 参数
        self.voice = opt.REF_FILE if opt.REF_FILE else 'Cherry'
        # 模型名
        self.model = getattr(opt, 'qwen_tts_model', 'qwen3-tts-flash-realtime')

        # ---------- 内部状态 ----------
        self._remainder = np.array([], dtype=np.float32)  # 上次重采样后不足一 chunk 的 16kHz 样本
        self._response_event = threading.Event()
        self._first_chunk = True          # 当前合成的一句话里的第一个音频包
        self._current_text = ''
        self._current_textevent = {}

        logger.info("Mock QwenTTS initialized (no remote API connection established)")

    # ========================== 核心方法 ==========================

    def txt_to_audio(self, msg: tuple[str, dict]):
        try:
            text, textevent = msg
            t_start = time.perf_counter()

            logger.info(f"Mock QwenTTS synthesis for text: {text}")
            
            # Output start frame
            eventpoint_start = {'status': 'start', 'text': text}
            eventpoint_start.update(**textevent)
            self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint_start)

            # Output mock silence
            for _ in range(10):
                if self.state != State.RUNNING:
                    break
                self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), textevent)

            # Output end frame
            eventpoint_end = {'status': 'end', 'text': text}
            eventpoint_end.update(**textevent)
            self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint_end)

            t_end = time.perf_counter()
            logger.info(f"Mock QwenTTS synthesis completed, time: {t_end - t_start:.2f}s")

        except Exception as e:
            logger.exception(f"QwenTTS txt_to_audio 异常: {e}")

    # ========================== 流式音频处理（回调中调用）==========================

    def _on_audio_data(self, pcm_data: bytes):
        """收到 PCM 24kHz 16bit mono 音频，一次性 resample 到 16kHz 后分块推送"""
        if self.state != State.RUNNING:
            self._remainder = np.array([], dtype=np.float32)
            return

        # 整段 24kHz PCM -> float32 -> 一次性 resample 到 16kHz
        samples_16k = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
        #samples_16k = resampy.resample(x=samples_24k, sr_orig=SRC_SR, sr_new=DST_SR)

        # 拼接上次剩余
        if self._remainder.shape[0] > 0:
            samples_16k = np.concatenate([self._remainder, samples_16k])

        # 按 self.chunk (320 samples = 20ms @16kHz) 分块推送
        idx = 0
        total = samples_16k.shape[0]
        while total - idx >= self.chunk and self.state == State.RUNNING:
            frame = samples_16k[idx:idx + self.chunk]
            eventpoint = {}
            if self._first_chunk:
                eventpoint = {'status': 'start', 'text': self._current_text}
                self._first_chunk = False
            eventpoint.update(**self._current_textevent)
            self.parent.put_audio_frame(frame, eventpoint)
            idx += self.chunk

        # 不足一 chunk 的留到下次
        self._remainder = samples_16k[idx:] if idx < total else np.array([], dtype=np.float32)

    def _flush_remainder(self):
        """合成完毕，推送剩余样本并发送 end 事件"""
        if self.state != State.RUNNING:
            self._remainder = np.array([], dtype=np.float32)
            return

        # 推送剩余完整 chunk
        if self._remainder.shape[0] >= self.chunk:
            idx = 0
            total = self._remainder.shape[0]
            while total - idx >= self.chunk and self.state == State.RUNNING:
                frame = self._remainder[idx:idx + self.chunk]
                eventpoint = {}
                if self._first_chunk:
                    eventpoint = {'status': 'start', 'text': self._current_text}
                    self._first_chunk = False
                eventpoint.update(**self._current_textevent)
                self.parent.put_audio_frame(frame, eventpoint)
                idx += self.chunk

        self._remainder = np.array([], dtype=np.float32)

        # 发送 end 事件
        eventpoint = {'status': 'end', 'text': self._current_text}
        eventpoint.update(**self._current_textevent)
        self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)

    def stop_tts(self):
        self._tts_client.close()
        logger.info("QwenTTS 已关闭")