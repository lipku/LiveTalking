import time
import numpy as np
import resampy
import requests
from typing import Iterator

from utils.logger import logger
from .base_tts import BaseTTS, State
from registry import register

@register("tts", "cosyvoice")
class CosyVoiceTTS(BaseTTS):
    def __init__(self, opt, parent=None):
        super().__init__(opt, parent)
        # 创建连接池，保持长连接
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=3
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        # 流式开关：默认关闭，仅当 COSYVOICE_STREAMING=True 时开启
        self.use_streaming = getattr(opt, 'COSYVOICE_STREAMING', False)
        self.stream_chunk_size = getattr(opt, 'COSYVOICE_STREAM_CHUNK', 9600)
        
        logger.info(f"CosyVoiceTTS initialized: streaming={self.use_streaming}")

    def txt_to_audio(self, msg: tuple[str, dict]):
        """保持接口不变，内部自动选择流式或非流式"""
        text, textevent = msg
        ref_file = textevent.get('tts', {}).get('ref_file', self.opt.REF_FILE)
        ref_text = textevent.get('tts', {}).get('ref_text', self.opt.REF_TEXT)
        
        if self.use_streaming:
            # 流式模式：边收边播
            self.stream_tts_realtime(
                self.cosy_voice_stream(text, ref_file, ref_text, "zh", self.opt.TTS_SERVER),
                msg
            )
        else:
            # 非流式模式：保持原有行为
            self.stream_tts(
                self.cosy_voice(text, ref_file, ref_text, "zh", self.opt.TTS_SERVER),
                msg
            )

    def cosy_voice_stream(self, text, reffile, reftext, language, server_url) -> Iterator[bytes]:
        """
        流式 TTS 请求，边接收边播放
        
        流程：
        1. 发送 HTTP POST 到 /inference_zero_shot_stream
        2. 使用 response.iter_content 流式读取
        3. 每收到一块音频立即 yield
        """
        import datetime
        start = time.perf_counter()
        start_time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        logger.info(f"[COSY_STREAM_START] text='{text[:30]}...' at {start_time_str}")
        
        payload = {
            'tts_text': text,
            'prompt_text': reftext,
            'chunk_size': self.stream_chunk_size
        }
        
        try:
            res = self.session.post(
                f"{server_url}/inference_zero_shot_stream", 
                data=payload, 
                stream=True,
                headers={'Accept': 'audio/mpeg'}
            )
            
            if res.status_code != 200:
                logger.error(f"[COSY_STREAM_ERROR] {res.text}")
                return
            
            first_chunk = True
            chunk_count = 0
            
            for chunk in res.iter_content(chunk_size=self.stream_chunk_size):
                if chunk and self.state == State.RUNNING:
                    chunk_count += 1
                    if first_chunk:
                        ttfb = time.perf_counter() - start
                        logger.info(f"[COSY_STREAM_FIRST] TTFB: {ttfb:.3f}s")
                        first_chunk = False
                    yield chunk
            
            total_time = time.perf_counter() - start
            logger.info(f"[COSY_STREAM_END] chunks={chunk_count}, time={total_time:.3f}s")
            
        except Exception as e:
            logger.exception(f"[COSY_STREAM_EX] {e}")

    def stream_tts_realtime(self, audio_stream, msg: tuple[str, dict]):
        """
        流式音频处理：带 jitter buffer 的边接收边重采样边播放
        
        解决流式推理画面不连贯问题：
        1. 预缓冲阶段：积累至少 PRE_BUFFER_MS 音频后才开始播放
        2. Jitter Buffer：播放期间维持最小缓冲，吸收网络抖动
        3. 缓冲欠载保护：缓冲不足时重复最后一帧，避免切换到静音
        """
        PRE_BUFFER_MS = 800   # 预缓冲 800ms，给 ASR 足够音频上下文
        JITTER_THRESHOLD_MS = 400  # 缓冲低于此值时等待补充
        
        text, textevent = msg
        first = True
        buffer = np.array([], dtype=np.float32)
        pre_buffer_samples = int(24000 * PRE_BUFFER_MS / 1000)  # 24kHz 采样率下的样本数
        jitter_threshold_samples = int(24000 * JITTER_THRESHOLD_MS / 1000)
        request_start_time = time.perf_counter()
        pre_buffering = True  # 是否处于预缓冲阶段
        
        for chunk in audio_stream:
            if chunk and len(chunk) > 0 and self.state == State.RUNNING:
                # 转换为 float32
                audio_data = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                buffer = np.concatenate([buffer, audio_data])
                
                # 预缓冲阶段：积累足够音频后再开始播放
                if pre_buffering:
                    if len(buffer) >= pre_buffer_samples:
                        pre_buffering = False
                        pre_buffer_time = time.perf_counter() - request_start_time
                        logger.info(f"[COSY_STREAM_BUFFER] Pre-buffer done: {len(buffer)/24000*1000:.0f}ms in {pre_buffer_time:.3f}s")
                    else:
                        continue  # 继续累积
                
                # 播放阶段：边缓冲边播放
                while True:
                    # 计算当前可处理的样本数
                    process_len = min(len(buffer), 2400)  # 每次最多处理 100ms
                    if process_len < 480:  # 至少 20ms
                        break
                        
                    to_resample = buffer[:process_len]
                    resampled = resampy.resample(x=to_resample, sr_orig=24000, sr_new=self.sample_rate)
                    
                    # 发送给播放器
                    streamlen = resampled.shape[0]
                    idx = 0
                    while streamlen >= self.chunk:
                        eventpoint = {}
                        if first:
                            first_frame_elapsed = time.perf_counter() - request_start_time
                            logger.info(f"[COSY_STREAM_PLAY] First frame to player, elapsed: {first_frame_elapsed:.3f}s")
                            eventpoint = {'status': 'start', 'text': text}
                            first = False
                        eventpoint.update(**textevent)
                        self.parent.put_audio_frame(resampled[idx:idx+self.chunk], eventpoint)
                        streamlen -= self.chunk
                        idx += self.chunk
                    
                    # 保留未处理数据
                    buffer = buffer[process_len:]
                    
                    # 缓冲不足时暂停播放，等待补充
                    if len(buffer) < jitter_threshold_samples:
                        if len(buffer) > 0:
                            logger.debug(f"[COSY_STREAM_JITTER] Buffer low: {len(buffer)/24000*1000:.0f}ms, waiting for refill")
                        break
        
        # 播放结束：将剩余缓冲全部发送
        if not first and len(buffer) > 0:
            resampled = resampy.resample(x=buffer, sr_orig=24000, sr_new=self.sample_rate)
            streamlen = resampled.shape[0]
            idx = 0
            while streamlen >= self.chunk:
                eventpoint = {}
                eventpoint.update(**textevent)
                self.parent.put_audio_frame(resampled[idx:idx+self.chunk], eventpoint)
                streamlen -= self.chunk
                idx += self.chunk
        
        # 发送结束标记
        if not first:  # 只有实际播放过才发送结束
            eventpoint = {'status': 'end', 'text': text}
            eventpoint.update(**textevent)
            self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)
            logger.info(f"[COSY_STREAM_DONE] Playback complete")

    def cosy_voice(self, text, reffile, reftext,language, server_url) -> Iterator[bytes]:
        """原有的非流式方法，保持不变"""
        import datetime
        start = time.perf_counter()
        start_time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        logger.info(f"[TTS_REQUEST_START] text='{text[:30]}...' at {start_time_str}")
        
        payload = {
            'tts_text': text,
            'prompt_text': reftext
        }
        try:
            res = self.session.request("POST", f"{server_url}/inference_zero_shot", data=payload, stream=True)
            
            end = time.perf_counter()
            post_time = end - start
            logger.info(f"[TTS_POST_SENT] Time to make POST: {post_time:.3f}s")

            if res.status_code != 200:
                logger.error("[TTS_ERROR] Error:%s", res.text)
                return
                
            first = True
            first_chunk_time = None
        
            for chunk in res.iter_content(chunk_size=9600): # 960 24K*20ms*2
                if first:
                    first_chunk_time = time.perf_counter()
                    ttfb = first_chunk_time - start
                    logger.info(f"[TTS_FIRST_CHUNK] Time to first chunk: {ttfb:.3f}s")
                    first = False
                if chunk and self.state==State.RUNNING:
                    yield chunk
            
            end_time = time.perf_counter()
            total_time = end_time - start
            end_time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            logger.info(f"[TTS_REQUEST_END] Total time: {total_time:.3f}s at {end_time_str}")
        except Exception as e:
            logger.exception('[TTS_EXCEPTION] cosyvoice error')

    def stream_tts(self,audio_stream,msg:tuple[str, dict]):
        text,textevent = msg
        first = True
        chunk_count = 0
        request_start_time = time.perf_counter()
        
        for chunk in audio_stream:
            if chunk is not None and len(chunk)>0:
                chunk_count += 1
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = resampy.resample(x=stream, sr_orig=24000, sr_new=self.sample_rate)
                streamlen = stream.shape[0]
                idx=0
                while streamlen >= self.chunk:
                    eventpoint={}
                    if first:
                        first_frame_elapsed = time.perf_counter() - request_start_time
                        logger.info(f"[TTS_FIRST_FRAME] First audio frame put to player, elapsed: {first_frame_elapsed:.3f}s")
                        eventpoint={'status':'start','text':text}
                        first = False
                    eventpoint.update(**textevent) 
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk],eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
        
        total_elapsed = time.perf_counter() - request_start_time
        logger.info(f"[TTS_STREAM_END] Total chunks: {chunk_count}, total elapsed: {total_elapsed:.3f}s")
        eventpoint={'status':'end','text':text}
        eventpoint.update(**textevent) 
        self.parent.put_audio_frame(np.zeros(self.chunk,np.float32),eventpoint) 
