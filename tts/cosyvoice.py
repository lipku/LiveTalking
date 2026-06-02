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
        logger.info("CosyVoiceTTS initialized with connection pool")

    def txt_to_audio(self,msg:tuple[str, dict]):
        text,textevent = msg
        ref_file = textevent.get('tts', {}).get('ref_file',self.opt.REF_FILE)
        ref_text = textevent.get('tts', {}).get('ref_text',self.opt.REF_TEXT) 
        self.stream_tts(
            self.cosy_voice(
                text,
                ref_file,  
                ref_text,
                "zh", #en args.language,
                self.opt.TTS_SERVER, #"http://127.0.0.1:5000", #args.server_url,
            ),
            msg
        )

    def cosy_voice(self, text, reffile, reftext,language, server_url) -> Iterator[bytes]:
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
