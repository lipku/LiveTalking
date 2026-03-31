import time
import numpy as np
import resampy
import soundfile as sf
import requests
from io import BytesIO
from typing import Iterator

from utils.logger import logger
from .base_tts import BaseTTS, State
from registry import register

@register("tts", "gpt-sovits")
class SovitsTTS(BaseTTS):
    def txt_to_audio(self,msg:tuple[str, dict]): 
        text,textevent = msg
        ref_file = textevent.get('tts', {}).get('ref_file',self.opt.REF_FILE)
        ref_text = textevent.get('tts', {}).get('ref_text',self.opt.REF_TEXT)
        self.stream_tts(
            self.gpt_sovits(
                text=text,
                reffile=ref_file,
                reftext=ref_text,
                language="zh", #en args.language,
                server_url=self.opt.TTS_SERVER, #"http://127.0.0.1:5000", #args.server_url,
            ),
            msg
        )

    def gpt_sovits(self, text, reffile, reftext,language, server_url) -> Iterator[bytes]:
        start = time.perf_counter()
        req={
            'text':text,
            'text_lang':language,
            'ref_audio_path':reffile,
            'prompt_text':reftext,
            'prompt_lang':language,
            'media_type':'ogg',
            'streaming_mode':True
        }
        # req["text"] = text
        # req["text_language"] = language
        # req["character"] = character
        # req["emotion"] = emotion
        # #req["stream_chunk_size"] = stream_chunk_size  # you can reduce it to get faster response, but degrade quality
        # req["streaming_mode"] = True
        try:
            res = requests.post(
                f"{server_url}/tts",
                json=req,
                stream=True,
            )
            end = time.perf_counter()
            logger.info(f"gpt_sovits Time to make POST: {end-start}s")

            if res.status_code != 200:
                logger.error("Error:%s", res.text)
                return
                
            first = True
        
            for chunk in res.iter_content(chunk_size=None): #12800 1280 32K*20ms*2
                logger.info('chunk len:%d',len(chunk))
                if first:
                    end = time.perf_counter()
                    logger.info(f"gpt_sovits Time to first chunk: {end-start}s")
                    first = False
                if chunk and self.state==State.RUNNING:
                    yield chunk
            #print("gpt_sovits response.elapsed:", res.elapsed)
        except Exception as e:
            logger.exception('sovits')

    def __create_bytes_stream(self,byte_stream):
        #byte_stream=BytesIO(buffer)
        stream, sample_rate = sf.read(byte_stream) # [T*sample_rate,] float64
        logger.info(f'[INFO]tts audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            logger.info(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]
    
        if sample_rate != self.sample_rate and stream.shape[0]>0:
            logger.info(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream

    def stream_tts(self,audio_stream,msg:tuple[str, dict]):
        text,textevent = msg
        first = True
        for chunk in audio_stream:
            if chunk is not None and len(chunk)>0:          
                #stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                #stream = resampy.resample(x=stream, sr_orig=32000, sr_new=self.sample_rate)
                byte_stream=BytesIO(chunk)
                stream = self.__create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx=0
                while streamlen >= self.chunk:
                    eventpoint={}
                    if first:
                        eventpoint={'status':'start','text':text}
                        first = False
                    eventpoint.update(**textevent) 
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk],eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
        eventpoint={'status':'end','text':text}
        eventpoint.update(**textevent) 
        self.parent.put_audio_frame(np.zeros(self.chunk,np.float32),eventpoint)
