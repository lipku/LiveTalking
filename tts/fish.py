import time
import numpy as np
import resampy
import requests
from typing import Iterator

from utils.logger import logger
from .base_tts import BaseTTS, State
from registry import register

@register("tts", "fishtts")
class FishTTS(BaseTTS):
    def txt_to_audio(self,msg:tuple[str, dict]): 
        text,textevent = msg
        ref_file = textevent.get('tts', {}).get('ref_file',self.opt.REF_FILE)
        ref_text = textevent.get('tts', {}).get('ref_text',self.opt.REF_TEXT)
        self.stream_tts(
            self.fish_speech(
                text,
                ref_file,  
                ref_text,
                "zh", #en args.language,
                self.opt.TTS_SERVER, #"http://127.0.0.1:5000", #args.server_url,
            ),
            msg
        )

    def fish_speech(self, text, reffile, reftext,language, server_url) -> Iterator[bytes]:
        start = time.perf_counter()
        req={
            'text':text,
            'reference_id':reffile,
            'format':'wav',
            'streaming':True,
            'use_memory_cache':'on'
        }
        try:
            res = requests.post(
                f"{server_url}/v1/tts",
                json=req,
                stream=True,
                headers={
                    "content-type": "application/json",
                },
            )
            end = time.perf_counter()
            logger.info(f"fish_speech Time to make POST: {end-start}s")

            if res.status_code != 200:
                logger.error("Error:%s", res.text)
                return
                
            first = True
        
            for chunk in res.iter_content(chunk_size=17640): # 1764 44100*20ms*2
                #print('chunk len:',len(chunk))
                if first:
                    end = time.perf_counter()
                    logger.info(f"fish_speech Time to first chunk: {end-start}s")
                    first = False
                if chunk and self.state==State.RUNNING:
                    yield chunk
            #print("gpt_sovits response.elapsed:", res.elapsed)
        except Exception as e:
            logger.exception('fishtts')

    def stream_tts(self,audio_stream,msg:tuple[str, dict]):
        text,textevent = msg
        first = True
        for chunk in audio_stream:
            if chunk is not None and len(chunk)>0:          
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = resampy.resample(x=stream, sr_orig=44100, sr_new=self.sample_rate)
                #byte_stream=BytesIO(buffer)
                #stream = self.__create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx=0
                while streamlen >= self.chunk:
                    eventpoint={}
                    if first:
                        eventpoint={'status':'start','text':text}
                        first = False
                    eventpoint.update(**textevent) #eventpoint={'status':'start','text':text,'msgevent':textevent}
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk],eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
        eventpoint={'status':'end','text':text}
        eventpoint.update(**textevent) #eventpoint={'status':'end','text':text,'msgevent':textevent}
        self.parent.put_audio_frame(np.zeros(self.chunk,np.float32),eventpoint)
