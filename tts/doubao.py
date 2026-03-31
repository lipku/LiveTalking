import os
import time
import numpy as np
import copy
import websockets
import gzip
import json
import uuid
import asyncio

from utils.logger import logger
from .base_tts import BaseTTS, State
from registry import register

@register("tts", "doubao")
class DoubaoTTS(BaseTTS):
    def __init__(self, opt, parent):
        super().__init__(opt, parent)
        # 从配置中读取火山引擎参数
        self.appid = os.getenv("DOUBAO_APPID")
        self.token = os.getenv("DOUBAO_TOKEN")
        _cluster = 'volcano_tts'
        _host = "openspeech.bytedance.com"
        self.api_url = f"wss://{_host}/api/v1/tts/ws_binary"
        
        self.request_json = {
            "app": {
                "appid": self.appid,
                "token": "access_token",
                "cluster": _cluster
            },
            "user": {
                "uid": "xxx"
            },
            "audio": {
                "voice_type": "xxx",
                "encoding": "pcm",
                "rate": 16000,
                "speed_ratio": 1.0,
                "volume_ratio": 1.0,
                "pitch_ratio": 1.0,
            },
            "request": {
                "reqid": "xxx",
                "text": "字节跳动语音合成。",
                "text_type": "plain",
                "operation": "xxx"
            }
        }

    async def doubao_voice(self, text, ref_file): # -> Iterator[bytes]:
        start = time.perf_counter()
        voice_type = ref_file #self.opt.REF_FILE

        try:
            # 创建请求对象
            default_header = bytearray(b'\x11\x10\x11\x00')
            submit_request_json = copy.deepcopy(self.request_json)
            submit_request_json["user"]["uid"] = self.parent.sessionid
            submit_request_json["audio"]["voice_type"] = voice_type
            submit_request_json["request"]["text"] = text
            submit_request_json["request"]["reqid"] = str(uuid.uuid4())
            submit_request_json["request"]["operation"] = "submit"
            payload_bytes = str.encode(json.dumps(submit_request_json))
            payload_bytes = gzip.compress(payload_bytes)  # if no compression, comment this line
            full_client_request = bytearray(default_header)
            full_client_request.extend((len(payload_bytes)).to_bytes(4, 'big'))  # payload size(4 bytes)
            full_client_request.extend(payload_bytes)  # payload

            header = {"Authorization": f"Bearer; {self.token}"}
            first = True
            async with websockets.connect(self.api_url, extra_headers=header, ping_interval=None) as ws:
                await ws.send(full_client_request)
                while True:
                    res = await ws.recv()
                    header_size = res[0] & 0x0f
                    message_type = res[1] >> 4
                    message_type_specific_flags = res[1] & 0x0f
                    payload = res[header_size*4:]

                    if message_type == 0xb:  # audio-only server response
                        if message_type_specific_flags == 0:  # no sequence number as ACK
                            #print("                Payload size: 0")
                            continue
                        else:
                            if first:
                                end = time.perf_counter()
                                logger.info(f"doubao tts Time to first chunk: {end-start}s")
                                first = False
                            sequence_number = int.from_bytes(payload[:4], "big", signed=True)
                            payload_size = int.from_bytes(payload[4:8], "big", signed=False)
                            payload = payload[8:]
                            yield payload
                        if sequence_number < 0:
                            break
                    else:
                        break
        except Exception as e:
            logger.exception('doubao')
        # # 检查响应状态码
        # if response.status_code == 200:
        #     # 处理响应数据
        #     audio_data = base64.b64decode(response.json().get('data'))
        #     yield audio_data
        # else:
        #     logger.error(f"请求失败，状态码: {response.status_code}")
        #     return

    def txt_to_audio(self, msg:tuple[str, dict]):
        text, textevent = msg
        ref_file = textevent.get('tts', {}).get('ref_file',self.opt.REF_FILE)
        #ref_text = textevent.get('tts', {}).get('ref_text')
        asyncio.new_event_loop().run_until_complete(
            self.stream_tts(
                self.doubao_voice(text, ref_file),
                msg
            )
        )

    async def stream_tts(self, audio_stream, msg:tuple[str, dict]):
        text, textevent = msg
        first = True
        last_stream = np.array([],dtype=np.float32)
        async for chunk in audio_stream:
            if chunk is not None and len(chunk) > 0:
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = np.concatenate((last_stream,stream))
                #stream = resampy.resample(x=stream, sr_orig=24000, sr_new=self.sample_rate)
                # byte_stream=BytesIO(buffer)
                # stream = self.__create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx = 0
                while streamlen >= self.chunk:
                    eventpoint = {}
                    if first:
                        eventpoint={'status':'start','text':text}
                        first = False
                    eventpoint.update(**textevent) 
                    self.parent.put_audio_frame(stream[idx:idx + self.chunk], eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
                last_stream = stream[idx:] #get the remain stream
        eventpoint={'status':'end','text':text}
        eventpoint.update(**textevent) 
        self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)
