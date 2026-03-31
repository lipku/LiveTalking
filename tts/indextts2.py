import os
import time
import numpy as np
import soundfile as sf
import resampy

from utils.logger import logger
from .base_tts import BaseTTS, State
from registry import register

class IndexTTS2(BaseTTS):
    def __init__(self, opt, parent):
        super().__init__(opt, parent)
        # IndexTTS2 配置参数
        self.server_url = opt.TTS_SERVER  # Gradio服务器地址，如 "http://127.0.0.1:7860/"
        self.ref_audio_path = opt.REF_FILE  # 参考音频文件路径
        self.max_tokens = getattr(opt, 'MAX_TOKENS', 120)  # 最大token数
        
        # 初始化Gradio客户端
        try:
            from gradio_client import Client, handle_file
            self.client = Client(self.server_url)
            self.handle_file = handle_file
            logger.info(f"IndexTTS2 Gradio客户端初始化成功: {self.server_url}")
        except ImportError:
            logger.error("IndexTTS2 需要安装 gradio_client: pip install gradio_client")
            raise
        except Exception as e:
            logger.error(f"IndexTTS2 Gradio客户端初始化失败: {e}")
            raise
        
    def txt_to_audio(self, msg):
        text, textevent = msg
        try:
            # 先进行文本分割
            segments = self.split_text(text)
            if not segments:
                logger.error("IndexTTS2 文本分割失败")
                return
            
            logger.info(f"IndexTTS2 文本分割为 {len(segments)} 个片段")
            
            # 循环生成每个片段的音频
            for i, segment_text in enumerate(segments):
                if self.state != State.RUNNING:
                    break
                    
                logger.info(f"IndexTTS2 正在生成第 {i+1}/{len(segments)} 段音频...")
                audio_file = self.indextts2_generate(segment_text)
                
                if audio_file:
                    # 为每个片段创建事件信息
                    segment_msg = (segment_text, textevent)
                    self.file_to_stream(audio_file, segment_msg, is_first=(i==0), is_last=(i==len(segments)-1))
                else:
                    logger.error(f"IndexTTS2 第 {i+1} 段音频生成失败")
                    
        except Exception as e:
            logger.exception(f"IndexTTS2 txt_to_audio 错误: {e}")

    def split_text(self, text):
        """使用 IndexTTS2 API 分割文本"""
        try:
            logger.info(f"IndexTTS2 开始分割文本，长度: {len(text)}")
            
            # 调用文本分割 API
            result = self.client.predict(
                text=text,
                max_text_tokens_per_segment=self.max_tokens,
                api_name="/on_input_text_change"
            )
            
            # 解析分割结果
            if 'value' in result and 'data' in result['value']:
                data = result['value']['data']
                logger.info(f"IndexTTS2 共分割为 {len(data)} 个片段")
                
                segments = []
                for i, item in enumerate(data):
                    序号 = item[0] + 1
                    分句内容 = item[1]
                    token数 = item[2]
                    logger.info(f"片段 {序号}: {len(分句内容)} 字符, {token数} tokens")
                    segments.append(分句内容)
                
                return segments
            else:
                logger.error(f"IndexTTS2 文本分割结果格式异常: {result}")
                return [text]  # 如果分割失败，返回原文本
                
        except Exception as e:
            logger.exception(f"IndexTTS2 文本分割失败: {e}")
            return [text]  # 如果分割失败，返回原文本

    def indextts2_generate(self, text):
        """调用 IndexTTS2 Gradio API 生成语音"""
        start = time.perf_counter()
        
        try:
            # 调用 gen_single API
            result = self.client.predict(
                emo_control_method="Same as the voice reference",
                prompt=self.handle_file(self.ref_audio_path),
                text=text,
                emo_ref_path=self.handle_file(self.ref_audio_path),
                emo_weight=0.8,
                vec1=0.5,
                vec2=0,
                vec3=0,
                vec4=0,
                vec5=0,
                vec6=0,
                vec7=0,
                vec8=0,
                emo_text="",
                emo_random=False,
                max_text_tokens_per_segment=self.max_tokens,
                param_16=True,
                param_17=0.8,
                param_18=30,
                param_19=0.8,
                param_20=0,
                param_21=3,
                param_22=10,
                param_23=1500,
                api_name="/gen_single"
            )
            
            end = time.perf_counter()
            logger.info(f"IndexTTS2 片段生成完成，耗时: {end-start:.2f}s")
            
            # 返回生成的音频文件路径
            if 'value' in result:
                audio_file = result['value']
                return audio_file
            else:
                logger.error(f"IndexTTS2 结果格式异常: {result}")
                return None
                
        except Exception as e:
            logger.exception(f"IndexTTS2 API调用失败: {e}")
            return None

    def file_to_stream(self, audio_file, msg, is_first=False, is_last=False):
        """将音频文件转换为音频流"""
        text, textevent = msg
        
        try:
            # 读取音频文件
            stream, sample_rate = sf.read(audio_file)
            logger.info(f'IndexTTS2 音频文件 {sample_rate}Hz: {stream.shape}')
            
            # 转换为float32
            stream = stream.astype(np.float32)
            
            # 如果是多声道，只取第一个声道
            if stream.ndim > 1:
                logger.info(f'IndexTTS2 音频有 {stream.shape[1]} 个声道，只使用第一个')
                stream = stream[:, 0]
            
            # 重采样到目标采样率
            if sample_rate != self.sample_rate and stream.shape[0] > 0:
                logger.info(f'IndexTTS2 重采样: {sample_rate}Hz -> {self.sample_rate}Hz')
                stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)
            
            # 分块发送音频流
            streamlen = stream.shape[0]
            idx = 0
            first_chunk = True
            
            while streamlen >= self.chunk and self.state == State.RUNNING:
                eventpoint = None
                
                # 只在第一个片段的第一个chunk发送start事件
                if is_first and first_chunk:
                    eventpoint = {'status': 'start', 'text': text, 'msgevent': textevent}
                    first_chunk = False
                
                self.parent.put_audio_frame(stream[idx:idx + self.chunk], eventpoint)
                idx += self.chunk
                streamlen -= self.chunk
            
            # 只在最后一个片段发送end事件
            if is_last:
                eventpoint = {'status': 'end', 'text': text, 'msgevent': textevent}
                self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)
            
            # 清理临时文件
            try:
                if os.path.exists(audio_file):
                    os.remove(audio_file)
                    logger.info(f"IndexTTS2 已删除临时文件: {audio_file}")
            except Exception as e:
                logger.warning(f"IndexTTS2 删除临时文件失败: {e}")
                
        except Exception as e:
            logger.exception(f"IndexTTS2 音频流处理失败: {e}")
