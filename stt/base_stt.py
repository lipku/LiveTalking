###############################################################################
#  STT 插件基类 — 用户语音识别（STT）引擎的约定接口
#
#  注意区分：stt/（经 /api/asr 暴露）做的是用户语音识别（STT）；
#  avatars/audio_features/base_asr.py 是口型特征提取，与语音识别无关。
###############################################################################


class BaseSTT:
    """语音识别（STT）引擎的约定接口。

    STT 插件通过 ``@register("stt", name)`` 注册到 registry，由
    server/asr_server.py 统一创建并驱动（当前唯一引擎：funasr）。

    约定接口：
    - ``ensure_loaded()``：懒加载模型（阻塞），返回底层模型对象；
    - ``transcribe(audio_bytes, ...)``：整段识别（阻塞），返回至少含
      ``{"text": str}`` 的 dict；
    - ``is_available()``：引擎依赖是否可用，不可用则 /api/asr 不挂载。
    """

    def ensure_loaded(self):
        raise NotImplementedError

    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000,
                   use_itn: bool = False) -> dict:
        raise NotImplementedError

    def is_available(self) -> bool:
        raise NotImplementedError
