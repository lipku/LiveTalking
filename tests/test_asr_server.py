"""Tests for the pluggable STT engine (stt/funasr.py) and the /api/asr WebSocket handler.

零重依赖：torch / funasr / soundfile / aiohttp / utils.logger（registry 与
stt 插件 import 链上）均以假模块注入 sys.modules，仅 numpy 依赖真实安装
（与旧版测试一致）。可直接运行：python tests/test_asr_server.py
"""

import asyncio
import importlib.util
import json
import sys
import threading
import time
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

# 仅 numpy 依赖真实安装（与旧版测试一致）。必须在任何 patch.dict 之前导入：
# patch.dict 退出时会 clear+恢复快照，若 numpy 首次导入发生在 patch 上下文内，
# 退出后它会被移出 sys.modules，而下一次 import 会触发扩展模块重复初始化
# （ImportError: cannot load module more than once per process）。
import numpy  # noqa: F401


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def exception(self, *args, **kwargs):
        pass


def build_fake_modules(auto_model):
    fake_utils = types.ModuleType("utils")
    fake_logger_module = types.ModuleType("utils.logger")
    fake_logger_module.logger = FakeLogger()

    fake_aiohttp = types.ModuleType("aiohttp")
    fake_aiohttp.web = types.SimpleNamespace(
        WebSocketResponse=object,
        WSMsgType=types.SimpleNamespace(
            TEXT="TEXT",
            BINARY="BINARY",
            ERROR="ERROR",
            CLOSE="CLOSE",
        ),
    )

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)

    fake_funasr = types.ModuleType("funasr")
    fake_funasr.AutoModel = auto_model
    fake_funasr_utils = types.ModuleType("funasr.utils")
    fake_postprocess = types.ModuleType("funasr.utils.postprocess_utils")
    fake_postprocess.rich_transcription_postprocess = lambda text: text

    fake_soundfile = types.ModuleType("soundfile")
    fake_soundfile.write = lambda *args, **kwargs: None

    return {
        "utils": fake_utils,
        "utils.logger": fake_logger_module,
        "aiohttp": fake_aiohttp,
        "torch": fake_torch,
        "funasr": fake_funasr,
        "funasr.utils": fake_funasr_utils,
        "funasr.utils.postprocess_utils": fake_postprocess,
        "soundfile": fake_soundfile,
    }


def _load_repo_module(name, relpath):
    """以规范模块名加载仓库内的 py 文件，使绝对/相对导入都解析到 sys.modules。"""
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_stt_and_asr_server(auto_model):
    """加载 registry + stt 插件 + server/asr_server.py（隔离环境，零重依赖）。

    返回 (asr_server, stt_funasr, injected_modules)；后续调用引擎方法时，
    需以 ``patch.dict(sys.modules, injected_modules)`` 提供假依赖。
    """
    injected = build_fake_modules(auto_model)
    asr_name = f"asr_server_under_test_{time.time_ns()}"
    with patch.dict(sys.modules, injected):
        _load_repo_module("registry", "registry.py")
        stt_pkg = types.ModuleType("stt")
        stt_pkg.__path__ = [str(REPO_ROOT / "stt")]
        sys.modules["stt"] = stt_pkg
        _load_repo_module("stt.base_stt", "stt/base_stt.py")
        stt_funasr = _load_repo_module("stt.funasr", "stt/funasr.py")
        asr_server = _load_repo_module(asr_name, "server/asr_server.py")
    return asr_server, stt_funasr, injected


class FunASRSTTConcurrencyTestCase(unittest.TestCase):
    """引擎行为（等价迁移自旧版 asr_server 测试）：懒加载单例 + 推理串行化。"""

    def test_lazy_model_load_constructs_one_model_across_threads(self):
        constructor_started = threading.Event()
        release_constructor = threading.Event()
        constructor_calls = []
        calls_lock = threading.Lock()

        class FakeModel:
            pass

        def auto_model(**options):
            with calls_lock:
                constructor_calls.append(options)
                constructor_started.set()
            release_constructor.wait(timeout=2)
            return FakeModel()

        _, stt_funasr, injected = load_stt_and_asr_server(auto_model)
        engine = stt_funasr.FunASRSTT()

        with patch.dict(sys.modules, injected):
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(engine.ensure_loaded)
                self.assertTrue(constructor_started.wait(timeout=1))
                second = pool.submit(engine.ensure_loaded)
                try:
                    time.sleep(0.1)
                    self.assertEqual(len(constructor_calls), 1)
                finally:
                    release_constructor.set()

                first_model = first.result(timeout=2)
                second_model = second.result(timeout=2)

        self.assertIs(first_model, second_model)

    def test_shared_model_generate_is_serialized_across_threads(self):
        first_generate_entered = threading.Event()
        second_generate_entered = threading.Event()
        release_generate = threading.Event()
        state_lock = threading.Lock()
        active_calls = 0

        class FakeModel:
            def generate(self, **options):
                nonlocal active_calls
                with state_lock:
                    active_calls += 1
                    if active_calls == 1:
                        first_generate_entered.set()
                    else:
                        second_generate_entered.set()
                try:
                    release_generate.wait(timeout=2)
                    return [{"text": "ok"}]
                finally:
                    with state_lock:
                        active_calls -= 1

        _, stt_funasr, injected = load_stt_and_asr_server(lambda **options: FakeModel())
        engine = stt_funasr.FunASRSTT()
        engine._model = FakeModel()
        audio = b"\x00\x00" * 1600  # 1600 samples of PCM16 silence

        with patch.dict(sys.modules, injected):
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(engine.transcribe, audio, 16000, False)
                self.assertTrue(first_generate_entered.wait(timeout=1))
                second = pool.submit(engine.transcribe, audio, 16000, False)
                try:
                    self.assertFalse(second_generate_entered.wait(timeout=0.2))
                finally:
                    release_generate.set()

                first.result(timeout=2)
                second.result(timeout=2)


class FakeMsg:
    def __init__(self, msg_type, data):
        self.type = msg_type
        self.data = data


class FakeWebSocket:
    """最小 WS 桩：__aiter__ 依次吐出预设消息，send_str 记录服务端回复。"""

    def __init__(self, messages):
        self._messages = list(messages)
        self.sent = []

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._messages:
            raise StopAsyncIteration
        return self._messages.pop(0)

    async def send_str(self, payload):
        self.sent.append(payload)


class ASRProtocolContractTestCase(unittest.TestCase):
    """/api/asr 协议契约：配置 JSON → 二进制帧 → is_speaking:false → 识别结果。"""

    def test_two_pass_offline_message_flow(self):
        class FakeModel:
            def generate(self, **options):
                return [{"text": "hello world"}]

        asr_server, _, injected = load_stt_and_asr_server(lambda **options: FakeModel())

        config_frame = json.dumps({
            "chunk_size": [5, 10, 5],
            "wav_name": "h5",
            "is_speaking": True,
            "mode": "2pass",
            "itn": False,
        })
        pcm_frame = b"\x01\x00" * 480  # 960 bytes = 60 ms @ 16 kHz PCM16
        stop_frame = json.dumps({"is_speaking": False})

        ws = FakeWebSocket([
            FakeMsg("TEXT", config_frame),
            FakeMsg("BINARY", pcm_frame),
            FakeMsg("TEXT", stop_frame),
        ])

        with patch.dict(sys.modules, injected):
            engine = asr_server.init_asr_engine("funasr")
            self.assertIsNotNone(engine)
            self.assertTrue(asr_server.is_funasr_available())
            asyncio.run(asr_server._run_asr_session(ws, "127.0.0.1"))

        self.assertEqual(len(ws.sent), 1)
        self.assertEqual(json.loads(ws.sent[0]), {
            "text": "hello world",
            "mode": "2pass-offline",
            "is_final": True,
            "timestamp": None,
        })


if __name__ == "__main__":
    unittest.main()
