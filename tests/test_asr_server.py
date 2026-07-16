import importlib.util
import sys
import threading
import time
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "server" / "asr_server.py"


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def exception(self, *args, **kwargs):
        pass


def load_asr_server(auto_model):
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

    injected_modules = {
        "utils": fake_utils,
        "utils.logger": fake_logger_module,
        "aiohttp": fake_aiohttp,
        "torch": fake_torch,
        "funasr": fake_funasr,
        "funasr.utils": fake_funasr_utils,
        "funasr.utils.postprocess_utils": fake_postprocess,
        "soundfile": fake_soundfile,
    }
    module_name = f"asr_server_under_test_{time.time_ns()}"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, injected_modules):
        spec.loader.exec_module(module)
    return module, injected_modules


class ASRServerConcurrencyTestCase(unittest.TestCase):
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

        module, injected_modules = load_asr_server(auto_model)
        with patch.dict(sys.modules, injected_modules):
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(module._load_sensevoice)
                self.assertTrue(constructor_started.wait(timeout=1))
                second = pool.submit(module._load_sensevoice)
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

        module, injected_modules = load_asr_server(lambda **options: FakeModel())
        module._sensevoice_model = FakeModel()
        audio = np.zeros(1600, dtype=np.float32)

        with patch.dict(sys.modules, injected_modules):
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(module._run_inference, audio, 16000, False)
                self.assertTrue(first_generate_entered.wait(timeout=1))
                second = pool.submit(module._run_inference, audio, 16000, False)
                try:
                    self.assertFalse(second_generate_entered.wait(timeout=0.2))
                finally:
                    release_generate.set()

                first.result(timeout=2)
                second.result(timeout=2)


if __name__ == "__main__":
    unittest.main()
