###############################################################################
#  FunASR/SenseVoice STT Plugin — registry category "stt", name "funasr"
#
#  Engine code migrated verbatim from server/asr_server.py: lazy-loading
#  singleton + double-checked locks (model construction and inference are
#  each serialized), model iic/SenseVoiceSmall with fsmn-vad.
#
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  Licensed under the Apache License, Version 2.0
###############################################################################

import io
import time
import threading

import numpy as np

from registry import register
from utils.logger import logger

from .base_stt import BaseSTT


@register("stt", "funasr")
class FunASRSTT(BaseSTT):
    """Local SenseVoice/FunASR speech-recognition engine.

    Created via the registry: ``registry.create("stt", "funasr")``.
    """

    def __init__(self):
        self._model = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    def is_available(self) -> bool:
        """Return True if the ``funasr`` package is importable."""
        try:
            import funasr  # noqa: F401
            return True
        except ImportError:
            return False

    def ensure_loaded(self):
        """
        Load the SenseVoice model on first call (lazy singleton).
        Concurrent first requests must share the same model initialization.
        """
        if self._model is not None:
            return self._model

        with self._load_lock:
            if self._model is not None:
                return self._model

            import torch
            from funasr import AutoModel

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            logger.info(
                f"[ASR] Loading SenseVoiceSmall on device='{device}' "
                f"(first run will download ~500MB from ModelScope)..."
            )

            t0 = time.perf_counter()
            self._model = AutoModel(
                model="iic/SenseVoiceSmall",
                vad_model="fsmn-vad",
                vad_kwargs={"max_single_segment_time": 30000},
                device=device,
                trust_remote_code=True,
            )
            elapsed = time.perf_counter() - t0
            logger.info(
                f"[ASR] ✅ SenseVoiceSmall ready — loaded in {elapsed:.1f}s on {device}"
            )
        return self._model

    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000,
                   use_itn: bool = False, mode: str = "offline") -> dict:
        """
        Run SenseVoice inference on a PCM16 mono audio segment and build the
        /api/asr reply payload.

        ``audio_bytes`` is little-endian 16-bit PCM (as received on the
        /api/asr WebSocket); it is converted to float32 in [-1, 1] here.
        Segments shorter than 20 ms are skipped silently (empty text).

        This is a **blocking** call — always invoke from ``run_in_executor``.

        Returns
        -------
        dict
            {"text": str, "mode": str, "is_final": True, "timestamp": None}
        """
        if len(audio_bytes) < 640:  # < 20 ms of audio — skip
            logger.warning("[ASR] Audio too short (< 20ms), returning empty")
            return {"text": "", "mode": mode, "is_final": True, "timestamp": None}

        import soundfile as sf
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        # Convert PCM16 → float32 in [-1, 1]
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        model = self.ensure_loaded()

        # Write to in-memory WAV so funasr can read the sample rate from the header
        wav_buf = io.BytesIO()
        sf.write(wav_buf, audio_float32, sample_rate, format="WAV")
        wav_buf.seek(0)

        t0 = time.perf_counter()
        with self._infer_lock:
            res = model.generate(
                input=wav_buf,
                cache={},
                language="auto",
                use_itn=use_itn,
                batch_size_s=60,
            )
        inference_ms = (time.perf_counter() - t0) * 1000

        text = ""
        if res and len(res) > 0 and res[0].get("text"):
            text = rich_transcription_postprocess(res[0]["text"])

        audio_duration_s = len(audio_float32) / sample_rate

        logger.info(
            f"[ASR] ✅ SenseVoice inference complete\n"
            f"       ├─ Latency     : {inference_ms:>8.0f} ms\n"
            f"       ├─ Audio length: {audio_duration_s:>8.1f} s\n"
            f"       ├─ RTF         : {inference_ms / 1000 / max(audio_duration_s, 0.001):>8.3f}\n"
            f"       └─ Text        : \"{text[:100]}{'…' if len(text) > 100 else ''}\""
        )

        # Map the client mode to the response mode the frontend expects
        if mode == "2pass":
            response_mode = "2pass-offline"
        else:
            response_mode = mode  # "online" or "offline"

        return {
            "text": text,
            "mode": response_mode,
            "is_final": True,
            "timestamp": None,
        }
