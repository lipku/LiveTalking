###############################################################################
#  ASR WebSocket Server — Local SenseVoice/FunASR Integration
#
#  Resolves: https://github.com/lipku/LiveTalking/issues/604
#
#  This module provides a WebSocket endpoint (/api/asr) that speaks the same
#  protocol as the external FunASR server (wss://www.funasr.com:10096/).
#  The browser client (web/asr/main.js) can connect here instead, keeping
#  all ASR processing local and cutting ~600ms of network + Whisper latency.
#
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  Licensed under the Apache License, Version 2.0
###############################################################################

import json
import time
import io
import asyncio
import numpy as np
from aiohttp import web

from utils.logger import logger


# ─── Lazy Model Loader ────────────────────────────────────────────────────

_sensevoice_model = None


def _load_sensevoice():
    """
    Load the SenseVoice model on first call (lazy singleton).
    Thread-safe via the GIL — only one thread will enter the init block.
    """
    global _sensevoice_model
    if _sensevoice_model is not None:
        return _sensevoice_model

    import torch
    from funasr import AutoModel

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(
        f"[ASR] Loading SenseVoiceSmall on device='{device}' "
        f"(first run will download ~500MB from ModelScope)..."
    )

    t0 = time.perf_counter()
    _sensevoice_model = AutoModel(
        model="iic/SenseVoiceSmall",
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device=device,
        trust_remote_code=True,
    )
    elapsed = time.perf_counter() - t0
    logger.info(f"[ASR] ✅ SenseVoiceSmall ready — loaded in {elapsed:.1f}s on {device}")
    return _sensevoice_model


def _run_inference(audio_float32: np.ndarray, sample_rate: int, use_itn: bool):
    """
    Run SenseVoice inference on a float32 audio array.

    This is a **blocking** call — always invoke from ``run_in_executor``.

    Returns
    -------
    tuple[str, float, float]
        (transcribed_text, inference_ms, audio_duration_s)
    """
    import soundfile as sf
    from funasr.utils.postprocess_utils import rich_transcription_postprocess

    model = _load_sensevoice()

    # Write to in-memory WAV so funasr can read the sample rate from the header
    wav_buf = io.BytesIO()
    sf.write(wav_buf, audio_float32, sample_rate, format="WAV")
    wav_buf.seek(0)

    t0 = time.perf_counter()
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

    return text, inference_ms, audio_duration_s


# ─── WebSocket Handler ─────────────────────────────────────────────────────

SAMPLE_RATE = 16000  # The browser client records at 16 kHz mono PCM16


async def asr_websocket_handler(request):
    """
    WebSocket handler implementing the FunASR client protocol.

    Protocol flow
    -------------
    1. Client opens connection
    2. Client sends JSON config::

           {"chunk_size":[5,10,5], "wav_name":"h5",
            "is_speaking":true, "mode":"2pass", "itn":false, ...}

    3. Client streams binary PCM16 audio chunks (960 bytes = 60 ms @ 16 kHz)
    4. Client sends stop signal::

           {"is_speaking":false, ...}

    5. Server responds with transcription::

           {"text":"hello world", "mode":"2pass-offline",
            "is_final":true, "timestamp":null}
    """
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    client_ip = request.remote
    logger.info(f"[ASR] 🔌 WebSocket connected from {client_ip}")

    audio_buffer = bytearray()
    config: dict = {}
    session_start = time.perf_counter()
    chunks_received = 0

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    logger.warning("[ASR] Received invalid JSON, ignoring")
                    continue

                if data.get("is_speaking") is True:
                    # ── Session start ──────────────────────────────────
                    config = data
                    audio_buffer = bytearray()
                    chunks_received = 0
                    session_start = time.perf_counter()
                    logger.info(
                        f"[ASR] 🎙️  Recording started | "
                        f"mode={config.get('mode', 'offline')} | "
                        f"itn={config.get('itn', False)} | "
                        f"hotwords={bool(config.get('hotwords'))}"
                    )

                elif data.get("is_speaking") is False:
                    # ── End of speech → run inference ──────────────────
                    buf_bytes = len(audio_buffer)
                    audio_seconds = buf_bytes / (SAMPLE_RATE * 2)  # 2 bytes per int16
                    session_elapsed = time.perf_counter() - session_start

                    logger.info(
                        f"[ASR] 🛑 Recording stopped | "
                        f"{chunks_received} chunks | "
                        f"{buf_bytes:,} bytes | "
                        f"{audio_seconds:.1f}s audio | "
                        f"session wall time {session_elapsed:.1f}s"
                    )

                    if buf_bytes < 640:  # < 20 ms of audio — skip
                        logger.warning("[ASR] Audio too short (< 20ms), returning empty")
                        await ws.send_str(json.dumps({
                            "text": "",
                            "mode": config.get("mode", "offline"),
                            "is_final": True,
                            "timestamp": None,
                        }))
                        continue

                    # Ensure even number of bytes for int16 conversion
                    if buf_bytes % 2 != 0:
                        logger.warning(f"[ASR] Odd number of bytes received ({buf_bytes}), dropping incomplete sample")
                        audio_buffer = audio_buffer[:-1]
                        buf_bytes -= 1

                    # Convert PCM16 → float32 in [-1, 1]
                    audio_int16 = np.frombuffer(bytes(audio_buffer), dtype=np.int16)
                    audio_float32 = audio_int16.astype(np.float32) / 32768.0
                    use_itn = config.get("itn", False)

                    # Offload blocking inference to a thread
                    loop = asyncio.get_event_loop()
                    try:
                        text, inference_ms, audio_dur = await loop.run_in_executor(
                            None,
                            _run_inference,
                            audio_float32,
                            SAMPLE_RATE,
                            use_itn,
                        )
                    except Exception as e:
                        logger.exception(f"[ASR] ❌ Inference failed: {e}")
                        text = ""

                    # Map the client mode to the response mode the frontend expects
                    mode = config.get("mode", "offline")
                    if mode == "2pass":
                        response_mode = "2pass-offline"
                    else:
                        response_mode = mode  # "online" or "offline"

                    await ws.send_str(json.dumps({
                        "text": text,
                        "mode": response_mode,
                        "is_final": True,
                        "timestamp": None,
                    }))
                    logger.info(f"[ASR] 📤 Result sent to client (mode={response_mode})")

            elif msg.type == web.WSMsgType.BINARY:
                audio_buffer.extend(msg.data)
                chunks_received += 1

            elif msg.type in (web.WSMsgType.ERROR, web.WSMsgType.CLOSE):
                break

    except asyncio.CancelledError:
        logger.info("[ASR] WebSocket handler cancelled")
    except Exception as e:
        logger.exception(f"[ASR] ❌ WebSocket handler error: {e}")

    logger.info(f"[ASR] 🔌 WebSocket disconnected ({client_ip})")
    return ws


# ─── Availability Check ───────────────────────────────────────────────────

def is_funasr_available() -> bool:
    """Return True if the ``funasr`` package is importable."""
    try:
        import funasr  # noqa: F401
        return True
    except ImportError:
        return False
