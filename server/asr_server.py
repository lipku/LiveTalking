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
#  The engine itself is a pluggable STT plugin (registry category "stt",
#  see stt/); FunASR/SenseVoice is the first registered engine ("funasr"),
#  selected via the --asr option.
#
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  Licensed under the Apache License, Version 2.0
###############################################################################

import json
import time
import asyncio

from aiohttp import web

import registry
import stt.funasr  # noqa: F401  # importing registers the ("stt", "funasr") plugin
from utils.logger import logger


# ─── Engine Bootstrap ──────────────────────────────────────────────────────

_engine = None


def init_asr_engine(engine_name: str = "funasr"):
    """Create the STT engine via the registry (once per process).

    Called once at server startup with the --asr option (see server/routes.py).
    On failure (unknown plugin / missing dependency) this degrades silently:
    the engine stays None, /api/asr is not registered and only an info log
    is emitted — same behaviour as the old funasr-availability check.
    """
    global _engine
    if _engine is not None:
        return _engine
    try:
        _engine = registry.create("stt", engine_name)
    except Exception as e:
        logger.info(f"[ASR] STT engine '{engine_name}' unavailable: {e}")
    return _engine


def is_funasr_available() -> bool:
    """Return True if an STT engine was created and its deps are importable."""
    return _engine is not None and _engine.is_available()


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
    await _run_asr_session(ws, request.remote)
    return ws


async def _run_asr_session(ws, client_ip):
    """Message loop of one /api/asr session (protocol documented above)."""
    if _engine is None:
        logger.warning("[ASR] No STT engine initialized, closing connection")
        return

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

                    # Ensure even number of bytes for int16 conversion
                    if buf_bytes % 2 != 0:
                        logger.warning(f"[ASR] Odd number of bytes received ({buf_bytes}), dropping incomplete sample")
                        audio_buffer = audio_buffer[:-1]
                        buf_bytes -= 1

                    use_itn = config.get("itn", False)

                    # Offload blocking inference to a thread (the engine does
                    # the short-audio skip, PCM16 → float32 conversion and
                    # builds the reply payload)
                    loop = asyncio.get_event_loop()
                    try:
                        reply = await loop.run_in_executor(
                            None,
                            _engine.transcribe,
                            bytes(audio_buffer),
                            SAMPLE_RATE,
                            use_itn,
                            config.get("mode", "offline"),
                        )
                    except Exception as e:
                        logger.exception(f"[ASR] ❌ Inference failed: {e}")
                        mode = config.get("mode", "offline")
                        reply = {
                            "text": "",
                            "mode": "2pass-offline" if mode == "2pass" else mode,
                            "is_final": True,
                            "timestamp": None,
                        }

                    await ws.send_str(json.dumps(reply))
                    logger.info(f"[ASR] 📤 Result sent to client (mode={reply['mode']})")

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
