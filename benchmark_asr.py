#!/usr/bin/env python3
###############################################################################
#  ASR Latency Benchmark — SenseVoice Small vs OpenAI Whisper Tiny
#
#  Usage:
#      python benchmark_asr.py                    # uses TTS-generated speech
#      python benchmark_asr.py --audio test.wav   # uses your own audio file
#      python benchmark_asr.py --runs 10          # more runs for stability
#
#  Resolves: https://github.com/lipku/LiveTalking/issues/604
###############################################################################

import argparse
import asyncio
import io
import os
import sys
import time
import tempfile

import numpy as np
import soundfile as sf


# ─── Utilities ─────────────────────────────────────────────────────────────

def generate_test_audio_tts(text: str, output_path: str) -> bool:
    """Generate real speech audio using edge_tts (requires network)."""
    try:
        import edge_tts

        async def _gen():
            communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
            await communicate.save(output_path)

        asyncio.run(_gen())
        print(f"  ✅ Generated speech audio via edge_tts → {output_path}")
        return True
    except Exception as e:
        print(f"  ⚠️  edge_tts failed ({e}), falling back to synthetic audio")
        return False


def generate_synthetic_audio(duration_s: float, sample_rate: int = 16000) -> np.ndarray:
    """Generate a synthetic audio signal (speech-like noise burst)."""
    rng = np.random.default_rng(42)
    # Mix of tones + noise to loosely simulate speech energy distribution
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), dtype=np.float32)
    audio = (
        0.3 * np.sin(2 * np.pi * 200 * t) +
        0.2 * np.sin(2 * np.pi * 500 * t) +
        0.1 * np.sin(2 * np.pi * 1200 * t) +
        0.15 * rng.standard_normal(len(t)).astype(np.float32)
    )
    # Fade in/out
    fade = int(0.05 * sample_rate)
    audio[:fade] *= np.linspace(0, 1, fade)
    audio[-fade:] *= np.linspace(1, 0, fade)
    return audio.astype(np.float32)


def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    """Load and resample audio to target sample rate."""
    import librosa
    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    return audio.astype(np.float32)


def format_table(headers, rows, col_widths=None):
    """Simple markdown table formatter."""
    if col_widths is None:
        col_widths = [max(len(str(h)), max(len(str(r[i])) for r in rows))
                      for i, h in enumerate(headers)]

    def fmt_row(cells):
        return "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, col_widths)) + " |"

    sep = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|"
    lines = [fmt_row(headers), sep] + [fmt_row(r) for r in rows]
    return "\n".join(lines)


# ─── Whisper Tiny Benchmark ───────────────────────────────────────────────

def benchmark_whisper_tiny(audio: np.ndarray, sample_rate: int, num_runs: int):
    """Benchmark OpenAI Whisper Tiny for speech-to-text."""
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    print("\n🔧 Loading Whisper Tiny (openai/whisper-tiny)...")
    t0 = time.perf_counter()
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    model.eval()
    load_time = time.perf_counter() - t0
    print(f"   Loaded in {load_time:.1f}s")

    # Warm-up run (not timed)
    print("   Warm-up run...")
    input_features = processor(
        audio, sampling_rate=sample_rate, return_tensors="pt"
    ).input_features
    with torch.no_grad():
        model.generate(input_features, max_new_tokens=128)

    # Timed runs
    times = []
    last_text = ""
    for i in range(num_runs):
        t = time.perf_counter()
        input_features = processor(
            audio, sampling_rate=sample_rate, return_tensors="pt"
        ).input_features
        with torch.no_grad():
            predicted_ids = model.generate(input_features, max_new_tokens=128)
        last_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        elapsed = (time.perf_counter() - t) * 1000
        times.append(elapsed)
        print(f"   Run {i+1}/{num_runs}: {elapsed:.0f}ms")

    return times, last_text


# ─── SenseVoice Benchmark ─────────────────────────────────────────────────

def benchmark_sensevoice(audio: np.ndarray, sample_rate: int, num_runs: int):
    """Benchmark SenseVoice Small for speech-to-text."""
    import torch
    from funasr import AutoModel
    from funasr.utils.postprocess_utils import rich_transcription_postprocess

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n🔧 Loading SenseVoice Small (iic/SenseVoiceSmall) on {device}...")
    t0 = time.perf_counter()
    model = AutoModel(
        model="iic/SenseVoiceSmall",
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device=device,
        trust_remote_code=True,
    )
    load_time = time.perf_counter() - t0
    print(f"   Loaded in {load_time:.1f}s")

    # Prepare WAV in memory
    wav_buf = io.BytesIO()
    sf.write(wav_buf, audio, sample_rate, format="WAV")

    # Warm-up run
    print("   Warm-up run...")
    wav_buf.seek(0)
    model.generate(input=wav_buf, cache={}, language="auto", use_itn=True, batch_size_s=60)

    # Timed runs
    times = []
    last_text = ""
    for i in range(num_runs):
        wav_buf.seek(0)
        t = time.perf_counter()
        res = model.generate(
            input=wav_buf, cache={}, language="auto", use_itn=True, batch_size_s=60
        )
        elapsed = (time.perf_counter() - t) * 1000
        last_text = rich_transcription_postprocess(res[0]["text"]) if res and res[0].get("text") else ""
        times.append(elapsed)
        print(f"   Run {i+1}/{num_runs}: {elapsed:.0f}ms")

    return times, last_text


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ASR Latency Benchmark: SenseVoice Small vs Whisper Tiny"
    )
    parser.add_argument("--audio", type=str, default=None,
                        help="Path to a WAV/MP3 audio file (default: auto-generate)")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of timed inference runs per model (default: 5)")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Duration of generated test audio in seconds (default: 5)")
    args = parser.parse_args()

    sample_rate = 16000
    print("=" * 60)
    print("  ASR Latency Benchmark")
    print("  SenseVoice Small  vs  OpenAI Whisper Tiny")
    print("=" * 60)

    import torch
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"\n📊 Device: {device}")
    print(f"📊 Runs per model: {args.runs}")

    # ── Prepare test audio ──────────────────────────────────────────────
    if args.audio and os.path.exists(args.audio):
        print(f"\n🎵 Loading audio from: {args.audio}")
        audio = load_audio(args.audio, target_sr=sample_rate)
    else:
        print(f"\n🎵 Generating {args.duration}s test audio...")
        # Try edge_tts first for real speech
        tts_text = "Hello, this is a benchmark test for speech recognition latency. The quick brown fox jumps over the lazy dog."
        tmp_wav = os.path.join(tempfile.gettempdir(), "benchmark_tts_test.wav")
        if generate_test_audio_tts(tts_text, tmp_wav):
            audio = load_audio(tmp_wav, target_sr=sample_rate)
        else:
            audio = generate_synthetic_audio(args.duration, sample_rate)
            print(f"  ✅ Generated {args.duration}s synthetic audio")

    audio_duration = len(audio) / sample_rate
    print(f"   Audio duration: {audio_duration:.1f}s | Samples: {len(audio):,} | SR: {sample_rate}")

    # ── Benchmark Whisper Tiny ──────────────────────────────────────────
    whisper_times, whisper_text = benchmark_whisper_tiny(audio, sample_rate, args.runs)

    # ── Benchmark SenseVoice ────────────────────────────────────────────
    sv_times, sv_text = benchmark_sensevoice(audio, sample_rate, args.runs)

    # ── Results ─────────────────────────────────────────────────────────
    def stats(times):
        return np.mean(times), np.min(times), np.max(times), np.std(times)

    w_avg, w_min, w_max, w_std = stats(whisper_times)
    s_avg, s_min, s_max, s_std = stats(sv_times)
    speedup = w_avg / s_avg if s_avg > 0 else float("inf")
    savings = w_avg - s_avg

    print("\n")
    print("=" * 60)
    print("  📊 RESULTS")
    print("=" * 60)
    print(f"\n  Audio: {audio_duration:.1f}s @ {sample_rate}Hz | Device: {device}")
    print(f"  Runs : {args.runs} (after 1 warm-up)\n")

    headers = ["Model", "Avg (ms)", "Min (ms)", "Max (ms)", "Std (ms)", "RTF"]
    rows = [
        ["Whisper Tiny", f"{w_avg:.0f}", f"{w_min:.0f}", f"{w_max:.0f}",
         f"{w_std:.0f}", f"{w_avg/1000/audio_duration:.3f}"],
        ["SenseVoice Small", f"{s_avg:.0f}", f"{s_min:.0f}", f"{s_max:.0f}",
         f"{s_std:.0f}", f"{s_avg/1000/audio_duration:.3f}"],
    ]
    print(format_table(headers, rows))

    print(f"\n  🚀 SenseVoice is {speedup:.1f}x faster than Whisper Tiny")
    print(f"  ⏱️  Latency savings: ~{savings:.0f}ms per utterance")

    print(f"\n  Whisper output   : \"{whisper_text[:80]}\"")
    print(f"  SenseVoice output: \"{sv_text[:80]}\"")

    # ── Markdown output for PR ──────────────────────────────────────────
    print("\n\n--- Markdown for PR description ---\n")
    print(f"### ASR Latency Benchmark (Issue #604)\n")
    print(f"**Audio:** {audio_duration:.1f}s | **Device:** {device} | **Runs:** {args.runs}\n")
    print(f"| Model | Avg (ms) | Min (ms) | Max (ms) | RTF |")
    print(f"|-------|----------|----------|----------|-----|")
    print(f"| Whisper Tiny | {w_avg:.0f} | {w_min:.0f} | {w_max:.0f} | {w_avg/1000/audio_duration:.3f} |")
    print(f"| SenseVoice Small | {s_avg:.0f} | {s_min:.0f} | {s_max:.0f} | {s_avg/1000/audio_duration:.3f} |")
    print(f"\n> **{speedup:.1f}x faster** — saves ~{savings:.0f}ms per utterance")
    print()


if __name__ == "__main__":
    main()
