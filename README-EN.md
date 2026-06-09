# English | [中文版](./README.md)

<p align="center">
  <img src="./assets/zl-talking-logo.jpg" align="middle" width="300"/>
</p>
<p align="center">
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
  <a href=""><img src="https://img.shields.io/badge/python-3.10+-aff.svg"></a>
  <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
</p>
<p align="center">
<a href="https://trendshift.io/repositories/12565" target="_blank"><img src="https://trendshift.io/api/badge/repositories/12565" alt="lipku%2FLiveTalking | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

A real-time interactive streaming digital human system enabling synchronized audio-video conversation, meeting commercial application standards.

<div align="center">
  <a href="https://youtu.be/-ss0H8qLr7E">wav2lip Demo</a> |
  <a href="https://www.bilibili.com/video/BV1G1421z73r/">ernerf Demo</a> |
  <a href="https://youtu.be/vzUMruoZlxc/">musetalk Demo</a>
</div>

**Statement**: This project is based on LiveTalking and modified by Sichuan Yuyang Technology Co., Ltd. All rights reserved.

---

## Features

- **Multi-Model Support**: wav2lip, musetalk, ultralight, ernerf
- **Low VRAM Inference**: ONNX model export & inference, ~50% VRAM reduction
- **Voice Cloning**: Multiple TTS engines (Edge, Doubao, GPT-SoVITS, CosyVoice, Fish Speech, etc.)
- **Real-time Interruption**: Interrupt the digital human while speaking
- **Multiple Outputs**: WebRTC, RTMP, Virtual Camera
- **Motion Choreography**: Play custom videos when idle
- **Session Management**: Multi-session concurrent, independent session lifecycle

---

## Installation

### Requirements

| Dependency  | Version      |
|------------|-------------|
| Python     | 3.10+       |
| CUDA       | 12.x (recommended) |
| GPU VRAM   | 6GB+ (wav2lip) / 8GB+ (musetalk) / 4GB (ONNX) |

### Setup

```bash
git clone https://github.com/zl-talking/zl-talking
cd zl-talking

conda create -n zl-talking python=3.10
conda activate zl-talking

# Install PyTorch (choose according to CUDA version)
conda install pytorch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 pytorch-cuda=12.9 -c pytorch -c nvidia

pip install -r requirements.txt
```

> If Hugging Face is inaccessible, run: `export HF_ENDPOINT=https://hf-mirror.com`

---

## Quick Start

### 1. Download Models

| Model | Download | Location |
|-------|----------|----------|
| wav2lip | [Quark Drive](https://pan.quark.cn/s/83a750323ef0) / [Google Drive](https://drive.google.com/drive/folders/1FOC_MD6wdogyyX_7V1d4NDIO7P9NlSAJ?usp=sharing) | `models/wav2lip.pth` |
| musetalk | Same as above | `models/musetalk/` |
| Avatar Assets | Same as above | `data/avatars/<avatar_id>/` |

### 2. Start Server

```bash
# wav2lip + WebRTC
python app.py --transport webrtc --model wav2lip --avatar_id wav2lip256_avatar1

# musetalk + ONNX (low VRAM)
python app.py --transport webrtc --model musetalk --avatar_id avatar1 --use_onnx

# musetalk + PyTorch (high precision)
python app.py --transport webrtc --model musetalk --avatar_id avatar1

# Custom TTS (default edgetts)
python app.py --transport webrtc --model wav2lip --tts doubao --TTS_SERVER ws://127.0.0.1:3000
```

> **Port Requirements**: TCP 8010; UDP 1-65536

### 3. Access Client

Open `http://serverip:8010/webrtcapi.html` in browser, click `start`, then enter text to interact.

---

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `wav2lip` | Model: musetalk / wav2lip / ultralight |
| `--avatar_id` | `wav2lip256_avatar1` | Avatar ID, maps to `data/avatars/` |
| `--use_onnx` | `False` | Enable ONNX inference (lower VRAM) |
| `--onnx_model_path` | auto | ONNX model path |
| `--batch_size` | `16` | Inference batch size |
| `--fps` | `25` | Video frame rate |
| `--tts` | `edgetts` | TTS engine: edgetts / doubao / cosyvoice / fishtts / tencent / indextts2 / azuretts / qwentts |
| `--REF_FILE` | `zh-CN-YunxiaNeural` | Reference file or voice model ID |
| `--TTS_SERVER` | `http://127.0.0.1:9880` | TTS service URL |
| `--transport` | `webrtc` | Output: webrtc / rtmp / virtualcam |
| `--listenport` | `8010` | Web listen port |
| `--max_session` | `1` | Max concurrent sessions |

---

## Architecture

```
User / Frontend Web ──▶ API Routes ──▶ Session Manager ──▶ Avatar Session
                          │                                        │
                          │  ┌── "chat" ──▶ LLM Engine (Qwen)  ──▶│
                          │  │                                     │
                          │  └── Text ──▶ TTS Engine ──▶ Audio ──▶│
                          │                         Features      │
                          │                                       │
                     Audio Upload ──▶ Audio Feature Extraction    │
                                                                  │
                          ◀── Output ──▶ Inference ──▶ Paste ─────┘
                              WebRTC /       Wav2Lip /    Back to
                              RTMP /         MuseTalk /   Video
                              Virtualcam     Ultralight
```

### Plugin System

Uses a decorator-based registry (`@register`) for three plugin types:

| Type | Registration | Examples |
|------|-------------|----------|
| TTS | `@register("tts", "edgetts")` | edge / doubao / cosyvoice / fish |
| Avatar | `@register("avatar", "musetalk")` | musetalk / wav2lip / ultralight |
| Output | `@register("output", "webrtc")` | webrtc / rtmp / virtualcam |

---

## Performance

| Model | GPU | Inference FPS |
|-------|-----|---------------|
| wav2lip256 | RTX 3060 | 60 |
| wav2lip256 | RTX 3080Ti | 120 |
| musetalk | RTX 3080Ti | 42 |
| musetalk | RTX 3090 | 45 |
| musetalk | RTX 4090 | 72 |
| musetalk (ONNX) | RTX 4090 | 60+ |

> `inferfps` = GPU inference FPS, `finalfps` = stream output FPS. Both must be ≥ 25 for real-time.

---

## Docker

```bash
docker run --gpus all -it --network=host --rm \
  registry.cn-beijing.aliyuncs.com/codewithgpu2/zl-talking:latest
```

Code is in `/root/zl-talking`. Run `git pull` first, then follow Quick Start.

**Cloud Images**:
- [UCloud Image](https://www.compshare.cn/images/4458094e-a43d-45fe-9b57-de79253befe4?referral_code=3XW3852OBmnD089hMMrtuU&ytag=GPU_GitHub_zl-talking)
- [AutoDL Image](https://www.codewithgpu.com/i/zl-talking/zl-talking/base)

---

## Commercial Version

Extended features:

1. HD wav2lip model
2. Full voice interaction + wake word interruption
3. Real-time subtitles + speech events
4. Real-time audio stream input API
5. Transparent background + dynamic background
6. Real-time avatar switching
7. Multi-digital-human in same scene
8. Camera-driven expressions
9. LiveKit integration

Details: <https://zl-talking-doc.readthedocs.io/en/latest/service.html>

---

## Statement

Videos published on Bilibili, WeChat Channels, Douyin etc. based on this project must include zl-talking watermark and logo.

**Copyright (C) 2025 Sichuan Yuyang Technology Co., Ltd.**

---

If this project helps you, please give it a star ⭐

- Knowledge Planet: <https://t.zsxq.com/7NMyO>
- WeChat: wxwubug
- Telegram: <https://t.me/zl-talking>
- Discord: <https://discord.gg/n5jSPCT3Uf>
- Email: contact@yuyangtech.com

<img src="./assets/qrcode-wechat.jpg" align="middle" width="200"/>
