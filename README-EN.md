<p align="center">
    <img src="./assets/LiveTalking-logo.png" align="middle" width="600"/>
</p>

English | [中文版](./README.md)

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href="https://github.com/lipku/LiveTalking/releases"><img src="https://img.shields.io/github/v/release/lipku/LiveTalking?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.10+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href="https://github.com/lipku/LiveTalking/graphs/contributors"><img src="https://img.shields.io/github/contributors/lipku/LiveTalking?color=c4f042&style=flat-square"></a>
</p>
<p align="center">
<a href="https://trendshift.io/repositories/12565" target="_blank"><img src="https://trendshift.io/api/badge/repositories/12565" alt="lipku%2FLiveTalking | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

A real-time interactive streaming digital human engine enabling synchronized audio-video conversation, widely adopted in commercial applications.

**Demos**: [wav2lip](https://youtu.be/-ss0H8qLr7E) | [ernerf](https://www.bilibili.com/video/BV1G1421z73r/) | [musetalk](https://youtu.be/vzUMruoZlxc/)

Domestic Mirror: <https://gitee.com/lipku/LiveTalking>

---

## Features
1. Supports multiple digital human models: ernerf, musetalk, wav2lip, Ultralight-Digital-Human
2. Supports voice cloning
3. Supports interrupting the digital human while speaking
4. Supports full-body video stitching
5. Supports WebRTC, RTMP, and virtual camera output
6. Supports action choreography: plays custom videos when not speaking
7. Supports multi-concurrency
8. Supports custom digital human avatars
9. Provides frontend API integration

---

## Usage Scenarios

LiveTalking leverages real-time streaming digital human technology to drive virtual avatars via text or voice, combined with LLM for intelligent conversation. Suitable for the following scenarios:

| Scenario | Description |
|----------|-------------|
| **Virtual Streamer / Live Commerce** | 24/7 unmanned live streaming with LLM-generated sales scripts and action choreography for natural performance |
| **AI Digital Human Customer Service** | Integrate enterprise knowledge bases for real-time voice Q&A with interruption support |
| **Online Education / Training** | Digital teacher分身 for course recording, or API-driven digital instructor for real-time lectures |
| **Intelligent Voice Assistant** | Pair with smart speakers or apps, calling the `/human` API to drive digital human voice interactions |
| **Large Screen Presentation** | Digital human presenter for exhibition halls, event venues, and other content narration scenarios |
| **Batch Short Video Creation** | Submit scripts in batch via API to generate digital human videos without real-person filming, using `/human` + `/record` APIs |

**Core Flow**: User input (text/audio) → LLM response (optional) → TTS speech synthesis → Real-time lip-sync → Audio/video streaming output

---

## 1. Installation

Tested on Ubuntu 24.04, Python 3.12, PyTorch 2.9.1, CUDA 13.0.

### 1.1 Install Dependencies

```bash
git clone https://github.com/lipku/LiveTalking.git 
conda create -n livetalking python=3.12
conda activate livetalking
# If CUDA version is not 13.0 (check via nvidia-smi), install the corresponding PyTorch version(https://pytorch.org/get-started/previous-versions)
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130
cd LiveTalking
pip install -r requirements.txt
```

Installation FAQ: <https://doc.livetalking.ai/en/docs/faq/>

Linux CUDA environment setup: <https://zhuanlan.zhihu.com/p/674972886>

---

## 2. Quick Start

### 2.1 Download Models

| Source | Link |
|--------|------|
| Quark Cloud | <https://pan.quark.cn/s/83a750323ef0> |
| Google Drive | <https://drive.google.com/drive/folders/1FOC_MD6wdogyyX_7V1d4NDIO7P9NlSAJ?usp=sharing> |

1. Copy `wav2lip256.pth` to the project's `models/` directory and rename it to `wav2lip.pth`
2. Extract `wav2lip256_avatar1.tar.gz` and copy the entire extracted folder to `data/avatars/`

### 2.2 Start the Server

```bash
python app.py --transport webrtc --model wav2lip --avatar_id wav2lip256_avatar1
```

> **Note**: The server must open ports TCP:8010, UDP:1-65536

### 2.3 Client Access

| Method | Description |
|--------|-------------|
| Browser | Open `http://serverip:8010/index.html`, click "Start Connection" to play the digital human video, then enter text and submit |
| API | See [API Docs](docs/api.md) for HTTP-based integration |
| Desktop App | Download: <https://pan.quark.cn/s/d7192d8ac19b> |

### 2.4 Web Pages

| Page | URL | Description |
|------|-----|-------------|
| Home | `/index.html` | WebRTC connection + text/audio driver + recording control |
| Avatar Creator | `/avatar.html` | Upload video to auto-generate digital human avatars |
| Admin Console | `/admin.html` | Real-time session monitoring & global configuration |

<img src="./assets/index.jpg" align="middle"/>

### 2.5 Quick Experience

Create an instance with a cloud image to run instantly:

- [UCloud Image](https://www.compshare.cn/images/4458094e-a43d-45fe-9b57-de79253befe4?referral_code=3XW3852OBmnD089hMMrtuU&ytag=GPU_GitHub_livetalking)

### 2.6 Documentation
<https://doc.livetalking.ai/en>

---

## 3. Architecture

### Dataflow Diagram

<img src="./assets/dataflow.png" align="middle" />

### Layer Overview

**API Layer**
- `/human`: Accepts text, supporting echo (direct playback) and chat (LLM conversation) modes
- `/humanaudio`: Accepts audio files for direct playback
- Each connection is assigned a unique `sessionid`, supporting multi-user concurrency

**Logic Layer**
- **LLM Engine**: Integrates with models like Qwen to generate conversational responses
- **TTS Engine**: Modular design supporting EdgeTTS, GPT-SoVITS, CosyVoice, Tencent Cloud, and more
- **Feature Extraction**: Synchronously extracts acoustic features (e.g., Mel spectrograms) for lip-sync inference

**Rendering Layer**
- **Model Inference**: Uses deep learning models (Wav2Lip, MuseTalk, etc.) to generate lip-sync frames from audio features
- **Post-Processing**: Smoothly overlays the generated mouth region back onto the original high-definition video

**Streaming Layer**
- **WebRTC**: Low-latency browser-based streaming
- **RTMP**: Standard live streaming protocol, supports pushing to platforms like Bilibili/YouTube
- **Virtual Camera**: Outputs as a system camera device

**Plugin System**
- Decentralized registration mechanism based on [registry.py](registry.py), allowing developers to extend TTS, Avatar, and Output modules

---

## 4. API Documentation

| Document | Description |
|----------|-------------|
| [docs/api.md](docs/api.md) | General API — WebRTC, text/audio driver, recording, action choreography |
| [docs/avatar_api.md](docs/avatar_api.md) | Avatar Generation API — create tasks, query progress, delete tasks |
| [docs/admin_api.md](docs/admin_api.md) | Admin API — global config, session monitoring, force stop |

---

## 5. Docker

Available images:
- **AutoDL**: <https://www.codewithgpu.com/i/lipku/livetalking/base> — [Tutorial](https://doc.livetalking.ai/en/docs/autodl/)
- **UCloud**: <https://www.compshare.cn/images/4458094e-a43d-45fe-9b57-de79253befe4?referral_code=3XW3852OBmnD089hMMrtuU&ytag=GPU_GitHub_livetalking> — Supports opening any port, no additional SRS deployment required — [Tutorial](https://doc.livetalking.ai/en/docs/ucloud/)

> AutoDL cannot open UDP ports, so you need to deploy SRS or TURN relay service separately.

---

## 6. Performance

- Each video stream compression consumes CPU; higher resolution means greater CPU usage. Each lip-sync inference consumes GPU
- Concurrent sessions when not speaking depend on CPU; concurrent speaking sessions depend on GPU
- In backend logs: `inferfps` = GPU inference frame rate, `finalfps` = final streaming frame rate. Both must be >= 25 for real-time performance

### Real-Time Inference Performance

| Model | GPU | FPS |
|:------|:----|:----|
| wav2lip256 | RTX 3060 | 60 |
| wav2lip256 | RTX 3080Ti | 120 |
| musetalk | RTX 3080Ti | 42 |
| musetalk | RTX 3090 | 45 |
| musetalk | RTX 4090 | 72 |

- wav2lip256: RTX 3060 or higher recommended
- musetalk: RTX 3080Ti or higher recommended

---

## 7. Statement

Videos developed based on this project and published on platforms such as Bilibili, WeChat Channels, and Douyin must include the LiveTalking watermark and logo.

---

If this project is helpful to you, please give it a Star. Contributors interested in improving this project are also welcome.

| Community | Link |
|-----------|------|
| Knowledge Planet | <https://t.zsxq.com/7NMyO> |
| WeChat | wxwubug (mention for group invite) |
| Telegram | <https://t.me/livetalking> |
| Discord | <https://discord.gg/n5jSPCT3Uf> |
| Email | lipku@foxmail.com |
| WeChat Official | 数字人技术 |

<img src="./assets/qrcode-wechat.jpg" align="middle" />
