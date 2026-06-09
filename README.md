# [English](./README-EN.md) | 中文版

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

实时交互流式数字人，实现音视频同步对话，基本可以达到商用效果。

**声明**：本项目基于 LiveTalking 改造，版权归四川昱扬科技有限公司所有。

---

## 特性

- **多模型支持**：wav2lip、musetalk、ultralight、ernerf
- **低显存推理**：支持 ONNX 模型导出与推理，显存占用降低约 50%
- **声音克隆**：支持多种 TTS 引擎（Edge、豆包、GPT-SoVITS、CosyVoice、Fish Speech 等）
- **实时打断**：支持数字人说话时打断并提问
- **多种输出**：WebRTC、RTMP、虚拟摄像头
- **动作编排**：不说话时播放自定义视频
- **会话管理**：多会话并发，独立的 session 生命周期

---

## 安装

### 环境要求

| 依赖         | 版本要求     |
|-------------|-------------|
| Python      | 3.10+       |
| CUDA        | 12.x (推荐) |
| GPU 显存     | 6GB+ (wav2lip) / 8GB+ (musetalk) / 4GB (ONNX) |

### 安装步骤

```bash
git clone https://github.com/zl-talking/zl-talking
cd zl-talking

conda create -n zl-talking python=3.10
conda activate zl-talking

# 安装 PyTorch（根据 CUDA 版本选择）
conda install pytorch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 pytorch-cuda=12.9 -c pytorch -c nvidia

pip install -r requirements.txt
```

> 如果无法访问 Hugging Face，运行前设置：`export HF_ENDPOINT=https://hf-mirror.com`

---

## 快速开始

### 1. 下载模型

| 模型 | 下载地址 | 放置位置 |
|-----|---------|---------|
| wav2lip | [夸克网盘](https://pan.quark.cn/s/83a750323ef0) / [Google Drive](https://drive.google.com/drive/folders/1FOC_MD6wdogyyX_7V1d4NDIO7P9NlSAJ?usp=sharing) | `models/wav2lip.pth` |
| musetalk | 同上 | `models/musetalk/` |
| 形象素材 | 同上 | `data/avatars/<avatar_id>/` |

### 2. 启动服务

```bash
# wav2lip + WebRTC
python app.py --transport webrtc --model wav2lip --avatar_id wav2lip256_avatar1

# musetalk + ONNX 推理（低显存）
python app.py --transport webrtc --model musetalk --avatar_id avatar1 --use_onnx

# musetalk + PyTorch 推理（高精度）
python app.py --transport webrtc --model musetalk --avatar_id avatar1

# 指定 TTS（默认 edgetts）
python app.py --transport webrtc --model wav2lip --tts doubao --TTS_SERVER ws://127.0.0.1:3000
```

> **端口要求**：TCP 8010；UDP 1-65536

### 3. 访问客户端

浏览器打开 `http://serverip:8010/webrtcapi.html`，点击 `start` 播放数字人视频，在文本框输入文字即可交互。

---

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `wav2lip` | 数字人模型：musetalk / wav2lip / ultralight |
| `--avatar_id` | `wav2lip256_avatar1` | 形象 ID，对应 `data/avatars/` 目录 |
| `--use_onnx` | `False` | 使用 ONNX 推理（降低显存） |
| `--onnx_model_path` | 自动选择 | ONNX 模型路径 |
| `--batch_size` | `16` | 推理批次大小 |
| `--fps` | `25` | 视频帧率 |
| `--tts` | `edgetts` | TTS 引擎：edgetts / doubao / cosyvoice / fishtts / tencent / indextts2 / azuretts / qwentts |
| `--REF_FILE` | `zh-CN-YunxiaNeural` | 参考文件名或语音模型 ID |
| `--TTS_SERVER` | `http://127.0.0.1:9880` | TTS 服务地址 |
| `--transport` | `webrtc` | 输出方式：webrtc / rtmp / virtualcam |
| `--listenport` | `8010` | Web 监听端口 |
| `--max_session` | `1` | 最大并发会话数 |

---

## 架构

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

### 插件系统

采用装饰器注册机制（`@register`），支持三种插件类型：

| 插件类型 | 注册名称 | 示例 |
|---------|---------|------|
| TTS | `@register("tts", "edgetts")` | edge / doubao / cosyvoice / fish 等 |
| Avatar | `@register("avatar", "musetalk")` | musetalk / wav2lip / ultralight |
| Output | `@register("output", "webrtc")` | webrtc / rtmp / virtualcam |

---

## 性能

| 模型 | GPU | 推理帧率 (fps) |
|------|-----|---------------|
| wav2lip256 | RTX 3060 | 60 |
| wav2lip256 | RTX 3080Ti | 120 |
| musetalk | RTX 3080Ti | 42 |
| musetalk | RTX 3090 | 45 |
| musetalk | RTX 4090 | 72 |
| musetalk (ONNX) | RTX 4090 | 60+ |

> `inferfps` 为 GPU 推理帧率，`finalfps` 为最终推流帧率。两者需 ≥ 25 才能实时。

---

## Docker 部署

```bash
docker run --gpus all -it --network=host --rm \
  registry.cn-beijing.aliyuncs.com/codewithgpu2/zl-talking:latest
```

代码位于 `/root/zl-talking`，先 `git pull` 拉取最新代码，然后按快速开始章节运行。


## 声明

基于本项目开发并发布在 B站、视频号、抖音等网站上的视频需带上 zl-talking 水印和标识。

**Copyright (C) 2025 四川昱扬科技有限公司**

---

