# Admin 接口 API 文档

基础路径：`http://<host>:<listenport>`

本文件包含全局配置与活跃会话查询；形象的增删改查见 [avatar_api.md](avatar_api.md) 的「形象管理接口」。

---

## 1. 获取全局配置

```
GET /api/admin/config
```

返回服务启动时的全局配置参数（来自 CLI 参数 / YAML 配置）。

**请求参数**: 无

**响应**:

```json
{
  "code": 0,
  "msg": "ok",
  "data": {
    "config": {
      "config": "config.yaml",
      "fps": 25,
      "l": 10,
      "m": 8,
      "r": 10,
      "model": "wav2lip",
      "avatar_id": "wav2lip256_avatar1",
      "batch_size": 16,
      "modelres": 192,
      "modelfile": "",
      "customvideo_config": "",
      "tts": "edgetts",
      "REF_FILE": "zh-CN-YunxiaNeural",
      "REF_TEXT": null,
      "TTS_SERVER": "http://127.0.0.1:9880",
      "llm_provider": "dashscope",
      "llm_model": "",
      "transport": "webrtc",
      "stun": "stun:stun.freeswitch.org:3478",
      "push_url": "http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream",
      "max_session": 5,
      "listenport": 8010,
      "audio_output_device": null,
      "customopt": []
    }
  }
}
```

### 配置字段说明

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `config` | string | "config.yaml" | YAML 配置文件路径（空字符串表示不加载） |
| `fps` | int | 25 | 视频帧率（必须为 25） |
| `l` | int | 10 | 音频缓冲参数 |
| `m` | int | 8 | 音频缓冲参数 |
| `r` | int | 10 | 音频缓冲参数 |
| `model` | string | "wav2lip" | Avatar 模型：musetalk / wav2lip / ultralight（启动时固定） |
| `avatar_id` | string | "wav2lip256_avatar1" | 默认 Avatar 标识（`data/avatars` 下的目录名） |
| `batch_size` | int | 16 | 推理批大小 |
| `modelres` | int | 192 | 模型分辨率 |
| `modelfile` | string | "" | 自定义模型文件路径 |
| `customvideo_config` | string | "" | 自定义动作 JSON 文件路径 |
| `tts` | string | "edgetts" | TTS 插件 |
| `REF_FILE` | string | "zh-CN-YunxiaNeural" | TTS 参考文件或语音模型 ID |
| `REF_TEXT` | string | null | TTS 参考文本 |
| `TTS_SERVER` | string | "http://127.0.0.1:9880" | TTS 服务地址 |
| `llm_provider` | string | "dashscope" | LLM 提供方：dashscope / orcarouter |
| `llm_model` | string | "" | LLM 模型名，空则使用提供方默认值 |
| `transport` | string | "webrtc" | 输出传输方式：rtcpush / webrtc / rtmp / virtualcam |
| `stun` | string | "stun:stun.freeswitch.org:3478" | STUN 服务器地址，空字符串表示禁用 |
| `push_url` | string | — | RTCPush 目标地址 |
| `max_session` | int | 5 | 最大并发会话数 |
| `listenport` | int | 8010 | HTTP 监听端口 |
| `audio_output_device` | int | null | 音频输出设备索引（仅 virtualcam 使用，null 为系统默认） |
| `customopt` | array | [] | 自定义动作配置（已解析） |

---

## 2. 获取活跃会话列表

```
GET /api/admin/sessions
```

返回当前所有活跃会话及其状态和配置。

**请求参数**: 无

**响应**:

```json
{
  "code": 0,
  "msg": "ok",
  "data": {
    "sessions": [
      {
        "sessionid": "uuid-string",
        "speaking": true,
        "recording": false,
        "model": "musetalk",
        "avatar_id": "avatar1",
        "REF_FILE": "zh-CN-YunxiaNeural",
        "transport": "webrtc",
        "batch_size": 16,
        "customopt": []
      }
    ]
  }
}
```

### 会话字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `sessionid` | string | 会话唯一标识 |
| `speaking` | bool | 当前是否正在说话 |
| `recording` | bool | 当前是否正在录制 |
| `model` | string | 会话使用的 Avatar 模型 |
| `avatar_id` | string | 会话使用的 Avatar 标识 |
| `REF_FILE` | string | TTS 参考文件/模型ID |
| `transport` | string | 传输方式 |
| `batch_size` | int | 推理批大小 |
| `customopt` | array | 自定义动作配置 |

---

