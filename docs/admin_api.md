# Admin 接口 API 文档

基础路径：`http://<host>:<listenport>`

---

## 1. 获取全局配置

```
GET /api/admin/config
```

返回服务启动时的全局配置参数（来自 CLI 参数）。

**请求参数**: 无

**响应**:

```json
{
  "code": 0,
  "msg": "ok",
  "data": {
    "config": {
      "fps": 25,
      "l": 10,
      "m": 8,
      "r": 10,
      "model": "wav2lip",
      "avatar_id": "wav2lip256_avatar1",
      "data_path": "data/avatars",
      "batch_size": 16,
      "modelres": 192,
      "modelfile": "",
      "customvideo_config": "",
      "tts": "edgetts",
      "REF_FILE": "zh-CN-YunxiaNeural",
      "REF_TEXT": null,
      "TTS_SERVER": "http://127.0.0.1:9880",
      "transport": "webrtc",
      "push_url": "http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream",
      "max_session": 1,
      "listenport": 8010,
      "customopt": []
    }
  }
}
```

### 配置字段说明

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `fps` | int | 25 | 视频帧率 |
| `l` | int | 10 | 音频缓冲参数 |
| `m` | int | 8 | 音频缓冲参数 |
| `r` | int | 10 | 音频缓冲参数 |
| `model` | string | "wav2lip" | Avatar 模型：musetalk / wav2lip / ultralight |
| `avatar_id` | string | "wav2lip256_avatar1" | 默认 Avatar 标识 |
| `data_path` | string | "data/avatars" | Avatar 数据目录 |
| `batch_size` | int | 16 | 推理批大小 |
| `modelres` | int | 192 | 模型分辨率 |
| `modelfile` | string | "" | 自定义模型文件路径 |
| `customvideo_config` | string | "" | 自定义动作 JSON 文件路径 |
| `tts` | string | "edgetts" | TTS 插件 |
| `REF_FILE` | string | "zh-CN-YunxiaNeural" | TTS 参考文件或语音模型ID |
| `REF_TEXT` | string | null | TTS 参考文本 |
| `TTS_SERVER` | string | "http://127.0.0.1:9880" | TTS 服务地址 |
| `transport` | string | "webrtc" | 输出传输方式：rtcpush / webrtc / rtmp / virtualcam |
| `push_url` | string | — | RTCPush 目标地址 |
| `max_session` | int | 1 | 最大会话数 |
| `listenport` | int | 8010 | HTTP 监听端口 |
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

