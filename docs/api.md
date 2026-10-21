# LiveTalking API 接口文档

基础路径：`http://<host>:<listenport>`

所有接口统一返回格式：

```json
{ "code": 0, "msg": "ok", "data": {} }
```

`code` 为 0 表示成功，非 0 表示错误。

---

## 1. WebRTC Offer (JSON)

交换 SDP 以建立 WebRTC 连接，扩展参数在 JSON body 中。

```
POST /offer
```

**Content-Type**: `application/json`

| 参数 | 必填 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `sdp` | 是 | string | — | WebRTC Offer SDP |
| `type` | 是 | string | — | 必须为 `offer` |
| `avatar` | 否 | string | 启动参数值 | 指定数字人 ID |
| `refaudio` | 否 | string | — | 参考音频 |
| `reftext` | 否 | string | — | 参考文本 |
| `custom_config` | 否 | string | — | 动作编排配置 JSON 字符串 |

**响应** (200):

```json
{
  "sdp": "v=0\r\n...",
  "type": "answer",
  "sessionid": "session-uuid"
}
```

---

## 2. WebRTC Offer (WHEP)

符合 [WHEP 协议](https://datatracker.ietf.org/doc/draft-ietf-wish-whep/)（WebRTC HTTP Egress Protocol）。
SDP offer 以 `application/sdp` 裸文本发送，扩展参数通过 query string 传递。

```
POST /whep
```

**Content-Type**: `application/sdp`

**Query 参数**:

| 参数 | 必填 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `avatar` | 否 | string | 启动参数值 | 指定数字人 ID |
| `refaudio` | 否 | string | — | 参考音频 |
| `reftext` | 否 | string | — | 参考文本 |
| `tts` | 否 | string | — | TTS 引擎 |
| `tts_server` | 否 | string | — | TTS 服务地址 |
| `tts_speed` | 否 | number | — | TTS 语速 |
| `custom_config` | 否 | string | — | 动作编排配置 JSON 字符串 |

**Body**: SDP offer 裸文本，例如：

```
v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\n...
```

**响应** (201):

- `Content-Type`: `application/sdp`
- `X-Session-ID`: 生成的会话 ID（UUID）

Body 为 SDP answer 裸文本：

```
v=0\r\no=- ...\r\n...
```

**客户端示例**:

```javascript
const params = new URLSearchParams({
  avatar: 'wav2lip256_avatar1',
  refaudio: 'zh-CN-YunxiaNeural',
});

const res = await fetch('/whep?' + params.toString(), {
  method: 'POST',
  headers: { 'Content-Type': 'application/sdp' },
  body: pc.localDescription.sdp,
});

const answerSdp = await res.text();
const sessionid = res.headers.get('X-Session-ID');
await pc.setRemoteDescription({ type: 'answer', sdp: answerSdp });
```

---

## 3. 文本驱动 (Human)

发送文本驱动数字人说话，支持直接复读或 LLM 对话。

```
POST /human
```

**Content-Type**: `application/json`

| 参数 | 必填 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `sessionid` | 是 | string | — | 会话 ID |
| `text` | 是 | string | — | 输入文本 |
| `type` | 是 | string | — | `echo`: 直接复读; `chat`: 触发 LLM 回答 |
| `interrupt` | 否 | bool | false | 是否打断当前播报 |
| `tts` | 否 | object | — | 透传给 TTS 的配置（如 `voice`, `emotion`） |

**响应**:

```json
{ "code": 0, "msg": "ok" }
```

---

## 4. 音频驱动 (Human Audio)

上传音频文件驱动数字人。

```
POST /humanaudio
```

**Content-Type**: `multipart/form-data`

| 参数 | 必填 | 类型 | 说明 |
|------|------|------|------|
| `sessionid` | 是 | string | 会话 ID |
| `file` | 是 | file | 音频文件 |

**响应**:

```json
{ "code": 0, "msg": "ok" }
```

---

## 5. 打断播报

立即清空当前会话的音频队列。

```
POST /interrupt_talk
```

| 参数 | 必填 | 类型 | 说明 |
|------|------|------|------|
| `sessionid` | 是 | string | 会话 ID |

**响应**:

```json
{ "code": 0, "msg": "ok" }
```

---

## 6. 查询说话状态

```
POST /is_speaking
```

| 参数 | 必填 | 类型 | 说明 |
|------|------|------|------|
| `sessionid` | 是 | string | 会话 ID |

**响应**:

```json
{
  "code": 0,
  "msg": "ok",
  "data": true
}
```

---

## 7. 录制控制

控制服务器端的渲染录制。

```
POST /record
```

| 参数 | 必填 | 类型 | 说明 |
|------|------|------|------|
| `sessionid` | 是 | string | 会话 ID |
| `type` | 是 | string | `start_record`: 开始录制; `end_record`: 停止并合成 |

**响应**:

```json
{ "code": 0, "msg": "ok" }
```

---

## 8. 下载录像

下载录制完成的 MP4 文件。

```
GET /record/{sessionid}
```

**路径参数**: `sessionid` — 会话 ID

**响应**: MP4 文件流。若文件不存在返回 404。

---

## 9. 设置动作编排 (Audiotype)

```
POST /set_audiotype
```

| 参数 | 必填 | 类型 | 说明 |
|------|------|------|------|
| `sessionid` | 是 | string | 会话 ID |
| `audiotype` | 是 | int | 预定义的动作/状态索引 |

**响应**:

```json
{ "code": 0, "msg": "ok" }
```

---

## 10. SSE 事件流

```
GET /sse?sessionid=<sessionid>
```

**协议**: Server-Sent Events (SSE)

用于接收**服务器→客户端**的异步状态推送（播报事件、状态变化等）

**请求参数 (Query)**:

| 参数 | 必填 | 类型 | 说明 |
|------|------|------|------|
| `sessionid` | 是 | string | 会话 ID |

**响应格式**: `Content-Type: text/event-stream`

每条事件一行 JSON，格式为：

```
data: {"status": "start"}

```

**客户端示例**:

```javascript
const es = new EventSource(`/sse?sessionid=${sessionid}`);
es.onmessage = (event) => {
    const data = JSON.parse(event.data);
    // data 为服务器推送的播报状态/事件
};
es.onerror = () => {
    // 连接出错或服务端主动断开，EventSource 会自动重连
};

// 断开时
es.close();
```

---

## 11. 本地语音识别（/api/asr）

用户语音识别（STT）的 WebSocket 接口，与外置 FunASR 服务（`wss://www.funasr.com:10096/`）客户端协议兼容，浏览器端语音可直接在本服务内完成识别。此接口为 WebSocket 协议，不适用上文统一的 `{code, msg, data}` JSON 响应格式。

**可用性**（可选依赖）：

- 安装 `pip install funasr modelscope` 后启动服务即自动注册 `/api/asr`；未安装时服务正常启动，端点不注册（仅日志提示）
- 引擎通过启动参数 `--asr` 选择，默认 `funasr`（SenseVoiceSmall + fsmn-vad，首次运行自动从 ModelScope 下载模型）

```
GET /api/asr
```

**协议**: WebSocket

**消息流**:

1. 客户端发送配置 JSON（开始说话）：

```json
{"chunk_size":[5,10,5], "wav_name":"h5", "is_speaking":true, "mode":"2pass", "itn":false}
```

2. 客户端持续发送二进制音频帧：16 kHz、单声道、PCM16 little-endian（每帧 960 字节 ≈ 60 ms）
3. 客户端发送停止信号：

```json
{"is_speaking":false}
```

4. 服务端返回整段识别结果：

```json
{"text":"hello world", "mode":"2pass-offline", "is_final":true, "timestamp":null}
```

**回复字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `text` | string | 识别文本；推理失败或音频过短（< 20ms）时为 `""` |
| `mode` | string | 正常识别时请求 `2pass` 映射为 `2pass-offline`，其他值原样返回；音频过短时原样返回请求的 `mode` |
| `is_final` | bool | 恒为 `true`（整段离线识别） |
| `timestamp` | null | 保留字段，恒为 `null` |

> 注意区分：`/api/asr`（`server/asr_server.py` + `stt/` 插件，registry 类别 `stt`）做的是**用户语音识别（STT）**；`avatars/audio_features/base_asr.py` 是数字人**口型特征提取**（audio features for lip-sync），与语音识别无关。
