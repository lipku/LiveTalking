# LiveTalking API 接口文档

基础路径：`http://<host>:<listenport>`

除 WebRTC 信令（`/offer`、`/whep`）和文件下载（`/record/{sessionid}`）外，接口统一返回格式：

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
| `avatar` | 否 | string | 启动参数值 | 指定数字人 ID（`data/avatars` 下的目录名） |
| `refaudio` | 否 | string | — | 参考音频 |
| `reftext` | 否 | string | — | 参考文本 |
| `custom_config` | 否 | string | — | 动作编排配置 JSON 字符串 |

> 模型（`--model`）在服务启动时固定，不能按会话切换。

**响应** (200):

```json
{
  "sdp": "v=0\r\n...",
  "type": "answer",
  "sessionid": "session-uuid"
}
```

> 客户端如需接收字幕等服务器推送，可在创建 offer 前自建 DataChannel，详见 [11. DataChannel 事件推送（字幕）](#11-datachannel-事件推送字幕)。

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
| `avatar` | 否 | string | 启动参数值 | 指定数字人 ID（`data/avatars` 下的目录名） |
| `refaudio` | 否 | string | — | 参考音频 |
| `reftext` | 否 | string | — | 参考文本 |
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

> SSE 与 DataChannel 推送的是同一份消息队列，二选一即可。

---

## 11. DataChannel 事件推送（字幕）

在 WebRTC 连接中通过 DataChannel 接收**服务器→客户端**的实时事件（字幕、播报状态），作为 SSE 的替代方案。

**建立方式**: 客户端在创建 offer 之前自建 data channel，label 固定为 `chat`：

```javascript
pc = new RTCPeerConnection(config);

dc = pc.createDataChannel('chat');
dc.addEventListener('message', (evt) => {
    const data = JSON.parse(evt.data);   // 字幕/状态事件
});
dc.addEventListener('open',  () => console.log('datachannel open'));
dc.addEventListener('close', () => console.log('datachannel close'));

negotiate();   // 之后照常 POST /offer
```

服务端在收到 `datachannel` 后，会为该通道创建一个消息队列，把 `avatar_session` 的所有事件以 JSON 字符串推送过来。

**服务器→客户端消息**:

```json
{ "status": "start", "text": "你好，欢迎使用LiveTalking" }
{ "status": "end",   "text": "你好，欢迎使用LiveTalking" }
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | string | `start`: 该句开始播报；`end`: 该句播报结束 |
| `text` | string | 播报文本（可据此显示字幕） |

> 每个句子会推送 `start` / `end` 两条消息，展示字幕时只取 `start` 可以避免重复。
> 消息中也可能携带 TTS 透传的 `tts` 字段；若后续接入语音识别，用户侧文本会带 `from: "user"`。

**客户端→服务器消息**:

| 消息 | 说明 |
|------|------|
| `ping` | 心跳，服务端回复 `pong`（原样带回 `ping` 后的内容） |

**生命周期**: data channel 关闭时，服务端会取消推送协程并从消息队列列表中移除，不会泄漏。多个通道互不影响。

`web/index.html` 与 `web/index-en.html` 已内置该逻辑，连接成功后即可在页面「字幕 (DataChannel)」区域滚动显示字幕。