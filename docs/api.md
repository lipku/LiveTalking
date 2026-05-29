# LiveTalking API 接口文档

基础路径：`http://<host>:<listenport>`

所有接口统一返回格式：

```json
{ "code": 0, "msg": "ok", "data": {} }
```

`code` 为 0 表示成功，非 0 表示错误。

---

## 1. WebRTC Offer

交换 SDP 以建立 WebRTC 连接。

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

**响应**:

```json
{
  "sdp": "v=0\r\n...",
  "type": "answer",
  "sessionid": "session-uuid"
}
```

---

## 2. 文本驱动 (Human)

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

## 3. 音频驱动 (Human Audio)

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

## 4. 打断播报

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

## 5. 查询说话状态

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

## 6. 录制控制

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

## 7. 下载录像

下载录制完成的 MP4 文件。

```
GET /record/{sessionid}
```

**路径参数**: `sessionid` — 会话 ID

**响应**: MP4 文件流。若文件不存在返回 404。

---

## 8. 设置动作编排 (Audiotype)

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
