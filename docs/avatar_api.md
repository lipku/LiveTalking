# Avatar 生成接口 API 文档

基础路径：`http://<host>:<listenport>`

---

## 1. 创建 Avatar 生成任务

```
POST /api/avatar/task
```

**Content-Type**: `application/json` 或 `multipart/form-data`（上传文件时）

| 参数 | 必填 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `model` | 是 | string | — | 模型类型：`wav2lip` / `musetalk`（可生成） |
| `avatar_id` | 是 | string | — | Avatar 唯一标识符 |
| `video_file` | 否 | file | — | 上传的视频文件（multipart），保存到 `./data/tmp/` |
| `video_path` | 条件 | string | — | 视频文件本地路径（未上传 `video_file` 时必填） |
| `img_size` | 否 | int | 256 | 输出图像尺寸 |
| `nosmooth` | 否 | bool | false | 禁用人脸检测平滑 |
| `bbox_shift` | 否 | int | 0 | 人脸框偏移（musetalk） |
| `extra_margin` | 否 | int | 10 | 人脸裁剪额外边距（musetalk） |
| `pads` | 否 | string | "0 10 0 0" | 填充：上 下 左 右（空格分隔） |
| `parsing_mode` | 否 | string | "jaw" | 人脸解析模式（musetalk） |
| `version` | 否 | string | "v15" | MuseTalk 版本：`v1` / `v15` |
| `face_det_batch_size` | 否 | int | 16 | 人脸检测批大小（wav2lip） |
| `task_id` | 否 | string | 自动UUID | 自定义任务ID |
| `notifyurl` | 否 | string | — | 回调URL，任务状态变更时POST通知 |

**响应**:

```json
{
  "code": 0,
  "msg": "ok",
  "data": {
    "task_id": "uuid-string"
  }
}
```

---

## 2. 查询任务状态

```
GET /api/avatar/task/{task_id}
```

**路径参数**: `task_id` — 创建任务时返回的ID

**响应**:

```json
{
  "code": 0,
  "msg": "ok",
  "data": {
    "task_id": "uuid-string",
    "model_type": "musetalk",
    "avatar_id": "avatar1",
    "status": "running",
    "progress": 45,
    "error_msg": "",
    "notify_url": "",
    "start_time": 1713916800.0,
    "end_time": null,
    "duration": 30.5
  }
}
```

`status` 取值：`pending` → `running` → `completed` / `failed`

`progress`：0-100 整数

---

## 3. 列出所有任务

```
GET /api/avatar/tasks
```

**响应**:

```json
{
  "code": 0,
  "msg": "ok",
  "data": {
    "tasks": [
      { "task_id": "...", "status": "completed", "progress": 100 },
      { "task_id": "...", "status": "running", "progress": 30 }
    ]
  }
}
```

按 `start_time` 降序排列。

---

## 4. 删除任务

```
DELETE /api/avatar/task/{task_id}
```

**路径参数**: `task_id`

**响应**:

```json
// 成功
{ "code": 0, "msg": "ok", "data": { "msg": "Task deleted" } }

// 失败 - 任务不存在
{ "code": -1, "msg": "Task not found" }

// 失败 - 任务不可删除
{ "code": -1, "msg": "Task is in running state, cannot delete" }
```

仅 `pending` 状态的任务可删除。

---

# 形象管理接口

形象元数据保存在 SQLite 数据库 `data/admin.db` 的 `avatars` 表；形象的文件数据（`full_imgs/` 等）仍由上方生成接口产出。

典型流程：

1. `POST /api/avatar/task` 触发生成任务；
2. `POST /api/admin/avatars` 写入元数据（`status: "pending"`）；
3. 前端轮询 `GET /api/avatar/task/{task_id}` 显示进度，结束后用 `PUT /api/admin/avatars/{avatar_id}` 把状态更新为 `completed` / `failed`。

> `GET /api/admin/avatars` 会扫描 `data/avatars/` 目录：已存在但没有元数据记录的形象会自动补登记（`status: "completed"`；若对应生成任务仍在进行则按任务状态登记）。

## 5. 列出形象

```
GET /api/admin/avatars
```

**响应**:

```json
{
  "code": 0,
  "msg": "ok",
  "data": {
    "avatars": [
      {
        "avatar_id": "my_avatar_001",
        "name": "我的数字人",
        "description": "",
        "model_type": "wav2lip",
        "status": "completed",
        "thumbnail": "",
        "created_at": "2026-10-06T12:00:00",
        "updated_at": "2026-10-06T12:05:00"
      }
    ]
  }
}
```

按 `created_at` 降序。`status` 取值：`pending` / `running` / `completed` / `failed` / `''`。

---

## 6. 新增形象

```
POST /api/admin/avatars
```

**Content-Type**: `application/json`

| 参数 | 必填 | 类型 | 说明 |
|------|------|------|------|
| `avatar_id` | 是 | string | Avatar 唯一标识符（主键） |
| `name` | 否 | string | 显示名称 |
| `description` | 否 | string | 描述 |
| `model_type` | 否 | string | 模型类型：`wav2lip` / `musetalk` |
| `status` | 否 | string | 生成状态，新建时通常传 `pending` |

**响应**:

```json
// 成功
{ "code": 0, "msg": "ok", "data": { "avatar_id": "my_avatar_001" } }

// 失败 - avatar_id 为空
{ "code": -1, "msg": "avatar_id is required" }
```

`avatar_id` 已存在时执行更新（upsert），未传入的字段保留原值。

---

## 7. 更新形象

```
PUT /api/admin/avatars/{avatar_id}
```

**路径参数**: `avatar_id`

**Content-Type**: `application/json`

请求体字段同「新增形象」（全部可选，未传的字段保留原值）。生成任务结束后，前端用它把 `status` 更新为 `completed` / `failed`。

**响应**:

```json
{ "code": 0, "msg": "ok", "data": { "avatar_id": "my_avatar_001" } }
```

---

## 8. 获取形象缩略图

```
GET /api/admin/avatars/{avatar_id}/thumbnail
```

**路径参数**: `avatar_id`

**响应**:

- 成功：返回图片二进制（优先 `data/avatars/<avatar_id>/thumbnail.jpg`，不存在时回退 `full_imgs/00000000.png`）
- 失败：`{ "code": -1, "msg": "Thumbnail not found" }`（HTTP 404）
- 失败：`{ "code": -1, "msg": "Invalid avatar_id" }`（路径穿越防护）

---

## 9. 删除形象

```
DELETE /api/admin/avatars/{avatar_id}
```

**路径参数**: `avatar_id`

**响应**:

```json
// 成功
{ "code": 0, "msg": "ok", "data": { "msg": "Avatar my_avatar_001 deleted" } }

// 失败 - 无效的 avatar_id
{ "code": -1, "msg": "Invalid avatar_id" }
```

同时删除数据库记录与 `data/avatars/<avatar_id>/` 目录（目录不存在时忽略），操作不可逆。

---

## 使用方式

形象管理页（`web/avatars.html` / `web/avatars-en.html`）的「使用」按钮会打开：

```
/index.html?avatar=<avatar_id>
```

主页面读取 `avatar` 参数后预填「Avatar ID」，连接时随 `/offer` 的 body 一起发送；服务端据此加载 `data/avatars/<avatar_id>`（模型仍使用启动参数 `--model`）。

---

## 模型与参数对应关系

| model | 专有参数 | 生成模块 |
|-------|----------|----------|
| `wav2lip` | `face_det_batch_size`, `pads`, `nosmooth`, `img_size` | `avatars/wav2lip/genavatar.py` |
| `musetalk` | `bbox_shift`, `extra_margin`, `parsing_mode`, `version` | `avatars/musetalk/genavatar.py` |

## 生成输出

Avatar 数据保存在 `data/avatars/<avatar_id>/` 目录下，包含：`full_imgs/`、`face_imgs/`、`coords.pkl` 及模型特定文件；生成完成后会自动用首帧写入 `thumbnail.jpg`（供形象管理页展示，失败只记录告警，不影响任务状态）。生成完成后可直接通过 `--avatar_id <avatar_id>` 启动服务。
