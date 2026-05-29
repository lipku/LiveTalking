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
| `model` | 是 | string | — | 模型类型：`wav2lip` / `musetalk` |
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

## 模型与参数对应关系

| model | 专有参数 | 生成模块 |
|-------|----------|----------|
| `wav2lip` | `face_det_batch_size`, `pads`, `nosmooth`, `img_size` | `avatars/wav2lip/genavatar.py` |
| `musetalk` | `bbox_shift`, `extra_margin`, `parsing_mode`, `version` | `avatars/musetalk/genavatar.py` |

## 生成输出

Avatar 数据保存在 `<data_path>/<avatar_id>/` 目录下，包含：`full_imgs/`、`face_imgs/`、`coords.pkl` 及模型特定文件。生成完成后可直接通过 `--avatar_id <avatar_id>` 启动服务。
