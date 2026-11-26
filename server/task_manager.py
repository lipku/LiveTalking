import uuid
import time
import threading
import json
import os
import requests
import cv2
from concurrent.futures import ThreadPoolExecutor
from utils.logger import logger

class AvatarTask:
    def __init__(self, task_id, model_type, avatar_id, params, notify_url=None):
        self.task_id = task_id
        self.model_type = model_type
        self.avatar_id = avatar_id
        self.params = params
        self.status = "pending"  # pending, running, completed, failed
        self.progress = 0
        self.error_msg = ""
        self.notify_url = notify_url
        self.start_time = time.time()
        self.end_time = None

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "model_type": self.model_type,
            "avatar_id": self.avatar_id,
            "status": self.status,
            "progress": self.progress,
            "error_msg": self.error_msg,
            "notify_url": self.notify_url,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": (self.end_time or time.time()) - self.start_time
        }

class TaskManager:
    def __init__(self, max_workers=1):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.tasks = {}
        self.lock = threading.Lock()

    def add_task(self, model_type, avatar_id, params, task_id=None, notify_url=None):
        if task_id is None:
            task_id = str(uuid.uuid4())

        task = AvatarTask(task_id, model_type, avatar_id, params, notify_url)
        with self.lock:
            self.tasks[task_id] = task

        self._notify(task)
        self.executor.submit(self._run_task, task_id)
        return task_id

    def get_task(self, task_id):
        with self.lock:
            return self.tasks.get(task_id)

    def delete_task(self, task_id):
        with self.lock:
            task = self.tasks.get(task_id)
            if not task:
                return False, "Task not found"

            if task.status != "pending":
                return False, f"Task is in {task.status} state, cannot delete"

            del self.tasks[task_id]
            return True, "Task deleted"

    def list_tasks(self):
        with self.lock:
            return sorted([task.to_dict() for task in self.tasks.values()], key=lambda x: x['start_time'], reverse=True)

    def _create_thumbnail(self, avatar_id, save_path):
        """从生成结果的首帧生成 thumbnail.jpg，供形象管理页展示。"""
        avatar_path = os.path.join(save_path, avatar_id)
        thumbnail_path = os.path.join(avatar_path, "thumbnail.jpg")
        source_candidates = [
            os.path.join(avatar_path, "full_imgs", "00000000.png"),
        ]

        source_path = next((path for path in source_candidates if os.path.exists(path)), None)
        if not source_path:
            raise FileNotFoundError("Thumbnail source image not found in full_imgs")

        image = cv2.imread(source_path)
        if image is None:
            raise ValueError(f"Failed to read thumbnail source image: {source_path}")

        height, width = image.shape[:2]
        max_edge = 256
        scale = min(max_edge / max(height, width), 1.0)
        if scale < 1.0:
            resized = cv2.resize(
                image,
                (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            resized = image

        if not cv2.imwrite(thumbnail_path, resized, [int(cv2.IMWRITE_JPEG_QUALITY), 90]):
            raise ValueError(f"Failed to write thumbnail image: {thumbnail_path}")

    def _run_task(self, task_id):
        task = self.get_task(task_id)
        if not task:
            return

        task.status = "running"
        self._notify(task)
        logger.info(f"Starting task {task_id} for model {task.model_type}, avatar {task.avatar_id}")

        try:
            def progress_callback(p):
                task.progress = p

            if task.model_type == "musetalk":
                from avatars.musetalk.genavatar import generate_avatar
                generate_avatar(
                    video_path=task.params['video_path'],
                    avatar_id=task.avatar_id,
                    save_path=task.params.get('save_path', './data/avatars'),
                    bbox_shift=task.params.get('bbox_shift', 0),
                    extra_margin=task.params.get('extra_margin', 10),
                    parsing_mode=task.params.get('parsing_mode', 'jaw'),
                    version=task.params.get('version', 'v15'),
                    progress_callback=progress_callback
                )
            elif task.model_type == "wav2lip":
                from avatars.wav2lip.genavatar import generate_avatar
                generate_avatar(
                    video_path=task.params['video_path'],
                    avatar_id=task.avatar_id,
                    save_path=task.params.get('save_path', './data/avatars'),
                    img_size=task.params.get('img_size', 96),
                    pads=task.params.get('pads', [0, 10, 0, 0]),
                    nosmooth=task.params.get('nosmooth', False),
                    face_det_batch_size=task.params.get('face_det_batch_size', 16),
                    progress_callback=progress_callback
                )
            else:
                raise ValueError(f"Unsupported model type: {task.model_type}")

            # 缩略图失败不影响形象生成结果
            try:
                self._create_thumbnail(task.avatar_id, task.params.get('save_path', './data/avatars'))
            except Exception as e:
                logger.warning(f"Failed to create thumbnail for {task.avatar_id}: {e}")

            task.status = "completed"
            task.progress = 100
            self._notify(task)
            logger.info(f"Task {task_id} completed successfully")
        except Exception as e:
            task.status = "failed"
            task.error_msg = str(e)
            self._notify(task)
            logger.error(f"Task {task_id} failed: {e}")
        finally:
            task.end_time = time.time()

    def _notify(self, task):
        if not task.notify_url:
            return

        try:
            payload = task.to_dict()
            logger.info(f"Sending notification for task {task.task_id} to {task.notify_url}")
            requests.post(task.notify_url, json=payload, timeout=5)
        except Exception as e:
            logger.error(f"Failed to send notification for task {task.task_id}: {e}")

task_manager = TaskManager()
