###############################################################################
#  管理 API 路由 —— 形象管理（avatars）
#
#  形象管理：SQLite 元数据 CRUD + 缩略图文件服务（复用 /api/avatar/task 生成）。
###############################################################################

import os
import json
import shutil

from aiohttp import web

from server import admin_db
from server.task_manager import task_manager
from utils.logger import logger


def json_ok(data=None):
    body = {"code": 0, "msg": "ok"}
    if data is not None:
        body["data"] = data
    return web.Response(
        content_type="application/json",
        text=json.dumps(body, ensure_ascii=False),
    )


def json_error(msg: str, code: int = -1, status: int = 200):
    return web.Response(
        status=status,
        content_type="application/json",
        text=json.dumps({"code": code, "msg": str(msg)}, ensure_ascii=False),
    )


def _data_path(request):
    opt = request.app.get("opt")
    return getattr(opt, 'data_path', './data') if opt else './data'


def _avatars_root(request):
    return os.path.abspath(os.path.join(_data_path(request), 'avatars'))


# ─── 形象管理 ──────────────────────────────────────────────────────────────

def _sync_avatars_from_disk(request):
    """把 data/avatars 下已存在但数据库没有记录的形象补登记为 completed。"""
    root = _avatars_root(request)
    if not os.path.isdir(root):
        return
    known = {a['avatar_id'] for a in admin_db.list_avatars()}
    # 通过 /avatar.html 等入口发起、仍在生成中的任务，按真实状态登记
    running = {}
    for t in task_manager.list_tasks():
        if t['status'] in ('pending', 'running'):
            running.setdefault(t['avatar_id'], t['status'])

    for name in os.listdir(root):
        if name.startswith('.') or name in known:
            continue
        if not os.path.isdir(os.path.join(root, name)):
            continue
        admin_db.upsert_avatar(name, name=name, status=running.get(name, 'completed'))
        logger.info("Registered existing avatar into admin db: %s", name)


async def list_avatars(request):
    """GET /api/admin/avatars —— 返回全部形象元数据。"""
    try:
        #_sync_avatars_from_disk(request)
        return json_ok(data={"avatars": admin_db.list_avatars()})
    except Exception as e:
        logger.exception("list_avatars error:")
        return json_error(str(e))


async def create_avatar(request):
    """POST /api/admin/avatars —— 新增形象元数据 {avatar_id,name,description,model_type,status?}"""
    try:
        params = await request.json()
        avatar_id = (params.get('avatar_id') or '').strip()
        if not avatar_id:
            return json_error("avatar_id is required", status=400)

        admin_db.upsert_avatar(
            avatar_id,
            name=params.get('name'),
            description=params.get('description'),
            model_type=params.get('model_type'),
            status=params.get('status'),
        )
        return json_ok(data={"avatar_id": avatar_id})
    except Exception as e:
        logger.exception("create_avatar error:")
        return json_error(str(e))


async def update_avatar(request):
    """PUT /api/admin/avatars/{avatar_id} —— 更新形象元数据。"""
    try:
        avatar_id = request.match_info.get('avatar_id')
        if not avatar_id:
            return json_error("avatar_id is required", status=400)

        params = await request.json()
        admin_db.upsert_avatar(
            avatar_id,
            name=params.get('name'),
            description=params.get('description'),
            model_type=params.get('model_type'),
            status=params.get('status'),
        )
        return json_ok(data={"avatar_id": avatar_id})
    except Exception as e:
        logger.exception("update_avatar error:")
        return json_error(str(e))


async def delete_avatar(request):
    """DELETE /api/admin/avatars/{avatar_id} —— 删除元数据与 data/avatars/<id> 目录。"""
    try:
        avatar_id = request.match_info.get('avatar_id')
        if not avatar_id:
            return json_error("avatar_id is required", status=400)

        avatars_root = _avatars_root(request)
        avatar_dir = os.path.abspath(os.path.join(avatars_root, avatar_id))

        # 安全检查：防止路径穿越
        if avatar_dir != avatars_root and not avatar_dir.startswith(avatars_root + os.sep):
            return json_error("Invalid avatar_id", code=400)

        if os.path.isdir(avatar_dir):
            shutil.rmtree(avatar_dir)
            logger.info(f"Avatar data deleted: {avatar_id}, path={avatar_dir}")

        admin_db.delete_avatar(avatar_id)
        return json_ok(data={"msg": f"Avatar {avatar_id} deleted"})
    except Exception as e:
        logger.exception("delete_avatar error:")
        return json_error(str(e))


async def get_avatar_thumbnail(request):
    """GET /api/admin/avatars/{avatar_id}/thumbnail —— 返回缩略图，回退首帧。"""
    try:
        avatar_id = request.match_info.get('avatar_id')
        avatars_root = _avatars_root(request)
        avatar_dir = os.path.abspath(os.path.join(avatars_root, avatar_id))

        if avatar_dir != avatars_root and not avatar_dir.startswith(avatars_root + os.sep):
            return json_error("Invalid avatar_id", code=400)

        for rel in ('thumbnail.jpg', os.path.join('full_imgs', '00000000.png')):
            path = os.path.join(avatar_dir, rel)
            if os.path.isfile(path):
                return web.FileResponse(path, headers={'Cache-Control': 'no-store'})

        return json_error("Thumbnail not found", code=404, status=404)
    except Exception as e:
        logger.exception("get_avatar_thumbnail error:")
        return json_error(str(e))


def setup_admin_routes(app):
    # 形象管理
    app.router.add_get("/api/admin/avatars", list_avatars)
    app.router.add_post("/api/admin/avatars", create_avatar)
    app.router.add_put("/api/admin/avatars/{avatar_id}", update_avatar)
    app.router.add_delete("/api/admin/avatars/{avatar_id}", delete_avatar)
    app.router.add_get("/api/admin/avatars/{avatar_id}/thumbnail", get_avatar_thumbnail)
