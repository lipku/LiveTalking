###############################################################################
#  SQLite 存储层 —— 形象管理（avatars）
#
#  使用 Python 标准库 sqlite3，零外部依赖。
#  管理类操作并发极低，采用「连接-执行-提交-关闭」的短连接，
#  避免 sqlite3 连接跨线程使用的限制。
###############################################################################

import os
import sqlite3
from datetime import datetime

from utils.logger import logger

# 由 init_db() 设置的全局数据库路径
_db_path = None


def _now():
    return datetime.now().isoformat(timespec='seconds')


def init_db(db_path):
    """初始化数据库与表结构，并记录全局 db_path。"""
    global _db_path
    _db_path = db_path
    parent = os.path.dirname(os.path.abspath(db_path))
    os.makedirs(parent, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS avatars (
                avatar_id   TEXT PRIMARY KEY,
                name        TEXT DEFAULT '',
                description TEXT DEFAULT '',
                model_type  TEXT DEFAULT '',
                status      TEXT DEFAULT '',
                thumbnail   TEXT DEFAULT '',
                created_at  TEXT,
                updated_at  TEXT
            );
            """
        )
        conn.commit()
    finally:
        conn.close()
    logger.info(f"admin db initialized: {db_path}")


def _connect(db_path=None):
    path = db_path or _db_path
    if not path:
        raise RuntimeError("admin db not initialized, call init_db() first")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


# ─── 形象管理 ──────────────────────────────────────────────────────────────

def list_avatars(db_path=None):
    """返回全部形象记录（按创建时间倒序）。"""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT avatar_id, name, description, model_type, status, thumbnail, "
            "created_at, updated_at FROM avatars ORDER BY created_at DESC, avatar_id ASC"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def upsert_avatar(avatar_id, name=None, description=None, model_type=None,
                  status=None, db_path=None):
    """新增或更新形象元数据。未传入的字段保留原值。"""
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM avatars WHERE avatar_id=?", (avatar_id,)
        ).fetchone()
        now = _now()
        if row is None:
            conn.execute(
                "INSERT INTO avatars "
                "(avatar_id, name, description, model_type, status, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (avatar_id, name or '', description or '', model_type or '',
                 status or '', now, now),
            )
        else:
            conn.execute(
                "UPDATE avatars SET name=?, description=?, model_type=?, status=?, updated_at=? "
                "WHERE avatar_id=?",
                (
                    name if name is not None else row['name'],
                    description if description is not None else row['description'],
                    model_type if model_type is not None else row['model_type'],
                    status if status is not None else row['status'],
                    now,
                    avatar_id,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def delete_avatar(avatar_id, db_path=None):
    """删除形象元数据，返回是否有记录被删除。"""
    conn = _connect(db_path)
    try:
        cur = conn.execute("DELETE FROM avatars WHERE avatar_id=?", (avatar_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()
