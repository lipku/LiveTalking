###############################################################################
#  全局会话管理器 (Session Manager)
###############################################################################

import asyncio
import random
from typing import Dict, Optional
from utils.logger import logger
from avatars.base_avatar import BaseAvatar

def _rand_session_id(n: int = 6) -> int:
    """生成 N 位随机 session ID"""
    return random.randint(10 ** (n - 1), 10 ** n - 1)

class SessionManager:
    """
    全局数字人会话管理器。
    
    统一管理 avatar_sessions 生命周期，并在脱离 WebRTC 时依然保持服务可用。
    """
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):
            self.sessions: Dict[int, BaseAvatar] = {}
            self.build_session_fn = None
            self.initialized = True

    def init_builder(self, build_session_fn):
        """配置用于构建 avatar_session 的工厂函数"""
        self.build_session_fn = build_session_fn
        
    def get_session(self, sessionid: int) -> Optional[BaseAvatar]:
        """获取已存活的会话"""
        return self.sessions.get(sessionid)

    def has_session(self, sessionid: int) -> bool:
        """检查会话是否存在"""
        return sessionid in self.sessions and self.sessions[sessionid] is not None
        
    async def create_session(self, params: dict, sessionid: int = None) -> int:
        """
        在异步环境中创建一个新会话
        如果 sessionid 为 None，则自动生成。
        """
        if self.build_session_fn is None:
            raise Exception("SessionManager builder not initialized")
            
        if sessionid is None:
            sessionid = _rand_session_id()
            
        logger.info('Creating sessionid=%d, current session num=%d', sessionid, len(self.sessions))
        # 预先占位防止重复
        self.sessions[sessionid] = None

        # 在线程池中构建 session（加载模型非常耗时）
        avatar_session = await asyncio.get_event_loop().run_in_executor(
            None, self.build_session_fn, sessionid, params
        )
        self.sessions[sessionid] = avatar_session
        return sessionid
        
    def add_session(self, sessionid: int, avatar_session: BaseAvatar):
        """同步添加静态或外部管理的会话（供非服务端入口调用）"""
        self.sessions[sessionid] = avatar_session
        
    def remove_session(self, sessionid: int):
        """销毁会话资源"""
        if sessionid in self.sessions:
            logger.info(f"Removing session {sessionid}")
            # todo: 还可以主动调 avatar_session 释放
            self.sessions.pop(sessionid, None)

# 单例抛出
session_manager = SessionManager()
