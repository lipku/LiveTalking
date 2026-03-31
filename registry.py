###############################################################################
#  插件注册表 — 通过装饰器注册，按名称创建实例
###############################################################################

from typing import Dict, Type, Any
from utils.logger import logger

_REGISTRY: Dict[str, Dict[str, Type]] = {
    "stt": {},
    "llm": {},
    "tts": {},
    "avatar": {},
    "output": {},
}


def register(category: str, name: str):
    """
    装饰器：注册插件类到全局注册表。

    用法::

        @register("tts", "edgetts")
        class EdgeTTS(BaseTTS): ...
    """
    def decorator(cls):
        if category not in _REGISTRY:
            _REGISTRY[category] = {}
        _REGISTRY[category][name] = cls
        logger.info(f"Registered {category}/{name}: {cls.__name__}")
        return cls
    return decorator


def create(category: str, name: str, **kwargs) -> Any:
    """
    按名称创建插件实例。

    Usage::

        tts = registry.create("tts", "edgetts", opt=opt)
    """
    if category not in _REGISTRY or name not in _REGISTRY[category]:
        available = list(_REGISTRY.get(category, {}).keys())
        raise ValueError(
            f"Plugin '{name}' not found in category '{category}'. "
            f"Available: {available}"
        )
    cls = _REGISTRY[category][name]
    return cls(**kwargs)


def list_plugins(category: str = None) -> Dict[str, list]:
    """列出已注册的插件"""
    if category:
        return {category: list(_REGISTRY.get(category, {}).keys())}
    return {cat: list(plugins.keys()) for cat, plugins in _REGISTRY.items()}
