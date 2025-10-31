###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku@foxmail.com
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################

"""
暂停控制配置管理模块
提供配置参数的验证、默认值处理和运行时更新支持
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from enum import Enum
from logger import logger


class ConfigError(Exception):
    """配置相关错误的基类"""
    pass


class ValidationError(ConfigError):
    """配置验证错误"""
    pass


class ConfigUpdateError(ConfigError):
    """配置更新错误"""
    pass


@dataclass
class PauseControlConfig:
    """
    暂停控制配置数据类
    
    包含所有可配置的参数及其默认值
    """
    # 基础配置
    pause_mode: str = "immediate"  # 暂停模式: "immediate" 或 "graceful"
    buffer_size: int = 1000  # 缓冲区最大大小
    max_buffer_age: float = 30.0  # 缓冲数据最大年龄（秒）
    
    # 错误处理配置
    deadlock_timeout: float = 30.0  # 死锁检测超时时间（秒）
    max_recovery_attempts: int = 3  # 最大恢复尝试次数
    enable_auto_recovery: bool = True  # 是否启用自动恢复
    
    # 性能监控配置
    enable_performance_monitoring: bool = True  # 是否启用性能监控
    monitoring_interval: float = 1.0  # 监控间隔（秒）
    alert_threshold_pause_duration: float = 60.0  # 暂停时长告警阈值（秒）
    alert_threshold_buffer_usage: float = 0.8  # 缓冲区使用率告警阈值（0-1）
    
    # 日志配置
    log_level: str = "INFO"  # 日志级别: DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_to_file: bool = False  # 是否记录到文件
    log_file_path: str = "pause_control.log"  # 日志文件路径
    log_max_bytes: int = 10485760  # 日志文件最大大小（10MB）
    log_backup_count: int = 5  # 日志文件备份数量
    
    # 高级配置
    enable_metrics_collection: bool = True  # 是否启用指标收集
    metrics_export_interval: float = 60.0  # 指标导出间隔（秒）
    enable_buffer_overflow_protection: bool = True  # 是否启用缓冲区溢出保护
    buffer_overflow_strategy: str = "drop_oldest"  # 溢出策略: "drop_oldest", "drop_newest", "block"
    
    # 线程安全配置
    lock_timeout: float = 5.0  # 锁超时时间（秒）
    enable_deadlock_detection: bool = True  # 是否启用死锁检测
    
    # 调试配置
    debug_mode: bool = False  # 是否启用调试模式
    verbose_logging: bool = False  # 是否启用详细日志
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PauseControlConfig':
        """从字典创建配置对象"""
        return cls(**config_dict)



# 默认配置字典
PAUSE_CONTROL_CONFIG = {
    # 基础配置
    "pause_mode": "immediate",
    "buffer_size": 1000,
    "max_buffer_age": 30.0,
    
    # 错误处理配置
    "deadlock_timeout": 30.0,
    "max_recovery_attempts": 3,
    "enable_auto_recovery": True,
    
    # 性能监控配置
    "enable_performance_monitoring": True,
    "monitoring_interval": 1.0,
    "alert_threshold_pause_duration": 60.0,
    "alert_threshold_buffer_usage": 0.8,
    
    # 日志配置
    "log_level": "INFO",
    "log_to_file": False,
    "log_file_path": "pause_control.log",
    "log_max_bytes": 10485760,
    "log_backup_count": 5,
    
    # 高级配置
    "enable_metrics_collection": True,
    "metrics_export_interval": 60.0,
    "enable_buffer_overflow_protection": True,
    "buffer_overflow_strategy": "drop_oldest",
    
    # 线程安全配置
    "lock_timeout": 5.0,
    "enable_deadlock_detection": True,
    
    # 调试配置
    "debug_mode": False,
    "verbose_logging": False,
}


class ConfigValidator:
    """配置验证器"""
    
    # 配置参数的验证规则
    VALIDATION_RULES = {
        "pause_mode": {
            "type": str,
            "allowed_values": ["immediate", "graceful"],
            "description": "暂停模式"
        },
        "buffer_size": {
            "type": int,
            "min": 1,
            "max": 100000,
            "description": "缓冲区大小"
        },
        "max_buffer_age": {
            "type": (int, float),
            "min": 0.1,
            "max": 3600.0,
            "description": "缓冲数据最大年龄"
        },
        "deadlock_timeout": {
            "type": (int, float),
            "min": 1.0,
            "max": 300.0,
            "description": "死锁检测超时"
        },
        "max_recovery_attempts": {
            "type": int,
            "min": 1,
            "max": 10,
            "description": "最大恢复尝试次数"
        },
        "enable_auto_recovery": {
            "type": bool,
            "description": "自动恢复开关"
        },
        "enable_performance_monitoring": {
            "type": bool,
            "description": "性能监控开关"
        },
        "monitoring_interval": {
            "type": (int, float),
            "min": 0.1,
            "max": 60.0,
            "description": "监控间隔"
        },
        "alert_threshold_pause_duration": {
            "type": (int, float),
            "min": 1.0,
            "max": 3600.0,
            "description": "暂停时长告警阈值"
        },
        "alert_threshold_buffer_usage": {
            "type": (int, float),
            "min": 0.0,
            "max": 1.0,
            "description": "缓冲区使用率告警阈值"
        },
        "log_level": {
            "type": str,
            "allowed_values": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            "description": "日志级别"
        },
        "log_to_file": {
            "type": bool,
            "description": "文件日志开关"
        },
        "log_file_path": {
            "type": str,
            "description": "日志文件路径"
        },
        "log_max_bytes": {
            "type": int,
            "min": 1024,
            "max": 104857600,
            "description": "日志文件最大大小"
        },
        "log_backup_count": {
            "type": int,
            "min": 0,
            "max": 100,
            "description": "日志备份数量"
        },
        "enable_metrics_collection": {
            "type": bool,
            "description": "指标收集开关"
        },
        "metrics_export_interval": {
            "type": (int, float),
            "min": 1.0,
            "max": 3600.0,
            "description": "指标导出间隔"
        },
        "enable_buffer_overflow_protection": {
            "type": bool,
            "description": "缓冲区溢出保护开关"
        },
        "buffer_overflow_strategy": {
            "type": str,
            "allowed_values": ["drop_oldest", "drop_newest", "block"],
            "description": "缓冲区溢出策略"
        },
        "lock_timeout": {
            "type": (int, float),
            "min": 0.1,
            "max": 60.0,
            "description": "锁超时时间"
        },
        "enable_deadlock_detection": {
            "type": bool,
            "description": "死锁检测开关"
        },
        "debug_mode": {
            "type": bool,
            "description": "调试模式开关"
        },
        "verbose_logging": {
            "type": bool,
            "description": "详细日志开关"
        },
    }

    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> List[str]:
        """
        验证配置参数
        
        Args:
            config: 配置字典
            
        Returns:
            List[str]: 验证错误列表，空列表表示验证通过
        """
        errors = []
        
        for key, value in config.items():
            if key not in cls.VALIDATION_RULES:
                errors.append(f"未知的配置参数: {key}")
                continue
            
            rule = cls.VALIDATION_RULES[key]
            
            # 类型检查
            if not isinstance(value, rule["type"]):
                expected_type = rule["type"].__name__ if hasattr(rule["type"], "__name__") else str(rule["type"])
                errors.append(f"{key}: 类型错误，期望 {expected_type}，实际 {type(value).__name__}")
                continue
            
            # 允许值检查
            if "allowed_values" in rule and value not in rule["allowed_values"]:
                errors.append(f"{key}: 值 '{value}' 不在允许的值列表中 {rule['allowed_values']}")
            
            # 范围检查
            if "min" in rule and value < rule["min"]:
                errors.append(f"{key}: 值 {value} 小于最小值 {rule['min']}")
            
            if "max" in rule and value > rule["max"]:
                errors.append(f"{key}: 值 {value} 大于最大值 {rule['max']}")
        
        return errors
    
    @classmethod
    def validate_and_raise(cls, config: Dict[str, Any]):
        """
        验证配置并在失败时抛出异常
        
        Args:
            config: 配置字典
            
        Raises:
            ValidationError: 配置验证失败
        """
        errors = cls.validate_config(config)
        if errors:
            error_msg = "配置验证失败:\n" + "\n".join(f"  - {err}" for err in errors)
            raise ValidationError(error_msg)



class ConfigManager:
    """
    配置管理器
    
    提供配置的加载、保存、验证和运行时更新功能
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_file: 可选的配置文件路径
        """
        self._config_file = config_file
        self._config = PauseControlConfig()
        self._config_dict = PAUSE_CONTROL_CONFIG.copy()
        self._update_callbacks: List[callable] = []
        
        # 如果指定了配置文件，尝试加载
        if config_file and os.path.exists(config_file):
            try:
                self.load_from_file(config_file)
                logger.info(f"配置已从文件加载: {config_file}")
            except Exception as e:
                logger.warning(f"加载配置文件失败，使用默认配置: {e}")
        else:
            logger.info("使用默认配置")
    
    def get_config(self) -> PauseControlConfig:
        """获取配置对象"""
        return self._config
    
    def get_config_dict(self) -> Dict[str, Any]:
        """获取配置字典"""
        return self._config_dict.copy()
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键
            default: 默认值
            
        Returns:
            配置值或默认值
        """
        return self._config_dict.get(key, default)
    
    def set_value(self, key: str, value: Any, validate: bool = True):
        """
        设置配置值
        
        Args:
            key: 配置键
            value: 配置值
            validate: 是否验证配置
            
        Raises:
            ValidationError: 配置验证失败
            ConfigUpdateError: 配置更新失败
        """
        # 验证配置
        if validate:
            temp_config = self._config_dict.copy()
            temp_config[key] = value
            ConfigValidator.validate_and_raise(temp_config)
        
        # 更新配置
        old_value = self._config_dict.get(key)
        self._config_dict[key] = value
        
        # 更新配置对象
        try:
            self._config = PauseControlConfig.from_dict(self._config_dict)
        except Exception as e:
            # 回滚
            if old_value is not None:
                self._config_dict[key] = old_value
            raise ConfigUpdateError(f"更新配置失败: {e}")
        
        logger.info(f"配置已更新: {key} = {value} (旧值: {old_value})")
        
        # 触发回调
        self._trigger_update_callbacks(key, value, old_value)
    
    def update_config(self, config_updates: Dict[str, Any], validate: bool = True):
        """
        批量更新配置
        
        Args:
            config_updates: 配置更新字典
            validate: 是否验证配置
            
        Raises:
            ValidationError: 配置验证失败
            ConfigUpdateError: 配置更新失败
        """
        # 验证配置
        if validate:
            temp_config = self._config_dict.copy()
            temp_config.update(config_updates)
            ConfigValidator.validate_and_raise(temp_config)
        
        # 保存旧配置用于回滚
        old_config = self._config_dict.copy()
        
        # 更新配置
        try:
            self._config_dict.update(config_updates)
            self._config = PauseControlConfig.from_dict(self._config_dict)
        except Exception as e:
            # 回滚
            self._config_dict = old_config
            self._config = PauseControlConfig.from_dict(old_config)
            raise ConfigUpdateError(f"批量更新配置失败: {e}")
        
        logger.info(f"配置已批量更新: {list(config_updates.keys())}")
        
        # 触发回调
        for key, value in config_updates.items():
            old_value = old_config.get(key)
            self._trigger_update_callbacks(key, value, old_value)

    
    def load_from_file(self, config_file: str):
        """
        从文件加载配置
        
        Args:
            config_file: 配置文件路径
            
        Raises:
            ConfigError: 加载配置失败
        """
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
            
            # 验证配置
            ConfigValidator.validate_and_raise(file_config)
            
            # 合并配置（文件配置覆盖默认配置）
            self._config_dict = PAUSE_CONTROL_CONFIG.copy()
            self._config_dict.update(file_config)
            
            # 更新配置对象
            self._config = PauseControlConfig.from_dict(self._config_dict)
            
            logger.info(f"配置已从文件加载: {config_file}")
            
        except json.JSONDecodeError as e:
            raise ConfigError(f"配置文件格式错误: {e}")
        except Exception as e:
            raise ConfigError(f"加载配置文件失败: {e}")
    
    def save_to_file(self, config_file: Optional[str] = None):
        """
        保存配置到文件
        
        Args:
            config_file: 可选的配置文件路径，如果不指定则使用初始化时的路径
            
        Raises:
            ConfigError: 保存配置失败
        """
        target_file = config_file or self._config_file
        
        if not target_file:
            raise ConfigError("未指定配置文件路径")
        
        try:
            with open(target_file, 'w', encoding='utf-8') as f:
                json.dump(self._config_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"配置已保存到文件: {target_file}")
            
        except Exception as e:
            raise ConfigError(f"保存配置文件失败: {e}")
    
    def load_from_env(self, prefix: str = "PAUSE_CONTROL_"):
        """
        从环境变量加载配置
        
        环境变量格式: {prefix}{KEY}，例如 PAUSE_CONTROL_BUFFER_SIZE
        
        Args:
            prefix: 环境变量前缀
        """
        updates = {}
        
        for key in PAUSE_CONTROL_CONFIG.keys():
            env_key = f"{prefix}{key.upper()}"
            env_value = os.environ.get(env_key)
            
            if env_value is not None:
                # 类型转换
                try:
                    rule = ConfigValidator.VALIDATION_RULES.get(key, {})
                    expected_type = rule.get("type", str)
                    
                    if expected_type == bool:
                        converted_value = env_value.lower() in ('true', '1', 'yes', 'on')
                    elif expected_type == int:
                        converted_value = int(env_value)
                    elif expected_type == float or expected_type == (int, float):
                        converted_value = float(env_value)
                    else:
                        converted_value = env_value
                    
                    updates[key] = converted_value
                    logger.debug(f"从环境变量加载配置: {env_key} = {converted_value}")
                    
                except (ValueError, TypeError) as e:
                    logger.warning(f"环境变量 {env_key} 类型转换失败: {e}")
        
        if updates:
            self.update_config(updates, validate=True)
            logger.info(f"从环境变量加载了 {len(updates)} 个配置项")
    
    def register_update_callback(self, callback: callable):
        """
        注册配置更新回调
        
        回调函数签名: callback(key: str, new_value: Any, old_value: Any)
        
        Args:
            callback: 回调函数
        """
        self._update_callbacks.append(callback)
        logger.debug(f"注册配置更新回调: {callback.__name__}")
    
    def _trigger_update_callbacks(self, key: str, new_value: Any, old_value: Any):
        """触发配置更新回调"""
        for callback in self._update_callbacks:
            try:
                callback(key, new_value, old_value)
            except Exception as e:
                logger.error(f"配置更新回调执行失败: {e}")
    
    def reset_to_defaults(self):
        """重置为默认配置"""
        self._config_dict = PAUSE_CONTROL_CONFIG.copy()
        self._config = PauseControlConfig.from_dict(self._config_dict)
        logger.info("配置已重置为默认值")
    
    def get_config_summary(self) -> str:
        """
        获取配置摘要
        
        Returns:
            str: 配置摘要字符串
        """
        lines = ["暂停控制配置摘要:"]
        lines.append("=" * 50)
        
        # 按类别组织配置
        categories = {
            "基础配置": ["pause_mode", "buffer_size", "max_buffer_age"],
            "错误处理": ["deadlock_timeout", "max_recovery_attempts", "enable_auto_recovery"],
            "性能监控": ["enable_performance_monitoring", "monitoring_interval", 
                       "alert_threshold_pause_duration", "alert_threshold_buffer_usage"],
            "日志配置": ["log_level", "log_to_file", "log_file_path", 
                       "log_max_bytes", "log_backup_count"],
            "高级配置": ["enable_metrics_collection", "metrics_export_interval",
                       "enable_buffer_overflow_protection", "buffer_overflow_strategy"],
            "线程安全": ["lock_timeout", "enable_deadlock_detection"],
            "调试配置": ["debug_mode", "verbose_logging"],
        }
        
        for category, keys in categories.items():
            lines.append(f"\n{category}:")
            for key in keys:
                value = self._config_dict.get(key)
                lines.append(f"  {key}: {value}")
        
        lines.append("=" * 50)
        return "\n".join(lines)



def setup_logging(config: PauseControlConfig):
    """
    根据配置设置日志系统
    
    Args:
        config: 暂停控制配置对象
    """
    # 设置日志级别
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # 清除现有的处理器
    logger.handlers.clear()
    
    # 创建格式化器
    if config.verbose_logging:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器
    if config.log_to_file:
        try:
            from logging.handlers import RotatingFileHandler
            
            file_handler = RotatingFileHandler(
                config.log_file_path,
                maxBytes=config.log_max_bytes,
                backupCount=config.log_backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            logger.info(f"日志文件已启用: {config.log_file_path}")
        except Exception as e:
            logger.error(f"创建日志文件处理器失败: {e}")
    
    logger.info(f"日志系统已配置: 级别={config.log_level}, 文件日志={config.log_to_file}")


# 全局配置管理器实例
_global_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_file: Optional[str] = None) -> ConfigManager:
    """
    获取全局配置管理器实例
    
    Args:
        config_file: 可选的配置文件路径（仅在首次调用时有效）
        
    Returns:
        ConfigManager: 配置管理器实例
    """
    global _global_config_manager
    
    if _global_config_manager is None:
        _global_config_manager = ConfigManager(config_file)
        
        # 尝试从环境变量加载配置
        try:
            _global_config_manager.load_from_env()
        except Exception as e:
            logger.warning(f"从环境变量加载配置失败: {e}")
        
        # 设置日志系统
        setup_logging(_global_config_manager.get_config())
    
    return _global_config_manager


def reset_config_manager():
    """重置全局配置管理器"""
    global _global_config_manager
    _global_config_manager = None
    logger.info("全局配置管理器已重置")


# 便捷函数
def get_config() -> PauseControlConfig:
    """获取当前配置"""
    return get_config_manager().get_config()


def get_config_value(key: str, default: Any = None) -> Any:
    """获取配置值"""
    return get_config_manager().get_value(key, default)


def update_config(config_updates: Dict[str, Any], validate: bool = True):
    """更新配置"""
    get_config_manager().update_config(config_updates, validate)


def print_config_summary():
    """打印配置摘要"""
    print(get_config_manager().get_config_summary())
