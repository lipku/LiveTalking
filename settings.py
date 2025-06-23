import os
import argparse
from ruamel.yaml import YAML
from collections.abc import Mapping

default_config_file = "config.yaml"

def get_project_dir():
    """获取项目根目录"""
    return os.path.dirname(os.path.abspath(__file__))


def read_config():
    """读取配置文件"""
    yaml = YAML()
    yaml.preserve_quotes = True
    # 调用 load_config 函数获取正确的配置文件路径
    config_path = load_config(default_config_file)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f)
    return config

def load_config(config_path=None):
    global default_config_file
    """获取配置文件路径，优先使用私有配置文件（若存在）。

    Returns:
       str: 配置文件路径（相对路径或默认路径）
    """
    config_file = default_config_file
    if os.path.exists(get_project_dir() + "/" + default_config_file):
        config_file = get_project_dir() + "/" + default_config_file
    print("config_path:", config_file)
    return config_file
    
def print_config(config_data, indent=0):
    """递归打印配置内容，格式化输出
    
    Args:
        config_data: 配置数据 (dict 或其他)
        indent: 缩进级别
    """
    if isinstance(config_data, Mapping):
        for key, value in config_data.items():
            print(' ' * indent + f"{key}:")
            print_config(value, indent + 4)
    elif isinstance(config_data, (list, tuple)):
        for i, item in enumerate(config_data):
            print(' ' * indent + f"- [{i}]")
            print_config(item, indent + 4)
    else:
        print(' ' * indent + str(config_data))

if __name__ == '__main__':
    try:
        # 确保传递正确的配置文件路径
        config = read_config()
        print(config["TTS"]["TencentTTS"]["appid"])
    except Exception as e:
        print(f"Error loading config: {e}")