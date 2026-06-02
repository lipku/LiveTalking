"""
硬件监控工具 — 检测 CPU/GPU 设备信息并周期性监控资源占用
不依赖 psutil，使用 torch.cuda + /proc 文件系统
"""

import os
import time
import threading
import torch

from utils.logger import logger


def detect_device():
    """检测当前使用的设备类型，返回 (device, is_gpu, device_name, meta)"""
    meta = {}

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        is_gpu = True

        try:
            device_name = torch.cuda.get_device_name(0)
            capability = torch.cuda.get_device_capability(0)
            meta['capability'] = f"{capability[0]}.{capability[1]}"
        except Exception:
            device_name = "Unknown GPU"
            meta['capability'] = "unknown"

        try:
            total_mem = torch.cuda.get_device_properties(0).total_memory
            meta['vram_total_gb'] = round(total_mem / (1024**3), 2)
        except Exception:
            meta['vram_total_gb'] = 'unknown'

        # 检测是否为沐曦 Metax
        if 'metax' in device_name.lower():
            meta['vendor'] = 'Metax (沐曦)'
        elif 'maca' in device_name.lower():
            meta['vendor'] = 'Metax/MACA (沐曦)'
        else:
            meta['vendor'] = 'NVIDIA'

        meta['gpu_count'] = torch.cuda.device_count()
    else:
        device = torch.device('cpu')
        is_gpu = False
        device_name = 'CPU'
        meta['vendor'] = 'CPU'

    return device, is_gpu, device_name, meta


def _read_proc(path):
    """安全读取 /proc 文件"""
    try:
        with open(path, 'r') as f:
            return f.read()
    except Exception:
        return None


def get_gpu_memory():
    """获取 GPU 显存占用 (MB)"""
    if not torch.cuda.is_available():
        return None
    try:
        free, total = torch.cuda.mem_get_info(0)
        used = total - free
        return {
            'used_mb': round(used / (1024**2), 1),
            'free_mb': round(free / (1024**2), 1),
            'total_mb': round(total / (1024**2), 1),
            'used_pct': round(used / total * 100, 1),
        }
    except Exception:
        return None


def _check_nvidia_smi():
    """检查 nvidia-smi 是否可用"""
    if not torch.cuda.is_available():
        return False
    try:
        name = torch.cuda.get_device_name(0).lower()
        if 'metax' in name or 'maca' in name:
            return False
    except Exception:
        pass
    return os.path.exists('/usr/bin/nvidia-smi') or os.path.exists('/usr/local/bin/nvidia-smi')


def get_gpu_utilization():
    """获取 GPU 利用率 (仅 NVIDIA GPU 有效)"""
    if not _check_nvidia_smi():
        return None
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            val = result.stdout.strip()
            if val:
                return {'gpu_util_pct': int(val)}
    except Exception:
        pass
    return None


def get_cpu_load():
    """获取 CPU 负载 (平均负载 + 使用率估算)"""
    loadavg = _read_proc('/proc/loadavg')
    cpu_count = os.cpu_count() or 1
    result = {'cpu_count': cpu_count}

    if loadavg:
        fields = loadavg.split()
        result['load_1min'] = float(fields[0])
        result['load_5min'] = float(fields[1])
        result['load_15min'] = float(fields[2])

    # 通过 /proc/stat 计算 CPU 使用率
    stat1 = _read_proc('/proc/stat')
    if stat1:
        time.sleep(0.5)
        stat2 = _read_proc('/proc/stat')
        if stat2:
            def _parse_cpu(s):
                parts = s.splitlines()[0].split()
                return sum(int(x) for x in parts[1:])

            idle1 = int(stat1.splitlines()[0].split()[4])
            idle2 = int(stat2.splitlines()[0].split()[4])
            total1 = _parse_cpu(stat1)
            total2 = _parse_cpu(stat2)

            delta_idle = idle2 - idle1
            delta_total = total2 - total1
            if delta_total > 0:
                result['cpu_usage_pct'] = round((1 - delta_idle / delta_total) * 100, 1)

    return result


def get_memory():
    """获取系统 RAM 信息"""
    meminfo = _read_proc('/proc/meminfo')
    if not meminfo:
        return None

    def _get_val(key):
        for line in meminfo.splitlines():
            if line.startswith(key + ':'):
                kb = int(line.split()[1])
                return round(kb / 1024, 1)  # KB → MB
        return 0

    total_mb = _get_val('MemTotal')
    free_mb = _get_val('MemFree')
    buffers_mb = _get_val('Buffers')
    cached_mb = _get_val('Cached')
    available_mb = _get_val('MemAvailable')

    if available_mb == 0:
        available_mb = free_mb + buffers_mb + cached_mb

    used_mb = total_mb - available_mb
    return {
        'total_mb': total_mb,
        'used_mb': round(used_mb, 1),
        'available_mb': available_mb,
        'used_pct': round(used_mb / total_mb * 100, 1) if total_mb > 0 else 0,
    }


def log_device_info():
    """一次性打印设备信息"""
    device, is_gpu, device_name, meta = detect_device()

    logger.info(f"")
    logger.info(f"{'='*50}")
    logger.info(f"  设备信息")
    logger.info(f"{'='*50}")
    logger.info(f"  计算设备: {device}")
    logger.info(f"  设备名称: {device_name}")
    logger.info(f"  厂商: {meta.get('vendor', 'unknown')}")

    if is_gpu:
        logger.info(f"  GPU 数量: {meta.get('gpu_count', 1)}")
        logger.info(f"  显存总量: {meta.get('vram_total_gb', '?')} GB")
        logger.info(f"  算力: {meta.get('capability', '?')}")
    else:
        logger.info(f"  CPU 核心数: {os.cpu_count() or '?'}")

    logger.info(f"{'='*50}")
    logger.info(f"")

    return device, is_gpu, device_name, meta


def log_resource_usage():
    """打印一次当前资源占用"""
    gpu_mem = get_gpu_memory()
    if gpu_mem:
        gpu_util = get_gpu_utilization()
        util_str = f", 利用率: {gpu_util['gpu_util_pct']}%" if gpu_util else ""
        logger.info(
            f"[资源] GPU {gpu_mem['used_mb']:.0f}/{gpu_mem['total_mb']:.0f} MB "
            f"({gpu_mem['used_pct']}%){util_str}"
        )

    ram = get_memory()
    if ram:
        cpu = get_cpu_load()
        cpu_str = f", CPU: {cpu.get('cpu_usage_pct', '?')}%" if cpu and 'cpu_usage_pct' in cpu else ""
        logger.info(
            f"[资源] RAM {ram['used_mb']:.0f}/{ram['total_mb']:.0f} MB "
            f"({ram['used_pct']}%){cpu_str}"
        )


def start_monitoring(interval=60):
    """启动后台监控线程，每隔 interval 秒打印一次资源占用"""
    def _loop():
        while True:
            time.sleep(interval)
            try:
                log_resource_usage()
            except Exception:
                pass

    thread = threading.Thread(target=_loop, daemon=True, name='hw-monitor')
    thread.start()
    return thread
