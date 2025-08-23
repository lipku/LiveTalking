#!/usr/bin/env python3
"""
GPU 监控模块
提供 GPU 使用率、显存、温度等信息
"""

import json
import subprocess
import platform
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class GPUMonitor:
    """GPU 监控类"""
    
    def __init__(self):
        self.has_nvidia_smi = self._check_nvidia_smi()
        self.has_pynvml = False
        
        # 尝试导入 pynvml
        try:
            import pynvml
            pynvml.nvmlInit()
            self.has_pynvml = True
            self.pynvml = pynvml
            logger.info("pynvml 初始化成功")
        except Exception as e:
            logger.warning(f"pynvml 初始化失败: {e}")
    
    def _check_nvidia_smi(self) -> bool:
        """检查 nvidia-smi 是否可用"""
        try:
            subprocess.run(['nvidia-smi'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def get_gpu_info_nvidia_smi(self) -> Optional[Dict[str, Any]]:
        """使用 nvidia-smi 获取 GPU 信息"""
        if not self.has_nvidia_smi:
            return None
        
        try:
            # 获取 GPU 使用率和显存信息
            result = subprocess.run([
                'nvidia-smi',
                '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, check=True)
            
            output = result.stdout.strip()
            if not output:
                return None
            
            # 解析输出
            values = output.split(', ')
            if len(values) >= 5:
                return {
                    'gpu_usage': float(values[0]),
                    'mem_used': float(values[1]),
                    'mem_total': float(values[2]),
                    'temperature': float(values[3]),
                    'power': float(values[4]) if values[4] != '[N/A]' else 0
                }
        except Exception as e:
            logger.error(f"nvidia-smi 获取信息失败: {e}")
        
        return None
    
    def get_gpu_info_pynvml(self) -> Optional[Dict[str, Any]]:
        """使用 pynvml 获取 GPU 信息"""
        if not self.has_pynvml:
            return None
        
        try:
            device_count = self.pynvml.nvmlDeviceGetCount()
            if device_count == 0:
                return None
            
            # 获取第一个 GPU 的信息
            handle = self.pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # 获取使用率
            utilization = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            # 获取显存信息
            mem_info = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # 获取温度
            temperature = self.pynvml.nvmlDeviceGetTemperature(handle, self.pynvml.NVML_TEMPERATURE_GPU)
            
            # 获取功耗
            try:
                power = self.pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # 转换为瓦特
            except:
                power = 0
            
            return {
                'gpu_usage': utilization.gpu,
                'mem_used': mem_info.used / (1024 * 1024),  # 转换为 MB
                'mem_total': mem_info.total / (1024 * 1024),  # 转换为 MB
                'temperature': temperature,
                'power': power
            }
        except Exception as e:
            logger.error(f"pynvml 获取信息失败: {e}")
        
        return None
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """获取 GPU 信息"""
        # 优先使用 pynvml
        info = self.get_gpu_info_pynvml()
        
        # 如果 pynvml 失败，尝试 nvidia-smi
        if info is None:
            info = self.get_gpu_info_nvidia_smi()
        
        # 如果都失败，返回模拟数据
        if info is None:
            import random
            info = {
                'gpu_usage': random.uniform(20, 80),
                'mem_used': random.uniform(2000, 8000),
                'mem_total': 24576,  # 24GB
                'temperature': random.uniform(40, 70),
                'power': random.uniform(50, 250),
                'is_mock': True  # 标记为模拟数据
            }
        
        # 添加额外信息
        info['mem_usage_percent'] = (info['mem_used'] / info['mem_total']) * 100 if info['mem_total'] > 0 else 0
        info['status'] = 'ok' if info.get('gpu_usage', 0) < 90 else 'high'
        
        return info
    
    def get_gpu_info_detailed(self) -> Dict[str, Any]:
        """获取详细的 GPU 信息"""
        basic_info = self.get_gpu_info()
        
        # 添加更多详细信息
        detailed_info = {
            **basic_info,
            'gpu_count': 0,
            'gpu_name': 'Unknown',
            'driver_version': 'Unknown',
            'cuda_version': 'Unknown'
        }
        
        if self.has_pynvml:
            try:
                detailed_info['gpu_count'] = self.pynvml.nvmlDeviceGetCount()
                if detailed_info['gpu_count'] > 0:
                    handle = self.pynvml.nvmlDeviceGetHandleByIndex(0)
                    detailed_info['gpu_name'] = self.pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                
                detailed_info['driver_version'] = self.pynvml.nvmlSystemGetDriverVersion().decode('utf-8')
            except:
                pass
        
        elif self.has_nvidia_smi:
            try:
                # 获取 GPU 名称
                result = subprocess.run([
                    'nvidia-smi',
                    '--query-gpu=name',
                    '--format=csv,noheader'
                ], capture_output=True, text=True, check=True)
                detailed_info['gpu_name'] = result.stdout.strip()
                
                # 获取 CUDA 版本
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
                for line in result.stdout.split('\n'):
                    if 'CUDA Version' in line:
                        cuda_version = line.split('CUDA Version:')[1].split()[0]
                        detailed_info['cuda_version'] = cuda_version
                        break
            except:
                pass
        
        return detailed_info


# 全局 GPU 监控实例
gpu_monitor = GPUMonitor()


def get_gpu_status() -> Dict[str, Any]:
    """获取 GPU 状态的便捷函数"""
    return gpu_monitor.get_gpu_info()


def get_gpu_status_detailed() -> Dict[str, Any]:
    """获取详细 GPU 状态的便捷函数"""
    return gpu_monitor.get_gpu_info_detailed()


if __name__ == "__main__":
    # 测试代码
    import time
    
    print("GPU 监控测试")
    print("-" * 50)
    
    # 获取详细信息
    detailed = get_gpu_status_detailed()
    print(f"GPU 名称: {detailed['gpu_name']}")
    print(f"驱动版本: {detailed['driver_version']}")
    print(f"CUDA 版本: {detailed['cuda_version']}")
    print(f"GPU 数量: {detailed['gpu_count']}")
    print("-" * 50)
    
    # 实时监控
    print("\n实时监控 (按 Ctrl+C 退出):")
    try:
        while True:
            info = get_gpu_status()
            
            if info.get('is_mock'):
                print("⚠️  使用模拟数据 (未检测到 GPU)")
            
            print(f"\r"
                  f"GPU: {info['gpu_usage']:.1f}% | "
                  f"显存: {info['mem_used']:.0f}/{info['mem_total']:.0f} MB ({info['mem_usage_percent']:.1f}%) | "
                  f"温度: {info['temperature']:.0f}°C | "
                  f"功耗: {info['power']:.1f}W",
                  end='', flush=True)
            
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n监控停止")
