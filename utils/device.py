import torch
import warnings


def _is_metax():
    """检测是否为沐曦 Metax GPU"""
    if not torch.cuda.is_available():
        return False
    try:
        name = torch.cuda.get_device_name(0).lower()
        return 'metax' in name or 'maca' in name
    except Exception:
        return False


def initialize_device():
    if torch.cuda.is_available():
        # Metax C500 兼容: 某些 Metax 驱动版本在 device_count 上不稳定
        # 强制使用 cuda:0，避免驱动层对 device>0 的检查
        return torch.device('cuda:0')
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def is_metax_gpu():
    return _is_metax()


def get_onnx_providers():
    """
    获取 ONNX Runtime 可用的 GPU ExecutionProvider。

    返回:
        list: 优先使用 GPU provider，最后兜底 CPU
              沐曦 Metax → ['MACAExecutionProvider', 'CPUExecutionProvider']
              NVIDIA     → ['CUDAExecutionProvider', 'CPUExecutionProvider']
              纯 CPU     → ['CPUExecutionProvider']
    """
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
    except ImportError:
        return ['CPUExecutionProvider']

    if 'CUDAExecutionProvider' in available:
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']

    if 'MACAExecutionProvider' in available:
        return ['MACAExecutionProvider', 'CPUExecutionProvider']

    if 'ROCMExecutionProvider' in available:
        return ['ROCMExecutionProvider', 'CPUExecutionProvider']

    return ['CPUExecutionProvider']
