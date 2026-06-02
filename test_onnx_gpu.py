import onnxruntime as ort

print("Available providers:", ort.get_available_providers())

available = ort.get_available_providers()
if 'CUDAExecutionProvider' in available:
    print("✅ CUDA — ONNX Runtime 使用 NVIDIA GPU")
elif 'MACAExecutionProvider' in available:
    print("✅ MACA — ONNX Runtime 使用沐曦 Metax GPU")
else:
    print("❌ ONNX Runtime 无 GPU Provider 可用，将使用 CPU")