import onnxruntime as ort

print("Available providers:", ort.get_available_providers())

if 'CUDAExecutionProvider' in ort.get_available_providers():
    print("✅ CUDA is available for ONNX Runtime")
else:
    print("❌ CUDA is NOT available")