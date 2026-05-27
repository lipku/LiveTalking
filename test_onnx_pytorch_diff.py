#!/usr/bin/env python3
"""
对比测试：ONNX 和 PyTorch UNet + VAE 输出差异
"""

import torch
import numpy as np
import os
import sys
import json
import time

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"[INFO] 使用 CUDA: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("[WARNING] CUDA 不可用，使用 CPU")


def sync_device():
    if device.type == 'cuda':
        torch.cuda.synchronize()


def compare_outputs(pytorch_output, onnx_output, name="模型"):
    """对比两个输出的差异"""
    diff = (pytorch_output - onnx_output).abs()
    print(f"\n   {name} 输出对比:")
    print(f"   PyTorch 输出范围: [{pytorch_output.min():.4f}, {pytorch_output.max():.4f}]")
    print(f"   ONNX    输出范围: [{onnx_output.min():.4f}, {onnx_output.max():.4f}]")
    print(f"   绝对差异范围: [{diff.min():.6f}, {diff.max():.6f}]")
    print(f"   平均绝对差异: {diff.mean():.6f}")
    
    # 检查是否有显著差异
    if diff.max() > 0.1:
        print(f"   ⚠️ 警告: {name} 存在显著差异!")
        return False
    elif diff.max() > 0.01:
        print(f"   ⚠️ 注意: {name} 有中等差异")
        return True
    else:
        print(f"   ✅ {name} 输出基本一致")
        return True


def test_unet_output():
    """对比 PyTorch 和 ONNX UNet 的输出"""
    
    print("=" * 60)
    print("对比测试：PyTorch vs ONNX UNet 输出")
    print("=" * 60)
    
    # 1. 加载 PyTorch 模型
    print("\n[1] 加载 PyTorch UNet 模型...")
    from diffusers import UNet2DConditionModel
    
    config_path = "models/musetalk/musetalk.json"
    with open(config_path, 'r') as f:
        unet_config = json.load(f)
    
    pytorch_unet = UNet2DConditionModel(**unet_config)
    weights = torch.load("models/musetalk/unet.pth", map_location='cpu')
    pytorch_unet.load_state_dict(weights)
    pytorch_unet = pytorch_unet.half().to(device)
    pytorch_unet.eval()
    print("   PyTorch UNet 模型加载完成")
    
    # 2. 加载 ONNX 模型
    print("\n[2] 加载 ONNX UNet 模型...")
    import onnxruntime as ort
    
    onnx_path = "./models/onnx/musetalk/unet.onnx"
    if not os.path.exists(onnx_path):
        print(f"   错误: ONNX UNet 模型不存在: {onnx_path}")
        return None
    
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    print(f"   ONNX UNet 模型加载完成，使用 providers: {session.get_providers()}")
    
    # 3. 创建相同的输入
    print("\n[3] 创建测试输入...")
    batch_size = 1
    latent_h, latent_w = 32, 32
    audio_dim = 384
    seq_len = 50
    
    # 使用相同的随机种子确保输入一致
    torch.manual_seed(42)
    np.random.seed(42)
    
    latent_batch = torch.randn(batch_size, 8, latent_h, latent_w, dtype=torch.float16, device=device)
    audio_batch = torch.randn(batch_size, seq_len, audio_dim, dtype=torch.float16, device=device)
    timestep = torch.tensor([0], device=device)
    
    print(f"   latent_batch: {latent_batch.shape}, dtype={latent_batch.dtype}")
    print(f"   audio_batch: {audio_batch.shape}, dtype={audio_batch.dtype}")
    print(f"   timestep: {timestep.shape}, dtype={timestep.dtype}")
    
    # 4. PyTorch 推理
    print("\n[4] PyTorch UNet 推理...")
    torch.cuda.synchronize() if device.type == 'cuda' else None
    pt_start = time.time()
    with torch.no_grad():
        pytorch_unet_output = pytorch_unet(
            latent_batch,
            timestep,
            encoder_hidden_states=audio_batch  # [B, 50, 384] — 使用全部时间步，与实际推理一致
        ).sample
    torch.cuda.synchronize() if device.type == 'cuda' else None
    pt_time = time.time() - pt_start
    print(f"   PyTorch UNet 输出: {pytorch_unet_output.shape}, 耗时: {pt_time*1000:.1f}ms")
    
    # 5. ONNX 推理
    print("\n[5] ONNX UNet 推理...")
    torch.cuda.synchronize() if device.type == 'cuda' else None
    onnx_start = time.time()
    
    # 准备 ONNX 输入 (转为 numpy float16)
    onnx_latents = latent_batch.cpu().half().numpy()
    onnx_audio = audio_batch.cpu().half().numpy()
    onnx_timestep = timestep.cpu().numpy().astype(np.int64)
    
    outputs = session.run(
        None,
        {
            'latents': onnx_latents,
            'timestep': onnx_timestep,
            'audio_embedding': onnx_audio
        }
    )
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    onnx_time = time.time() - onnx_start
    
    onnx_unet_output = torch.from_numpy(outputs[0]).to(device=device, dtype=torch.float16)
    print(f"   ONNX UNet 输出: {onnx_unet_output.shape}, 耗时: {onnx_time*1000:.1f}ms")
    
    # 6. 对比 UNet 差异
    print("\n[6] 对比 UNet 差异...")
    compare_outputs(pytorch_unet_output, onnx_unet_output, "UNet")
    
    return pytorch_unet_output, onnx_unet_output


def test_vae_output(unet_output_pytorch, unet_output_onnx):
    """对比 PyTorch 和 ONNX VAE 的输出"""
    
    print("\n" + "=" * 60)
    print("对比测试：PyTorch vs ONNX VAE 输出")
    print("=" * 60)
    
    # 检查 ONNX VAE 是否存在
    vae_onnx_path = "./models/onnx/musetalk/vae_decoder.onnx"
    if not os.path.exists(vae_onnx_path):
        print(f"\n   ONNX VAE 模型不存在: {vae_onnx_path}")
        print("   跳过 VAE 对比测试")
        return
    
    # 1. 加载 PyTorch VAE
    print("\n[1] 加载 PyTorch VAE 模型...")
    from diffusers import AutoencoderKL
    
    pytorch_vae = AutoencoderKL.from_pretrained("models/musetalk/vae")
    pytorch_vae = pytorch_vae.half().to(device)
    pytorch_vae.eval()
    print("   PyTorch VAE 模型加载完成")
    
    # 2. 加载 ONNX VAE
    print("\n[2] 加载 ONNX VAE 模型...")
    import onnxruntime as ort
    
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    vae_session = ort.InferenceSession(vae_onnx_path, providers=providers)
    print(f"   ONNX VAE 模型加载完成，使用 providers: {vae_session.get_providers()}")
    
    # 3. 准备输入 — 使用 UNet 输出的前 4 个通道作为 VAE 输入
    print("\n[3] 准备 VAE 输入...")
    if unet_output_pytorch is not None:
        # 取 UNet 输出 noise_pred 的前 4 个通道作为 VAE latent
        test_latent = unet_output_pytorch[:, :4, :, :].detach()
        print(f"   从 UNet 输出取前4通道作为 latent: {test_latent.shape}")
    else:
        torch.manual_seed(42)
        test_latent = torch.randn(1, 4, 32, 32, dtype=torch.float16, device=device)
        print(f"   使用随机 latent: {test_latent.shape}")
    
    # 4. PyTorch VAE 推理
    print("\n[4] PyTorch VAE 推理...")
    torch.cuda.synchronize() if device.type == 'cuda' else None
    pt_start = time.time()
    with torch.no_grad():
        scaling_factor = pytorch_vae.config.scaling_factor
        latent_input = test_latent / scaling_factor
        pytorch_vae_output = pytorch_vae.decode(latent_input).sample
        pytorch_vae_output = (pytorch_vae_output / 2 + 0.5).clamp(0, 1)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    pt_time = time.time() - pt_start
    print(f"   PyTorch VAE 输出: {pytorch_vae_output.shape}, 耗时: {pt_time*1000:.1f}ms")
    
    # 5. ONNX VAE 推理
    print("\n[5] ONNX VAE 推理...")
    torch.cuda.synchronize() if device.type == 'cuda' else None
    onnx_start = time.time()
    
    onnx_latent = test_latent.cpu().half().numpy()
    vae_outputs = vae_session.run(None, {'latents': onnx_latent})
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    onnx_time = time.time() - onnx_start
    
    onnx_vae_output = torch.from_numpy(vae_outputs[0]).to(device=device, dtype=torch.float16)
    print(f"   ONNX VAE 输出: {onnx_vae_output.shape}, 耗时: {onnx_time*1000:.1f}ms")
    
    # 6. 对比 VAE 差异
    print("\n[6] 对比 VAE 差异...")
    compare_outputs(pytorch_vae_output, onnx_vae_output, "VAE")


def test_full_pipeline():
    """测试完整流程：UNet + VAE"""
    
    print("=" * 60)
    print("完整流程测试：UNet + VAE")
    print("=" * 60)
    
    # 测试 UNet
    result = test_unet_output()
    if result is None:
        return
    
    pytorch_unet_output, onnx_unet_output = result
    
    # 测试 VAE
    test_vae_output(pytorch_unet_output, onnx_unet_output)
    
    # 清理 GPU 缓存
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    print("\n" + "=" * 60)
    print("所有测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_full_pipeline()
