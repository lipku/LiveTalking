#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch 模型转 ONNX 脚本
支持 wav2lip、musetalk、ultralight 等模型的转换

用法:
    python convert_to_onnx.py --model wav2lip --input models/wav2lip.pth --output models/onnx/wav2lip.onnx
    python convert_to_onnx.py --model musetalk --input models/musetalk.pth --output models/onnx/musetalk.onnx
"""

import os
import sys
import argparse
import torch
import torch.onnx
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))


def ensure_dir(path):
    """确保目录存在"""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def convert_wav2lip(input_path, output_path, opset_version=11):
    """
    转换 Wav2Lip 模型到 ONNX
    
    输入:
        - audio_sequences: [B, 1, 80, 16] 或 [B, T, 1, 80, 16] (mel spectrogram)
        - face_sequences: [B, 6, H, W] 或 [B, T, 6, H, W] (reference face frames)
    输出:
        - outputs: [B, 3, H, W] 或 [B, T, 3, H, W] (generated face)
    """
    # 注意：实际使用的是 wav2lip_v2.py 中的 Wav2Lip 类（256x256 分辨率）
    from avatars.wav2lip.models import Wav2Lip
    
    print(f"[INFO] 加载 Wav2Lip 模型: {input_path}")
    model = Wav2Lip()
    
    # 加载权重
    checkpoint = torch.load(input_path, map_location='cpu')
    state_dict = checkpoint.get("state_dict", checkpoint)
    
    # 移除 'module.' 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace('module.', '')] = v
    
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    
    # 设置输入尺寸 (batch_size=1, 支持动态 batch)
    # wav2lip_v2 使用 256x256 分辨率
    batch_size = 1
    mel_size = 80
    mel_time = 16
    img_size = 256  # wav2lip_v2 使用 256x256
    
    # 创建示例输入
    dummy_audio = torch.randn(batch_size, 1, mel_size, mel_time)
    dummy_face = torch.randn(batch_size, 6, img_size, img_size)
    
    print(f"[INFO] 输入尺寸: audio={dummy_audio.shape}, face={dummy_face.shape}")
    
    # 导出 ONNX
    ensure_dir(output_path)
    print(f"[INFO] 导出 ONNX 到: {output_path}")
    
    torch.onnx.export(
        model,
        (dummy_audio, dummy_face),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['audio_sequences', 'face_sequences'],
        output_names=['outputs'],
        dynamic_axes={
            'audio_sequences': {0: 'batch_size'},
            'face_sequences': {0: 'batch_size'},
            'outputs': {0: 'batch_size'}
        }
    )
    
    print(f"[SUCCESS] Wav2Lip ONNX 模型已保存: {output_path}")
    return output_path


class MuseTalkUNetWrapper(torch.nn.Module):
    """
    MuseTalk UNet 包装器，用于 ONNX 导出
    将 audio_embedding 直接作为 encoder_hidden_states 传入 UNet
    """
    def __init__(self, unet_model):
        super().__init__()
        self.unet = unet_model
    
    def forward(self, latents, timestep, audio_embedding):
        if audio_embedding.dim() == 3:
            encoder_hidden_states = audio_embedding  # [B, T, 384] -> 保持全部时间步
        else:
            encoder_hidden_states = audio_embedding.unsqueeze(1)  # [B, 384] -> [B, 1, 384]
        output = self.unet(
            latents,
            timestep,
            encoder_hidden_states=encoder_hidden_states
        )
        return output.sample


def convert_musetalk(input_path, output_path, opset_version=14):
    """
    转换 MuseTalk 模型到 ONNX
    
    输入:
        - latents: [B, 8, H, W] (VAE 编码后的图像特征)
        - timestep: [B] (扩散模型时间步)
        - audio_embedding: [B, 384] (Whisper 音频特征)
    输出:
        - noise_pred: [B, 8, H, W] (预测的噪声)
    
    注意: MuseTalk 使用 UNet2DConditionModel 的 attention 机制，
          需要 opset_version >= 14 以支持 scaled_dot_product_attention 操作符
    """
    import json
    import time
    from diffusers import UNet2DConditionModel
    
    print(f"[INFO] 加载 MuseTalk 模型: {input_path}")
    start_time = time.time()
    
    # 加载到 GPU（加速导出）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] 使用设备: {device}")
    
    # 加载配置
    config_path = input_path.replace('unet.pth', 'musetalk.json')
    with open(config_path, 'r') as f:
        unet_config = json.load(f)
    
    # 创建模型
    unet = UNet2DConditionModel(**unet_config)
    
    # 加载权重
    weights = torch.load(input_path, map_location='cpu')
    unet.load_state_dict(weights)
    unet = unet.to(device)
    unet.eval()
    
    # 使用包装器
    model = MuseTalkUNetWrapper(unet)
    model = model.to(device)
    model.eval()
    
    # 转为 float16（与原始 PyTorch 推理一致）
    model = model.half()
    
    print(f"[INFO] 模型加载耗时: {time.time() - start_time:.2f}s")
    
    # 设置输入尺寸
    batch_size = 1
    # MuseTalk: 输入图像 256x256，VAE 编码后为 32x32 (1/8)
    # UNet 处理的是 latent 空间，所以是 32x32
    height = 32
    width = 32
    audio_dim = 384  # Whisper 特征维度
    seq_len = 50  # 音频特征时间步数
    
    # 创建示例输入（使用 float16，放在 GPU 上）
    dummy_latents = torch.randn(batch_size, 8, height, width, dtype=torch.float16, device=device)
    dummy_audio = torch.randn(batch_size, seq_len, audio_dim, dtype=torch.float16, device=device)
    dummy_timestep = torch.tensor([0], device=device)
    
    print(f"[INFO] 输入尺寸: latents={dummy_latents.shape}, timestep={dummy_timestep.shape}, audio={dummy_audio.shape}")
    
    # 预热 - 确保模型在 GPU 上运行正常
    print(f"[INFO] 预热模型...")
    with torch.no_grad():
        _ = model(dummy_latents, dummy_timestep, dummy_audio)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    print(f"[INFO] 预热完成")
    
    # 导出 ONNX
    ensure_dir(output_path)
    print(f"[INFO] 导出 ONNX 到: {output_path}")
    
    export_start = time.time()
    
    with torch.no_grad():
        torch.set_grad_enabled(False)
        torch.onnx.export(
            model,
            (dummy_latents, dummy_timestep, dummy_audio),
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['latents', 'timestep', 'audio_embedding'],
            output_names=['noise_pred'],
            dynamic_axes={
                'latents': {0: 'batch_size'},
                'timestep': {0: 'batch_size'},
                'audio_embedding': {0: 'batch_size'},
                'noise_pred': {0: 'batch_size'}
            },
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
            verbose=False
        )
    
    export_time = time.time() - export_start
    total_time = time.time() - start_time
    
    print(f"[SUCCESS] MuseTalk ONNX 模型已保存: {output_path}")
    print(f"[INFO] ONNX 导出耗时: {export_time:.2f}s")
    print(f"[INFO] 总耗时: {total_time:.2f}s")
    
    # 清理 GPU 缓存
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return output_path


def convert_ultralight(input_path, output_path, opset_version=11):
    """
    转换 UltraLight 模型到 ONNX
    
    输入:
        - audio_features: [B, 1, 80, 16] (mel spectrogram)
        - face_features: [B, 3, H, W] (reference face frame)
    输出:
        - output: [B, 3, H, W] (generated face)
    """
    from avatars.ultralight.unet import UNet
    
    print(f"[INFO] 加载 UltraLight 模型: {input_path}")
    
    # 创建模型 (UltraLight 使用 96x96 或 128x128)
    model = UNet()
    
    # 加载权重
    checkpoint = torch.load(input_path, map_location='cpu')
    state_dict = checkpoint.get("state_dict", checkpoint)
    
    # 移除 'module.' 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace('module.', '')] = v
    
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    
    # 设置输入尺寸
    batch_size = 1
    mel_size = 80
    mel_time = 16
    img_size = 96  # UltraLight 使用 96x96 或 128x128
    
    # 创建示例输入
    dummy_audio = torch.randn(batch_size, 1, mel_size, mel_time)
    dummy_face = torch.randn(batch_size, 3, img_size, img_size)
    
    print(f"[INFO] 输入尺寸: audio={dummy_audio.shape}, face={dummy_face.shape}")
    
    # 导出 ONNX
    ensure_dir(output_path)
    print(f"[INFO] 导出 ONNX 到: {output_path}")
    
    torch.onnx.export(
        model,
        (dummy_audio, dummy_face),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['audio_features', 'face_features'],
        output_names=['output'],
        dynamic_axes={
            'audio_features': {0: 'batch_size'},
            'face_features': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"[SUCCESS] UltraLight ONNX 模型已保存: {output_path}")
    return output_path


class VAEDecoderWrapper(torch.nn.Module):
    """
    VAE 解码器包装器，用于 ONNX 导出
    只导出 decode 部分，用于将 latent 解码为图像
    """
    def __init__(self, vae_model):
        super().__init__()
        self.vae = vae_model
        self.scaling_factor = vae_model.config.scaling_factor
    
    def forward(self, latents):
        # latents: [B, 4, H, W] -> 先除以 scaling_factor
        latents = latents / self.scaling_factor
        # 解码
        image = self.vae.decode(latents).sample
        # 归一化到 [0, 1]
        image = (image / 2 + 0.5).clamp(0, 1)
        return image


def convert_musetalk_vae(model_path, output_path, opset_version=14):
    """
    转换 MuseTalk VAE 解码器到 ONNX
    
    输入:
        - latents: [B, 4, 32, 32] (VAE latent，已乘以 scaling_factor)
    输出:
        - image: [B, 3, 256, 256] (解码后的图像，范围 [0, 1])
    
    注意: VAE 使用 diffusers 的 AutoencoderKL，需要 opset_version >= 14
    优化: 使用 torch.jit.trace 预编译加速导出
    """
    from diffusers import AutoencoderKL
    import time
    
    print(f"[INFO] 加载 MuseTalk VAE 模型: {model_path}")
    start_time = time.time()
    
    # 加载 VAE 模型到 GPU（加速导出）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] 使用设备: {device}")
    
    vae = AutoencoderKL.from_pretrained(model_path)
    vae = vae.to(device)
    vae.eval()
    
    # 使用包装器（包含 scaling_factor 处理）
    model = VAEDecoderWrapper(vae)
    model = model.to(device)
    model.eval()
    
    # 转为 float16（与原始 PyTorch 推理一致）
    model = model.half()
    
    # 设置输入尺寸
    batch_size = 1
    height = 32
    width = 32
    
    # 创建示例输入（使用 float16，放在 GPU 上）
    dummy_latents = torch.randn(batch_size, 4, height, width, dtype=torch.float16, device=device)
    
    print(f"[INFO] 输入尺寸: latents={dummy_latents.shape}")
    print(f"[INFO] 模型加载耗时: {time.time() - start_time:.2f}s")
    
    # 预热 - 确保模型在 GPU 上运行正常
    print(f"[INFO] 预热模型...")
    with torch.no_grad():
        _ = model(dummy_latents)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    print(f"[INFO] 预热完成")
    
    # 导出 ONNX
    ensure_dir(output_path)
    print(f"[INFO] 导出 ONNX 到: {output_path}")
    
    export_start = time.time()
    
    # 使用 torch.no_grad() 和优化选项加速导出
    with torch.no_grad():
        # 设置模型为 eval 模式并禁用梯度计算
        torch.set_grad_enabled(False)
        
        torch.onnx.export(
            model,
            dummy_latents,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['latents'],
            output_names=['image'],
            dynamic_axes={
                'latents': {0: 'batch_size'},
                'image': {0: 'batch_size'}
            },
            # 优化选项
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
            # 禁用详细日志以减少 I/O 开销
            verbose=False
        )
    
    export_time = time.time() - export_start
    total_time = time.time() - start_time
    
    print(f"[SUCCESS] MuseTalk VAE ONNX 模型已保存: {output_path}")
    print(f"[INFO] ONNX 导出耗时: {export_time:.2f}s")
    print(f"[INFO] 总耗时: {total_time:.2f}s")
    
    # 清理 GPU 缓存
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return output_path


def verify_onnx(onnx_path):
    """验证导出的 ONNX 模型"""
    try:
        import onnx
        import onnxruntime as ort
        
        print(f"[INFO] 验证 ONNX 模型: {onnx_path}")
        
        # 检查模型
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("[SUCCESS] ONNX 模型检查通过")
        
        # 测试推理
        session = ort.InferenceSession(onnx_path)
        print(f"[INFO] 输入: {[inp.name for inp in session.get_inputs()]}")
        print(f"[INFO] 输出: {[out.name for out in session.get_outputs()]}")
        
        return True
    except ImportError:
        print("[WARNING] 未安装 onnx 或 onnxruntime，跳过验证")
        return True
    except Exception as e:
        print(f"[ERROR] ONNX 验证失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='将 PyTorch 模型转换为 ONNX 格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 转换 wav2lip 模型 (默认 opset 11)
  python convert_to_onnx.py --model wav2lip --input models/wav2lip/wav2lip.pth
  
  # 转换 musetalk 模型 (自动使用 opset 14)
  python convert_to_onnx.py --model musetalk --input models/musetalk/unet.pth
  
  # 转换 musetalk VAE 解码器 (自动使用 opset 14)
  python convert_to_onnx.py --model musetalk_vae --input models/musetalk/vae
  
  # 转换 ultralight 模型 (默认 opset 11)
  python convert_to_onnx.py --model ultralight --input models/ultralight/ultralight.pth
  
  # 指定 ONNX opset 版本
  python convert_to_onnx.py --model wav2lip --input models/wav2lip/wav2lip.pth --opset 12
  
  # 跳过验证
  python convert_to_onnx.py --model wav2lip --input models/wav2lip/wav2lip.pth --no-verify
  
注意:
  - musetalk 模型需要 opset 14+ 以支持 scaled_dot_product_attention 操作符
  - musetalk_vae 是 VAE 解码器，输入是 latent，输出是图像
  - 如果不指定 --opset，脚本会根据模型类型自动选择合适的版本
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        choices=['wav2lip', 'musetalk', 'musetalk_vae', 'ultralight'],
        help='模型类型: wav2lip, musetalk, musetalk_vae, ultralight'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='输入的 PyTorch 模型路径 (.pth 文件或 VAE 目录)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='输出的 ONNX 模型路径 (默认: models/onnx/{model_name}.onnx)'
    )
    
    parser.add_argument(
        '--opset',
        type=int,
        default=None,
        help='ONNX opset 版本 (默认: wav2lip/ultralight 使用 11, musetalk/musetalk_vae 使用 14)'
    )
    
    parser.add_argument(
        '--no-verify',
        action='store_true',
        help='跳过 ONNX 模型验证'
    )
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"[ERROR] 输入文件不存在: {args.input}")
        sys.exit(1)
    
    # 设置默认输出路径 (根据模型类型使用规范的目录结构)
    if args.output is None:
        if args.model == 'musetalk_vae':
            args.output = f"models/onnx/musetalk/vae_decoder.onnx"
        else:
            model_name = os.path.splitext(os.path.basename(args.input))[0]
            args.output = f"models/onnx/{args.model}/{model_name}.onnx"
    
    # 根据模型类型设置默认 opset 版本
    if args.opset is None:
        if args.model in ['musetalk', 'musetalk_vae']:
            args.opset = 14  # musetalk 需要 opset 14+ 支持 scaled_dot_product_attention
        else:
            args.opset = 11  # wav2lip 和 ultralight 使用 opset 11
    
    print(f"[INFO] 使用 ONNX opset 版本: {args.opset}")
    
    # 确保输出目录存在
    ensure_dir(args.output)
    
    # 执行转换
    try:
        if args.model == 'wav2lip':
            convert_wav2lip(args.input, args.output, args.opset)
        elif args.model == 'musetalk':
            convert_musetalk(args.input, args.output, args.opset)
        elif args.model == 'musetalk_vae':
            convert_musetalk_vae(args.input, args.output, args.opset)
        elif args.model == 'ultralight':
            convert_ultralight(args.input, args.output, args.opset)
        else:
            print(f"[ERROR] 不支持的模型类型: {args.model}")
            sys.exit(1)
        
        # 验证
        if not args.no_verify:
            verify_onnx(args.output)
        
        print(f"\n[SUCCESS] 转换完成!")
        print(f"  输入: {args.input}")
        print(f"  输出: {args.output}")
        
    except Exception as e:
        print(f"[ERROR] 转换失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
