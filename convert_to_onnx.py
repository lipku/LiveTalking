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


def convert_musetalk(input_path, output_path, opset_version=11):
    """
    转换 MuseTalk 模型到 ONNX
    
    注意: MuseTalk 使用 UNet 架构，需要根据其具体实现调整
    """
    print(f"[INFO] MuseTalk 转换需要具体模型结构信息")
    print(f"[INFO] 输入路径: {input_path}")
    print(f"[INFO] 请确认 models/musetalk/unet.py 中的模型定义")
    
    # TODO: 根据实际的 MuseTalk UNet 实现补充转换逻辑
    raise NotImplementedError("MuseTalk 转换需要具体模型结构，请先检查 avatars/musetalk/ 目录")


def convert_ultralight(input_path, output_path, opset_version=11):
    """
    转换 UltraLight 模型到 ONNX
    """
    print(f"[INFO] UltraLight 转换需要具体模型结构信息")
    print(f"[INFO] 输入路径: {input_path}")
    print(f"[INFO] 请确认 avatars/ultralight/unet.py 中的模型定义")
    
    # TODO: 根据实际的 UltraLight 实现补充转换逻辑
    raise NotImplementedError("UltraLight 转换需要具体模型结构，请先检查 avatars/ultralight/ 目录")


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
  # 转换 wav2lip 模型
  python convert_to_onnx.py --model wav2lip --input models/wav2lip.pth --output models/onnx/wav2lip.onnx
  
  # 指定 ONNX opset 版本
  python convert_to_onnx.py --model wav2lip --input models/wav2lip.pth --output models/onnx/wav2lip.onnx --opset 12
  
  # 跳过验证
  python convert_to_onnx.py --model wav2lip --input models/wav2lip.pth --output models/onnx/wav2lip.onnx --no-verify
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        choices=['wav2lip', 'musetalk', 'ultralight'],
        help='模型类型: wav2lip, musetalk, ultralight'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='输入的 PyTorch 模型路径 (.pth 文件)'
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
        default=11,
        help='ONNX opset 版本 (默认: 11)'
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
    
    # 设置默认输出路径
    if args.output is None:
        model_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"models/onnx/{model_name}.onnx"
    
    # 确保输出目录存在
    ensure_dir(args.output)
    
    # 执行转换
    try:
        if args.model == 'wav2lip':
            convert_wav2lip(args.input, args.output, args.opset)
        elif args.model == 'musetalk':
            convert_musetalk(args.input, args.output, args.opset)
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
