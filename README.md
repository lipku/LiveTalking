# LiveTalking 数字人系统

实时交互式数字人系统，支持语音识别、TTS、LLM 对话和 WebRTC 流媒体。

## 系统要求

- **操作系统**: Ubuntu 20.04+ / CentOS 7+
- **GPU**: NVIDIA GPU with CUDA 11.8+
- **显存**: 至少 8GB (推荐 24GB)
- **Python**: 3.8 - 3.10
- **CUDA**: 11.8 或更高版本
- **cuDNN**: 8.6+

## 快速安装

### 1. 环境准备

```bash
# 创建虚拟环境
conda create -n livetalking python=3.10
conda activate livetalking

# 安装 PyTorch (CUDA 11.8)
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# 安装依赖
pip install -r requirements.txt
```

### 2. 模型下载

```bash
# 创建模型目录
mkdir -p models

# 下载必要的模型文件
# MuseTalk 模型
wget -O models/musetalk.pth https://your-model-url/musetalk.pth

# Wav2Lip 模型
wget -O models/wav2lip.pth https://your-model-url/wav2lip.pth

# 其他必要模型...
```

### 3. 配置设置

```bash
# 设置豆包 TTS 配置（已内置在代码中）
# 目前限制 20000 字符，10并发
# APP ID: 7082366049
# Access Token: 1fE0k8y_gCudCL8b9CLK4YXaFANOWrcH

# 如需修改，编辑 ttsreal.py 中的 DoubaoTTS 类
```

## 运行指南

### 启动主服务

```bash
# 基础启动
python app.py

# 指定参数启动
python app.py \
    --tts doubao \
    --model musetalk \
    --transport webrtc \
    --listenport 8010
```

### 启动 MCP 服务器

```bash
# 在新终端启动 MCP 服务
python mcp_server.py
```

### 测试 MCP 接口

```bash
# 运行完整测试
python test_mcp.py

# 交互式测试
python test_mcp.py --mode interactive

# 性能测试
python test_mcp.py --mode performance
```

## 访问界面

- **主界面**: http://localhost:8010/dashboard.html
- **增强版界面**: http://localhost:8010/dashboard_enhanced.html
- **MCP 测试页**: http://localhost:8011/

## 主要功能

### 1. TTS 服务
- 支持多种 TTS: EdgeTTS, 豆包, 腾讯云, CosyVoice 等
- 流式音频输出
- 低延迟实时合成

### 2. MCP 接口
- `POST /api/speak` - 让数字人说话
- `POST /api/interrupt` - 打断数字人
- `GET /api/status` - 获取说话状态

### 3. GPU 监控
- 实时 GPU 使用率
- 显存占用监控
- 温度和功耗显示
- 历史曲线图表

### 4. WebRTC 支持
- 低延迟视频流
- 双向音频通信
- 自适应码率

## 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--tts` | doubao | TTS 服务类型 |
| `--model` | musetalk | 数字人模型 |
| `--transport` | webrtc | 传输协议 |
| `--listenport` | 8010 | Web 服务端口 |
| `--fps` | 50 | 音频帧率 |
| `--batch_size` | 16 | 推理批大小 |
| `--REF_FILE` | zh-CN-XiaoxiaoNeural | 语音类型 |

## 故障排除

### CUDA 未检测到
```bash
# 检查 CUDA 安装
nvcc --version
nvidia-smi

# 重新安装 PyTorch
pip uninstall torch torchvision torchaudio
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

### 端口被占用
```bash
# 查看端口占用
lsof -i:8010
lsof -i:8011

# 终止占用进程
kill -9 <PID>
```

### 模型加载失败
```bash
# 检查模型文件
ls -la models/

# 重新下载模型
wget -O models/[model_name].pth [model_url]
```

## 性能优化

### GPU 优化
```bash
# 设置 CUDA 缓存
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"

# 启用混合精度
python app.py --mixed_precision
```

### 内存优化
```bash
# 限制 GPU 内存增长
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 清理缓存
python -c "import torch; torch.cuda.empty_cache()"
```

## 开发调试

### 日志级别
```bash
# 设置日志级别
export LOG_LEVEL=DEBUG
python app.py
```

### 测试模式
```bash
# 启用测试模式（使用模拟数据）
python app.py --test_mode
```

## 项目结构

```
LiveTalking/
├── app.py              # 主服务入口
├── mcp_server.py       # MCP 接口服务
├── test_mcp.py         # MCP 测试工具
├── ttsreal.py          # TTS 实现
├── basereal.py         # 基础数字人类
├── gpu_monitor.py      # GPU 监控模块
├── models/             # 模型文件目录
├── web/                # Web 前端文件
│   ├── dashboard_enhanced.html  # 增强版界面
│   └── ...
├── llm/                # LLM 集成
│   └── one_api.py      # OneAPI 接口
└── requirements.txt    # 依赖列表
```

## 更新日志

查看 [CHANGELOG.md](CHANGELOG.md) 了解详细改动。

## 许可证

Apache License 2.0

## 支持

如有问题，请联系 waple0820@gmail.com