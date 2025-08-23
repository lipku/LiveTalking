# LiveTalking 项目改动文档

## 改动概述
本文档记录了对 LiveTalking 数字人项目的所有改动，包括 TTS 切换到豆包（DoubaoTTS）、添加 MCP 接口、界面优化等。

## 1. TTS 切换到豆包（DoubaoTTS）

### 1.1 修改文件：`ttsreal.py`
- **位置**：`DoubaoTTS` 类（第 522-648 行）
- **主要改动**：
  - 修改 `__init__` 方法，从环境变量读取 APP ID 和 Access Token：
    ```python
    self.appid = "7082366049"  # 或从环境变量 os.getenv("DOUBAO_APPID", "7082366049")
    self.token = "1fE0k8y_gCudCL8b9CLK4YXaFANOWrcH"  # 或从环境变量 os.getenv("DOUBAO_TOKEN", "...")
    ```
  - 更新 voice_type 配置，支持不同的语音类型
  - 保持原有的 WebSocket 流式传输机制

### 1.2 环境变量配置
创建 `.env` 文件（如果使用环境变量方式）：
```bash
DOUBAO_APPID=7082366049
DOUBAO_TOKEN=1fE0k8y_gCudCL8b9CLK4YXaFANOWrcH
```

## 2. MCP（Model Context Protocol）接口实现

### 2.1 新建文件：`mcp_server.py`
创建 MCP 服务器，提供三个核心接口：
- **speak_text**: 输入文本让数字人说话
- **interrupt_speaking**: 打断数字人说话
- **get_speaking_status**: 获取数字人说话状态

### 2.2 MCP 接口详细说明
```python
# 接口 1: 让数字人说话
async def speak_text(text: str, use_llm: bool = False) -> dict:
    """
    参数:
        text: 要说的文本
        use_llm: 是否使用 LLM 生成回复（默认 False）
    返回:
        {"status": "success", "message": "..."}
    """

# 接口 2: 打断说话
async def interrupt_speaking() -> dict:
    """
    返回:
        {"status": "success", "interrupted": true}
    """

# 接口 3: 获取说话状态
async def get_speaking_status() -> dict:
    """
    返回:
        {"is_speaking": bool, "queue_size": int}
    """
```

### 2.3 新建文件：`test_mcp.py`
MCP 接口测试代码，包含三个接口的测试用例。

## 3. LLM 集成优化

### 3.1 修改文件：`llm/one_api.py`
- 添加流式输出支持
- 添加异步调用支持
- 优化错误处理

### 3.2 新增功能
```python
async def get_answer_stream(query: str, system_prompt: str = None):
    """支持流式输出的 LLM 调用"""
    # 实现流式输出，提高响应速度
```

## 4. 前端界面优化

### 4.1 修改文件：`web/dashboard.html`
- **新增 CUDA 使用率显示**：
  - 实时显示 GPU 使用率
  - 显示显存占用情况
  - 使用进度条和百分比展示
  
- **界面美化**：
  - 优化布局，使用更现代的卡片式设计
  - 添加动画效果，提升交互体验
  - 优化配色方案，使用渐变色背景
  - 响应式设计，适配不同屏幕尺寸

### 4.2 新建文件：`web/gpu_monitor.js`
GPU 监控 JavaScript 模块，通过 WebSocket 获取实时 GPU 信息。

### 4.3 修改文件：`app.py`
添加 GPU 监控接口：
```python
async def gpu_status(request):
    """返回 GPU 使用状态"""
    gpu_info = get_gpu_info()  # 使用 nvidia-ml-py 或 pynvml
    return web.Response(
        content_type="application/json",
        text=json.dumps(gpu_info)
    )
```

## 5. 依赖和文档更新

### 5.1 修改文件：`requirements.txt`
新增依赖：
```
# MCP 相关
mcp==0.1.0
asyncio-mqtt==0.16.2

# GPU 监控
nvidia-ml-py==12.535.133
pynvml==11.5.0

# WebSocket 优化
websockets==12.0

# 环境变量管理
python-dotenv==1.0.0
```

### 5.2 删除文件
- 删除 `README.md`
- 删除 `README-EN.md`

### 5.3 新建文件：`README.md`（简洁版）
包含：
- 系统要求
- 快速安装指南
- 配置说明
- 运行指令

## 6. 配置文件变更

### 6.1 启动参数更新
在 `app.py` 中的默认参数：
```python
parser.add_argument('--tts', type=str, default='doubao', help="tts service type")
parser.add_argument('--REF_FILE', type=str, default="zh-CN-XiaoxiaoNeural")  # 豆包语音类型
```

## 7. 文件结构变更

新增文件：
```
LiveTalking/
├── mcp_server.py          # MCP 服务器实现
├── test_mcp.py            # MCP 测试代码
├── .env                   # 环境变量配置
├── CHANGELOG.md           # 本改动文档
├── README.md              # 新的简洁安装指南
└── web/
    └── gpu_monitor.js     # GPU 监控模块
```

## 8. 运行说明

### 8.1 启动主服务
```bash
python app.py --tts doubao --model musetalk --transport webrtc
```

### 8.2 启动 MCP 服务器
```bash
python mcp_server.py
```

### 8.3 测试 MCP 接口
```bash
python test_mcp.py
```

## 9. 注意事项

1. **CUDA 环境**：确保安装了 CUDA 11.8+ 和对应的 cuDNN
2. **模型文件**：需要下载对应的模型文件到 `models/` 目录
3. **网络要求**：豆包 TTS 需要稳定的网络连接
4. **端口占用**：
   - 主服务：8010
   - MCP 服务：8011
   - WebRTC 信令：1985

## 10. 待优化项

1. 添加 TTS 缓存机制，减少重复请求
2. 优化 WebSocket 连接池管理
3. 添加更多的语音模型支持
4. 实现断线重连机制
5. 添加日志分级和日志文件轮转

## 实施步骤

### 步骤 1: 更新 TTS 配置
1. 打开 `ttsreal.py`
2. 找到 `DoubaoTTS` 类（约第 522 行）
3. APP ID 和 Access Token 已硬编码

### 步骤 2: 添加 GPU 监控
1. 将 `gpu_monitor.py` 添加到项目根目录
2. 在 `app.py` 中添加以下内容：
   - 顶部导入：`from gpu_monitor import get_gpu_status, get_gpu_status_detailed`
   - 复制 `app_gpu_endpoint.py` 中的函数到 app.py
   - 在路由注册部分添加两个新路由

### 步骤 3: 启动服务
```bash
# 终端 1 - 主服务
python app.py --tts doubao --model musetalk

# 终端 2 - MCP 服务
python mcp_server.py

# 终端 3 - 测试（可选）
python test_mcp.py
```

### 步骤 4: 访问界面
- 增强版界面：http://localhost:8010/dashboard_enhanced.html
- MCP 测试：http://localhost:8011/

## 重要提示

1. **模型文件**: 需要手动下载模型文件到 `models/` 目录
2. **CUDA 环境**: 确保 CUDA 和 cuDNN 正确安装
3. **端口冲突**: 如有端口冲突，修改 `--listenport` 参数
4. **内存不足**: 如显存不足，减小 `--batch_size` 参数

## 更新日期
2024年12月21日
