#!/bin/bash
# LiveTalking 快速启动脚本

echo "========================================="
echo "    LiveTalking 数字人系统启动脚本"
echo "========================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查 Python 环境
echo -e "${YELLOW}[1/5] 检查 Python 环境...${NC}"
if ! command -v python &> /dev/null; then
    echo -e "${RED}❌ Python 未安装${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python 已安装: $(python --version)${NC}"

# 检查 CUDA
echo -e "${YELLOW}[2/5] 检查 CUDA 环境...${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ CUDA 已安装${NC}"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo -e "${YELLOW}⚠️  CUDA 未检测到，将使用 CPU 模式${NC}"
fi

# 检查必要文件
echo -e "${YELLOW}[3/5] 检查必要文件...${NC}"
if [ ! -f "app.py" ]; then
    echo -e "${RED}❌ app.py 文件不存在${NC}"
    exit 1
fi
if [ ! -f "mcp_server.py" ]; then
    echo -e "${RED}❌ mcp_server.py 文件不存在${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 必要文件已就绪${NC}"

# 检查端口占用
echo -e "${YELLOW}[4/5] 检查端口...${NC}"
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${RED}❌ 端口 $port 已被占用${NC}"
        echo "   请使用以下命令查看占用进程："
        echo "   lsof -i:$port"
        return 1
    else
        echo -e "${GREEN}✓ 端口 $port 可用${NC}"
        return 0
    fi
}

PORT_8010_OK=$(check_port 8010 && echo "true" || echo "false")
PORT_8011_OK=$(check_port 8011 && echo "true" || echo "false")

if [ "$PORT_8010_OK" = "false" ] || [ "$PORT_8011_OK" = "false" ]; then
    echo -e "${YELLOW}提示: 可以修改端口后重试${NC}"
    read -p "是否继续？(y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 启动服务
echo -e "${YELLOW}[5/5] 启动服务...${NC}"

# 创建日志目录
mkdir -p logs

# 启动主服务
echo -e "${GREEN}启动主服务...${NC}"
nohup python app.py \
    --tts doubao \
    --model musetalk \
    --transport webrtc \
    --listenport 8010 \
    > logs/app.log 2>&1 &
APP_PID=$!
echo "主服务 PID: $APP_PID"

# 等待主服务启动
sleep 3

# 启动 MCP 服务
echo -e "${GREEN}启动 MCP 服务...${NC}"
nohup python mcp_server.py > logs/mcp.log 2>&1 &
MCP_PID=$!
echo "MCP 服务 PID: $MCP_PID"

# 保存 PID 到文件
echo $APP_PID > .app.pid
echo $MCP_PID > .mcp.pid

# 等待服务完全启动
sleep 2

# 显示访问信息
echo ""
echo "========================================="
echo -e "${GREEN}✨ 服务启动成功！${NC}"
echo "========================================="
echo ""
echo "访问地址:"
echo "  📺 增强版界面: http://localhost:8010/dashboard_enhanced.html"
echo "  🎮 标准界面:   http://localhost:8010/dashboard.html"
echo "  🔧 MCP 测试:   http://localhost:8011/"
echo ""
echo "日志文件:"
echo "  主服务日志: logs/app.log"
echo "  MCP 日志:   logs/mcp.log"
echo ""
echo "停止服务:"
echo "  ./stop.sh"
echo ""
echo "查看日志:"
echo "  tail -f logs/app.log"
echo "  tail -f logs/mcp.log"
echo ""
echo "========================================="
