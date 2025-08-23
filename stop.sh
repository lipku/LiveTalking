#!/bin/bash
# LiveTalking 停止脚本

echo "========================================="
echo "    停止 LiveTalking 服务"
echo "========================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 停止主服务
if [ -f .app.pid ]; then
    APP_PID=$(cat .app.pid)
    if kill -0 $APP_PID 2>/dev/null; then
        echo -e "${YELLOW}停止主服务 (PID: $APP_PID)...${NC}"
        kill $APP_PID
        sleep 1
        if kill -0 $APP_PID 2>/dev/null; then
            echo -e "${YELLOW}强制停止主服务...${NC}"
            kill -9 $APP_PID
        fi
        echo -e "${GREEN}✓ 主服务已停止${NC}"
    else
        echo -e "${YELLOW}主服务未运行${NC}"
    fi
    rm -f .app.pid
else
    echo -e "${YELLOW}未找到主服务 PID 文件${NC}"
fi

# 停止 MCP 服务
if [ -f .mcp.pid ]; then
    MCP_PID=$(cat .mcp.pid)
    if kill -0 $MCP_PID 2>/dev/null; then
        echo -e "${YELLOW}停止 MCP 服务 (PID: $MCP_PID)...${NC}"
        kill $MCP_PID
        sleep 1
        if kill -0 $MCP_PID 2>/dev/null; then
            echo -e "${YELLOW}强制停止 MCP 服务...${NC}"
            kill -9 $MCP_PID
        fi
        echo -e "${GREEN}✓ MCP 服务已停止${NC}"
    else
        echo -e "${YELLOW}MCP 服务未运行${NC}"
    fi
    rm -f .mcp.pid
else
    echo -e "${YELLOW}未找到 MCP 服务 PID 文件${NC}"
fi

# 检查是否还有遗留进程
echo -e "${YELLOW}检查遗留进程...${NC}"

# 查找并停止可能的遗留进程
for proc in "app.py" "mcp_server.py"; do
    PIDS=$(ps aux | grep python | grep $proc | grep -v grep | awk '{print $2}')
    if [ ! -z "$PIDS" ]; then
        echo -e "${YELLOW}发现遗留进程 $proc:${NC}"
        for PID in $PIDS; do
            echo "  停止 PID: $PID"
            kill $PID 2>/dev/null
        done
    fi
done

echo ""
echo -e "${GREEN}✓ 所有服务已停止${NC}"
echo "========================================="
