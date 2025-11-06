#!/bin/bash
# Docker 容器日志查看脚本
# 用途：查看 LiveTalking 容器的实时日志

set -e

CONTAINER_NAME="livetalking"

echo "查看 ${CONTAINER_NAME} 容器日志..."
echo "按 Ctrl+C 退出"
echo ""

# 检查容器是否存在
if [ "$(sudo docker ps -q -f name=${CONTAINER_NAME})" ]; then
    sudo docker logs -f ${CONTAINER_NAME}
else
    echo "⚠️  容器 ${CONTAINER_NAME} 未运行"
    echo ""
    echo "提示: 使用以下命令启动容器"
    echo "  ./scripts/run-daemon.sh  # 后台运行"
    echo "  ./scripts/run.sh         # 前台运行"
fi

