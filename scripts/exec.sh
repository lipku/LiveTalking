#!/bin/bash
# Docker 容器进入脚本
# 用途：进入正在运行的 LiveTalking 容器

set -e

CONTAINER_NAME="livetalking"

echo "进入 ${CONTAINER_NAME} 容器..."

# 检查容器是否在运行
if [ "$(sudo docker ps -q -f name=${CONTAINER_NAME})" ]; then
    sudo docker exec -it ${CONTAINER_NAME} /bin/bash
else
    echo "⚠️  容器 ${CONTAINER_NAME} 未运行"
    echo ""
    echo "提示: 使用以下命令启动容器"
    echo "  ./scripts/run-daemon.sh  # 后台运行"
    echo "  ./scripts/run.sh         # 前台运行"
fi

