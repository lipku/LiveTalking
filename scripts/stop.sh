#!/bin/bash
# Docker 容器停止脚本
# 用途：停止并删除 LiveTalking 容器

set -e

CONTAINER_NAME="livetalking"

echo "停止 LiveTalking 容器..."

# 检查容器是否存在
if [ "$(sudo docker ps -aq -f name=${CONTAINER_NAME})" ]; then
    # 停止容器（如果正在运行）
    if [ "$(sudo docker ps -q -f name=${CONTAINER_NAME})" ]; then
        echo "停止容器 ${CONTAINER_NAME}..."
        sudo docker stop ${CONTAINER_NAME}
    fi
    
    # 删除容器
    echo "删除容器 ${CONTAINER_NAME}..."
    sudo docker rm ${CONTAINER_NAME}
    
    echo "✅ 容器已停止并删除"
else
    echo "⚠️  容器 ${CONTAINER_NAME} 不存在"
fi

