#!/bin/bash
# Docker 容器启动脚本（前台运行，方便开发调试）
# 用途：启动 LiveTalking 容器，挂载本地代码目录

set -e

IMAGE_NAME="lightmoutain-digital:latest"
CONTAINER_NAME="livetalking"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "启动 LiveTalking 容器..."
echo "项目目录: ${PROJECT_DIR}"
echo ""

# 检查是否已有同名容器在运行
if [ "$(sudo docker ps -q -f name=${CONTAINER_NAME})" ]; then
    echo "⚠️  容器 ${CONTAINER_NAME} 已在运行"
    echo "停止现有容器..."
    sudo docker stop ${CONTAINER_NAME}
    sudo docker rm ${CONTAINER_NAME}
fi

# 检查是否有停止的同名容器
if [ "$(sudo docker ps -aq -f name=${CONTAINER_NAME})" ]; then
    echo "删除已停止的容器..."
    sudo docker rm ${CONTAINER_NAME}
fi

echo "启动新容器..."
sudo docker run -it --rm \
  --name ${CONTAINER_NAME} \
  --gpus all \
  --net host \
  -v ${PROJECT_DIR}:/nerfstream \
  ${IMAGE_NAME}

echo ""
echo "容器已停止"

