#!/bin/bash
# Docker 容器交互式 Bash 脚本
# 用途：启动容器并进入 bash 环境，方便调试

set -e

IMAGE_NAME="lightmoutain-digital:latest"
CONTAINER_NAME="livetalking-dev"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "启动开发容器（Bash 模式）..."
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
echo ""
echo "💡 提示: 在容器内手动启动应用："
echo "   source /root/miniconda3/etc/profile.d/conda.sh"
echo "   conda activate nerfstream"
echo "   python3 app.py"
echo ""

sudo docker run -it --rm \
  --name ${CONTAINER_NAME} \
  --gpus all \
  --net host \
  -v ${PROJECT_DIR}:/nerfstream \
  ${IMAGE_NAME} \
  /bin/bash

echo ""
echo "容器已退出"

