#!/bin/bash
# Docker 容器后台运行脚本
# 用途：以守护进程模式启动 LiveTalking 容器

set -e

IMAGE_NAME="lightmoutain-digital:latest"
CONTAINER_NAME="livetalking"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "以守护进程模式启动 LiveTalking 容器..."
echo "项目目录: ${PROJECT_DIR}"
echo ""

# 检查是否已有同名容器在运行
if [ "$(sudo docker ps -q -f name=${CONTAINER_NAME})" ]; then
    echo "⚠️  容器 ${CONTAINER_NAME} 已在运行"
    echo "容器状态:"
    sudo docker ps -f name=${CONTAINER_NAME}
    exit 1
fi

# 检查是否有停止的同名容器
if [ "$(sudo docker ps -aq -f name=${CONTAINER_NAME})" ]; then
    echo "删除已停止的容器..."
    sudo docker rm ${CONTAINER_NAME}
fi

echo "启动新容器..."
sudo docker run -d \
  --name ${CONTAINER_NAME} \
  --gpus all \
  --net host \
  --restart unless-stopped \
  -v ${PROJECT_DIR}:/nerfstream \
  -e PYTHONUNBUFFERED=1 \
  ${IMAGE_NAME}

echo ""
echo "✅ 容器已启动"
echo ""
echo "查看日志: sudo docker logs -f ${CONTAINER_NAME}"
echo "停止容器: sudo docker stop ${CONTAINER_NAME}"
echo "进入容器: sudo docker exec -it ${CONTAINER_NAME} /bin/bash"

