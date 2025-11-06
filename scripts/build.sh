#!/bin/bash
# Docker 镜像构建脚本
# 用途：构建 LiveTalking 项目的 Docker 镜像

set -e

IMAGE_NAME="lightmoutain-digital"
IMAGE_TAG="latest"

echo "开始构建 Docker 镜像..."
echo "镜像名称: ${IMAGE_NAME}:${IMAGE_TAG}"

cd "$(dirname "$0")/.."

sudo docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

echo ""
echo "✅ 镜像构建成功！"
echo ""
echo "镜像信息:"
sudo docker images | grep ${IMAGE_NAME}

