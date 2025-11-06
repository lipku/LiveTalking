# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

ARG BASE_IMAGE=nvcr.io/nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04
FROM $BASE_IMAGE

RUN apt-get update -yq --fix-missing \
 && DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
    pkg-config \
    wget \
    cmake \
    curl \
    git \
    vim

#ENV PYTHONDONTWRITEBYTECODE=1
#ENV PYTHONUNBUFFERED=1

# nvidia-container-runtime
#ENV NVIDIA_VISIBLE_DEVICES all
#ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    sh Miniconda3-latest-Linux-x86_64.sh -b -u -p ~/miniconda3 && \
    rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH="/root/miniconda3/bin:${PATH}"

# Accept conda terms of service
RUN conda config --set channel_priority flexible && \
    conda config --set always_yes yes && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -n nerfstream python=3.10 && \
    conda clean -afy

# Activate conda environment and install dependencies
SHELL ["/bin/bash", "-c"]

RUN source /root/miniconda3/etc/profile.d/conda.sh && \
    conda activate nerfstream && \
    pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch -y && \
    conda clean -afy

COPY requirements.txt ./
RUN source /root/miniconda3/etc/profile.d/conda.sh && \
    conda activate nerfstream && \
    pip install -r requirements.txt

# additional libraries
# RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git"
# RUN pip install tensorflow-gpu==2.8.0

# RUN pip uninstall protobuf
# RUN pip install protobuf==3.20.1

# RUN conda install ffmpeg
# Copy ../python_rtmpstream /python_rtmpstream
# WORKDIR /python_rtmpstream/python
# RUN pip install .

# Copy ../nerfstream /nerfstream
WORKDIR /nerfstream
CMD ["python3", "app.py"]
