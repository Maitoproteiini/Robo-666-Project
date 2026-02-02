FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    cmake \
    pkg-config \
    ffmpeg \
    xvfb \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    mesa-utils \
    libosmesa6-dev \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Python
RUN python3 -m pip install --upgrade pip wheel setuptools

# OmniSafe, MuJoCo and Safety Gymnasium installation
RUN pip install \
    numpy==1.23.0 \
    mujoco==2.3.3 \
    gymnasium \
    safety-gymnasium \
    omnisafe \
    wandb weave

# Other useful Python packages
RUN pip install \
    jupyter \
    matplotlib \
    seaborn \
    tensorboard \
    ipykernel

WORKDIR /workspace
