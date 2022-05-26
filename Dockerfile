FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install \
    -y \
    --no-install-recommends \
    apt-utils \
    binutils \
    build-essential \
    ca-certificates \
    curl \
    gcc \
    git \
    htop \
    less \
    libaio-dev \
    libxext6 \
    libx11-6 \
    libglib2.0-0 \
    libxrender1 \
    libxtst6 \
    libxi6 \
    locales \
    nano \
    ninja-build \
    python3-dev \
    python3-setuptools \
    python3-venv \
    python3-pip \
    screen \
    ssh \
    sudo \
    tmux \
    unzip \
    vim \
    wget

RUN python3 -m \
    pip install \
    --no-cache-dir \
    --upgrade \
    pip

RUN python3 -m \
    pip install \
    --no-cache-dir \
    torch==1.11.0 \
    --extra-index-url https://download.pytorch.org/whl/cu113

RUN python3 -m \
    pip install \
    --no-cache-dir \
    deepspeed==0.6.3 \ 
    einops==0.4.0 \
    load-confounds==0.12.0 \
    matplotlib==3.5.1 \
    nibabel==3.2.2 \
    nilearn==0.9.0 \
    numpy==1.22.0 \
    pandas==1.4.1 \
    Pillow==9.0.1 \
    pytest==7.0.1 \
    seaborn==0.11.2 \
    templateflow==0.7.1 \
    torchviz==0.0.2 \
    transformers==4.16.2 \
    tqdm==4.62.3 \
    wandb==0.12.11 \
    webdataset==0.1.103

CMD /bin/bash