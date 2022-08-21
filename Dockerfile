FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ARG NODEJS_VERSION=14.x
ARG GOLANG_VERSION=1.18.5
ARG GOOFYS_VERSION=0.24.0
ARG ML_INSTALL_CMD="python3 -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113"

ENV DEBIAN_FRONTEND=nointeractive \
    TZ=Asia/Tokyo \
    PATH=${PATH}:/usr/local/go/bin \
    GOPATH=/usr/local/go \
    GO111MODULE=on
ARG username=vscode
ARG useruid=1000
ARG usergid=${useruid}
ARG requirements_txt=pytorch.requirements.txt

RUN \
    # Install build modules
    sed -i 's@archive.ubuntu.com@ftp.jaist.ac.jp/pub/Linux@g' /etc/apt/sources.list && \
    # Update nvidia GPG key
    rm /etc/apt/sources.list.d/cuda.list && \
    # rm /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-key del 7fa2af80 && \
    apt-get update && apt-get install -y --no-install-recommends \
    wget \
    software-properties-common && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    add-apt-repository ppa:git-core/ppa && \
    apt-get update && \
    apt-get upgrade -y && \
    # apt-get upgrade -y && \
    apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    tar \
    unzip \
    python3 \
    python3-pip \
    s3fs \
    fuse-emulator-utils \
    fuse \
    dirmngr \
    apt-transport-https \
    lsb-release \
    ca-certificates \
    zlib1g-dev \
    libjpeg-dev \
    graphviz \
    sudo && \
    # ln /bin/python3 /bin/python \
    #
    # Create non-root user: https://aka.ms/vscode-remote/containers/non-root-user
    groupadd --gid ${usergid} ${username} && \
    useradd -s /bin/bash --uid ${useruid} --gid ${usergid} -m ${username} && \
    echo ${username} ALL=\(root\) NOPASWD:ALL > /etc/sudoers.d/${username} && \
    chmod 0440 /etc/sudoers.d/${username} && \
    #
    # Install aws-cli
    curl -q "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip -q awscliv2.zip && \
    ./aws/install && \
    #
    # Install nodejs
    curl -q -sL https://deb.nodesource.com/setup_${NODEJS_VERSION} | bash - && \
    apt-get install -y nodejs && \
    #
    # Install golang
    wget https://go.dev/dl/go${GOLANG_VERSION}.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go${GOLANG_VERSION}.linux-amd64.tar.gz && \
    #
    # Install goofys from binary
    wget -q https://github.com/kahing/goofys/releases/download/v${GOOFYS_VERSION}/goofys -P /usr/local/bin/ && \
    chmod +x /usr/local/bin/goofys && \
    apt-get clean

COPY requirements/${requirements_txt} .

RUN \
    --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -r ${requirements_txt} \
    --extra-index-url https://download.pytorch.org/whl/cu113 \
    && \
    # Setting jupyter lab configurations
    python3 -m jupyter lab build --dev-build=False && \
    python3 -m bash_kernel.install && \
    echo ""
