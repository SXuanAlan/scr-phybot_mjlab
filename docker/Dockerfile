FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

# System dependencies (MuJoCo rendering + general dev tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget ca-certificates \
    libgl1 libglib2.0-0 libglew-dev libglfw3 libosmesa6-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/miniconda && \
    rm /tmp/miniconda.sh
ENV PATH="/opt/miniconda/bin:$PATH"

# Accept conda ToS and create env
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -n env_mjlab python=3.11 -y

# Install PyTorch (CUDA 12.8)
RUN conda run -n env_mjlab pip install --no-cache-dir \
    torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128

# Copy and install pip requirements (excluding torch and editable installs)
COPY requirements.txt /tmp/requirements.txt
RUN grep -v "^torch==\|^torchvision==\|^-e " /tmp/requirements.txt > /tmp/reqs_filtered.txt && \
    conda run -n env_mjlab pip install --no-cache-dir -r /tmp/reqs_filtered.txt || true

# Install Claude Code (do this before switching shell)
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g @anthropic-ai/claude-code

# Reuse existing ubuntu user (UID=1000, GID=1000) and add to video group
RUN usermod -aG video ubuntu && \
    chown -R ubuntu:ubuntu /opt/miniconda

# Switch to non-root user
USER ubuntu
RUN conda init bash && \
    echo "conda activate env_mjlab" >> ~/.bashrc

WORKDIR /workspace
CMD ["bash", "-l"]
