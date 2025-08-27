FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    cmake \
    ffmpeg \
    build-essential \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libcairo2-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

# Accept conda Terms of Service for Anaconda channels
RUN conda config --set always_yes yes && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda config --set solver libmamba || true

# Create conda environment
RUN conda create --name svgrender python=3.10 --yes && \
    conda clean -afy

# Set up environment variables for conda environment
ENV CONDA_DEFAULT_ENV=svgrender
ENV CONDA_PREFIX=/opt/conda/envs/svgrender
ENV PATH=${CONDA_PREFIX}/bin:${PATH}

# Upgrade pip first
RUN /opt/conda/envs/svgrender/bin/pip install --upgrade pip setuptools wheel

# Install PyTorch 1.12.1 with CUDA 11.3 via pip (more reliable than conda for specific versions)
RUN /opt/conda/envs/svgrender/bin/pip install --no-cache-dir \
    torch==1.12.1+cu113 \
    torchvision==0.13.1+cu113 \
    torchaudio==0.12.1 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# Install xformers compatible with PyTorch 1.12.1
RUN /opt/conda/envs/svgrender/bin/pip install --no-cache-dir xformers==0.0.13

# Install conda build tools
RUN conda install -n svgrender -y -c anaconda cmake && \
    conda install -n svgrender -y -c conda-forge ffmpeg && \
    conda clean -afy

# Install core dependencies first
RUN /opt/conda/envs/svgrender/bin/pip install --no-cache-dir \
    numpy scipy scikit-fmm scikit-image scikit-learn && \
    rm -rf ~/.cache/pip

# Install compute libraries
RUN /opt/conda/envs/svgrender/bin/pip install --no-cache-dir \
    numba triton==2.0.0 && \
    rm -rf ~/.cache/pip

# Install config and utility libraries
RUN /opt/conda/envs/svgrender/bin/pip install --no-cache-dir \
    hydra-core omegaconf easydict \
    ftfy regex tqdm && \
    rm -rf ~/.cache/pip

# Install graphics and SVG libraries
RUN /opt/conda/envs/svgrender/bin/pip install --no-cache-dir \
    freetype-py shapely svgutils svgwrite \
    svgpathtools cssutils cairosvg \
    opencv-python && \
    rm -rf ~/.cache/pip

# Install visualization and logging
RUN /opt/conda/envs/svgrender/bin/pip install --no-cache-dir \
    matplotlib visdom wandb BeautifulSoup4 && \
    rm -rf ~/.cache/pip

# Install deep learning libraries (pinning versions to avoid conflicts)
RUN /opt/conda/envs/svgrender/bin/pip install --no-cache-dir \
    einops==0.6.1 \
    timm==0.9.2 \
    fairscale==0.4.13 \
    pytorch-lightning==2.1.0 \
    torchmetrics==0.11.4 \
    webdataset && \
    rm -rf ~/.cache/pip

# Install torch-tools separately
RUN /opt/conda/envs/svgrender/bin/pip install --no-cache-dir torch-tools && \
    rm -rf ~/.cache/pip

# Install transformers ecosystem
RUN /opt/conda/envs/svgrender/bin/pip install --no-cache-dir \
    transformers==4.30.2 \
    accelerate==0.20.3 \
    safetensors \
    datasets==2.14.0 && \
    rm -rf ~/.cache/pip

# Install diffusers
RUN /opt/conda/envs/svgrender/bin/pip install --no-cache-dir diffusers==0.20.2 && \
    rm -rf ~/.cache/pip

# Install CLIP
RUN /opt/conda/envs/svgrender/bin/pip install --no-cache-dir git+https://github.com/openai/CLIP.git && \
    rm -rf ~/.cache/pip

# Clone and install DiffVG
WORKDIR /tmp
RUN git clone https://github.com/BachiLi/diffvg.git && \
    cd diffvg && \
    git submodule update --init --recursive && \
    /opt/conda/envs/svgrender/bin/python setup.py install && \
    cd .. && \
    rm -rf diffvg && \
    rm -rf ~/.cache/pip

# Set working directory
WORKDIR /workspace

# Copy the SVGDreamer code from your repository
COPY . /workspace/

# Create necessary directories
RUN mkdir -p logs checkpoint outputs

# Set Python path
ENV PYTHONPATH=/workspace

# Create entrypoint script
RUN echo '#!/bin/bash\n\
source /opt/conda/etc/profile.d/conda.sh\n\
conda activate svgrender\n\
exec "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]