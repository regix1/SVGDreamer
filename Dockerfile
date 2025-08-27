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

# Activate environment for all subsequent commands
SHELL ["conda", "run", "-n", "svgrender", "/bin/bash", "-c"]

# Install PyTorch and related libraries
RUN conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch --yes && \
    conda clean -afy

# Install xformers
RUN conda install xformers -c xformers --yes && \
    conda clean -afy

# Install build tools through conda
RUN conda install -y -c anaconda cmake && \
    conda install -y -c conda-forge ffmpeg && \
    conda clean -afy

# Install Python dependencies - core libraries in smaller chunks
# First, upgrade pip and install core build dependencies
RUN pip install --upgrade pip setuptools wheel

# Install numpy and scipy first as many packages depend on them
RUN pip install --no-cache-dir numpy scipy

# Install core ML/compute libraries
RUN pip install --no-cache-dir \
    scikit-fmm \
    scikit-image \
    scikit-learn \
    numba \
    triton

# Install config and utility libraries
RUN pip install --no-cache-dir \
    hydra-core \
    omegaconf \
    easydict \
    ftfy \
    regex \
    tqdm

# Install graphics and SVG libraries
RUN pip install --no-cache-dir \
    freetype-py \
    shapely \
    svgutils \
    svgwrite \
    svgpathtools \
    cssutils \
    cairosvg \
    opencv-python

# Install visualization and logging
RUN pip install --no-cache-dir \
    matplotlib \
    visdom \
    wandb \
    BeautifulSoup4

# Install deep learning libraries
RUN pip install --no-cache-dir \
    einops \
    timm \
    fairscale==0.4.13 \
    pytorch_lightning==2.1.0 \
    webdataset \
    torch-tools

# Install transformers ecosystem
RUN pip install --no-cache-dir \
    accelerate \
    transformers \
    safetensors \
    datasets

# Install CLIP
RUN pip install --no-cache-dir git+https://github.com/openai/CLIP.git

# Install diffusers
RUN pip install --no-cache-dir diffusers==0.20.2

# Clone and install DiffVG
WORKDIR /tmp
RUN git clone https://github.com/BachiLi/diffvg.git && \
    cd diffvg && \
    git submodule update --init --recursive && \
    python setup.py install && \
    cd .. && \
    rm -rf diffvg

# Set working directory
WORKDIR /workspace

# Copy the SVGDreamer code from your repository
COPY . /workspace/

# Create necessary directories
RUN mkdir -p logs checkpoint outputs

# Set environment variables for runtime
ENV CONDA_DEFAULT_ENV=svgrender
ENV CONDA_PREFIX=/opt/conda/envs/svgrender
ENV PATH=${CONDA_PREFIX}/bin:${PATH}
ENV PYTHONPATH=/workspace

# Create entrypoint script
RUN echo '#!/bin/bash\n\
source /opt/conda/etc/profile.d/conda.sh\n\
conda activate svgrender\n\
exec "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]