FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget git cmake ffmpeg build-essential \
    libjpeg-dev libpng-dev libtiff-dev \
    libcairo2-dev libgl1-mesa-dev \
    curl unzip python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

# Create conda environment with Python 3.10
RUN conda create -n svgdreamer python=3.10 -y && \
    conda install -n svgdreamer -c conda-forge \
        numpy=1.24.3 \
        scipy \
        scikit-image \
        pillow \
        matplotlib \
        pybind11 \
        libstdcxx-ng -y && \
    conda clean -afy

# Set conda environment
SHELL ["conda", "run", "-n", "svgdreamer", "/bin/bash", "-c"]
ENV CONDA_DEFAULT_ENV=svgdreamer
ENV CONDA_PREFIX=/opt/conda/envs/svgdreamer
ENV PATH=${CONDA_PREFIX}/bin:${PATH}
ENV CONDA_PYTHON_EXE=${CONDA_PREFIX}/bin/python

# Install PyTorch with CUDA 11.3
RUN pip install --no-cache-dir \
    torch==1.12.1+cu113 \
    torchvision==0.13.1+cu113 \
    torchaudio==0.12.1+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# Install xformers for PyTorch 1.12.1
RUN pip install --no-cache-dir xformers==0.0.13

# Install core dependencies
RUN pip install --no-cache-dir \
    hydra-core==1.3.2 \
    omegaconf==2.3.0 \
    einops==0.6.1 \
    timm==0.9.2 \
    fairscale==0.4.13 \
    pytorch-lightning==2.1.0 \
    torchmetrics==0.11.4 \
    scikit-fmm \
    scikit-learn \
    numba \
    triton==2.0.0

# Install SVG and graphics libraries
RUN pip install --no-cache-dir \
    svgwrite \
    svgpathtools \
    svgutils \
    cairosvg \
    cssutils \
    freetype-py \
    shapely \
    opencv-python==4.8.1.78

# Install ML packages with version compatibility
RUN pip install --no-cache-dir \
    diffusers==0.20.2 \
    transformers==4.30.2 \
    accelerate==0.20.3 \
    safetensors \
    tokenizers==0.13.3 \
    huggingface-hub==0.16.4

# Install datasets and dependencies
RUN pip install --no-cache-dir \
    datasets==2.14.0 \
    pyarrow \
    xxhash \
    dill \
    multiprocess \
    fsspec==2023.6.0 \
    aiohttp \
    responses \
    psutil

# Install additional utilities
RUN pip install --no-cache-dir \
    ftfy \
    regex \
    tqdm \
    wandb \
    beautifulsoup4 \
    webdataset \
    torch-tools \
    visdom \
    easydict

# Install CLIP
RUN pip install --no-cache-dir git+https://github.com/openai/CLIP.git@main

# Clone and install DiffVG
WORKDIR /opt
RUN git clone https://github.com/BachiLi/diffvg.git && \
    cd diffvg && \
    git submodule update --init --recursive && \
    python setup.py install && \
    cd .. && rm -rf diffvg

# Clone SVGDreamer repository
WORKDIR /workspace
RUN git clone https://github.com/regix1/SVGDreamer.git . && \
    rm -rf .git

# Clone ImageReward if needed
RUN if [ ! -d "ImageReward" ]; then \
        git clone https://github.com/THUDM/ImageReward.git; \
    fi

# Create necessary directories
RUN mkdir -p logs checkpoint outputs

# Set Python path
ENV PYTHONPATH=/workspace:${PYTHONPATH}

# Verify installation
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}')" && \
    python -c "import numpy; print(f'NumPy: {numpy.__version__}')" && \
    python -c "import pydiffvg; print('DiffVG: OK')"

# Create entrypoint
RUN echo '#!/bin/bash\n\
source /opt/conda/etc/profile.d/conda.sh\n\
conda activate svgdreamer\n\
exec "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]