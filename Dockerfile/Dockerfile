FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libx11-6 \
    libxrender1 \
    libxext6 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install \
    opencv-python \
    matplotlib \
    pandas \
    open3d \
    transformers \
    safetensors \
    spconv-cu118 \
    deformable_attention \
    tensorboard \
    wandb \ 
    pykitti

ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH

RUN ldconfig
