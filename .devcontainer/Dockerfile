# Use an appropriate base image with Python pre-installed
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04


ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES compute,graphics,utility

RUN apt-get update \
    && apt-get install -y \
    libxext6 \
    libvulkan1 \
    libvulkan-dev \
    vulkan-tools \
    pciutils \
    mesa-utils

RUN apt-get update && apt-get install -y cmake
# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3 and pip
RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set the default Python version to 3
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install necessary Python packages
RUN pip3 install ipykernel transforms3d trimesh plotly pyyaml lxml urdf_parser_py scipy networkx rtree tqdm

# Install PyTorch with CUDA support
RUN pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# # Install pytorch3d
# RUN export FORCE_CUDA=1 && \
#     export TORCH_CUDA_ARCH_LIST="7.0 7.2 7.5 8.0 8.6" && \
#     git clone https://github.com/facebookresearch/pytorch3d.git \
#     && cd pytorch3d \
#     && pip3 install -e .

# # Move to the thirdparty directory
# WORKDIR /app/DexGraspNet/thirdparty

# # Clone TorchSDF repository
# RUN git clone https://github.com/wrc042/TorchSDF.git

# # Move into TorchSDF directory
# WORKDIR /app/DexGraspNet/thirdparty/TorchSDF

# # Checkout to version 0.1.0
# RUN git checkout 0.1.0

# # Run the install script
# RUN bash install.sh && bash clean.sh && pip install -e .

# # Move back to the thirdparty directory
# WORKDIR /app/DexGraspNet/thirdparty

# # Move into pytorch_kinematics directory
# WORKDIR /app/DexGraspNet/thirdparty/pytorch_kinematics

# # Install pytorch_kinematics
# RUN pip install -e .

# # Set the working directory back to /app
# WORKDIR /app

# Set the default command to run when the container starts
CMD ["bash"]