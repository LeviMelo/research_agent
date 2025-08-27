# Use an official NVIDIA CUDA base image. This includes the necessary drivers.
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set environment variables to prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash

# Install system dependencies and Miniconda
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py311_23.10.0-1-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH

# Create and activate a Conda environment
RUN conda create -n research_env python=3.11 -y
ENV CONDA_DEFAULT_ENV=research_env
ENV CONDA_PREFIX=/opt/conda/envs/research_env
ENV PATH=$CONDA_PREFIX/bin:$PATH

# Install Python dependencies using pip within the Conda environment
# vLLM is installed first to ensure its dependencies are prioritized
RUN pip install "vllm>=0.5.0.post1" torch biopython

# Create a working directory and copy your script into it
WORKDIR /app
COPY . .

# Set the default command to run your Python script
CMD ["python", "./your_script_name.py"]