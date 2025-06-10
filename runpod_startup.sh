#!/bin/bash
set -e  # Exit on any error

# Update and install dependencies
apt install g++ -y # for tcnn if needed
DEBIAN_FRONTEND=noninteractive apt-get install openssh-server  -y
export TZ=Etc/UTC

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
. $HOME/.local/bin/env

# Set up SSH for remote access
mkdir -p ~/.ssh
chmod 700 ~/.ssh
echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys

# Set up github credentials
echo "$DEPLOY_KEY" | base64 -d > ~/.ssh/id_ed25519
chmod 600 ~/.ssh/id_ed25519
ssh-keyscan github.com >> ~/.ssh/known_hosts

# Export CUDA environment variables
export CUDACXX=/usr/local/cuda/bin/nvcc
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc

# Clone repository
cd /workspace
git clone git@github.com:abhyantrika/SIEDD.git
cd SIEDD
git checkout new

# Install dependencies
export UV_CACHE_DIR=/workspace/.cache/uv
export UV_PYTHON_INSTALL_DIR=/workspace/.py_install
uv sync --inexact
. .venv/bin/activate

# Set up b2 and download configs
b2 sync ${B2_EXP_SCRATCH_PATH} experiment_scratch

# Write configs from environment variables (passthrough) to files
mkdir -p experiment_scratch

# Run experiments
{{CMD}}

# Kills the pod after it's done
runpodctl remove pod $RUNPOD_POD_ID
