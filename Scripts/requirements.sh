#!/bin/bash

# =====================================================
# TrOCR Hindi Handwriting Project - VM Setup Script
# =====================================================

# Exit on any error
set -e

echo "Starting setup for TrOCR Hindi Handwriting Training..."

# 1. Update and upgrade the system
echo "Updating system packages..."
sudo apt update -y && sudo apt upgrade -y

# 2. Install basic utilities
echo "Installing basic utilities..."
sudo apt install -y git wget unzip curl build-essential python3-venv python3-pip

# 3. Set up Python environment
echo "Setting up Python virtual environment..."
python3 -m venv trocr_env
source trocr_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# 4. Install PyTorch with CUDA support for T4 (CUDA 11.8)
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU
echo "Verifying GPU..."
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device name:', torch.cuda.get_device_name(0))"

# 5. Install Hugging Face Transformers and datasets
echo "Installing Hugging Face libraries..."
pip install transformers datasets sentencepiece evaluate

# 6. Install image processing libraries
echo "Installing image processing libraries..."
pip install pillow torchvision opencv-python

# 8. Install additional useful libraries
pip install tqdm matplotlib scikit-learn

# 9. Confirm installation
echo "Installation complete. Installed packages:"
pip list | grep -E "torch|transformers|datasets|sentencepiece|evaluate|Pillow|opencv"
echo "source trocr_env/bin/activate"

# =====================================================
