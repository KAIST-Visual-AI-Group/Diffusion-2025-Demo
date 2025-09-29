#!/bin/bash

# SD3.5 LoRA Training Setup Script
# This script helps users set up their training environment

echo "🔧 SD3.5 LoRA Training Environment Setup"
echo "========================================"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python
if ! command_exists python; then
    echo "❌ Python not found. Please install Python 3.8+ first."
    exit 1
fi

python_version=$(python --version 2>&1 | awk '{print $2}')
echo "✅ Python version: $python_version"

# Check pip
if ! command_exists pip; then
    echo "❌ pip not found. Please install pip first."
    exit 1
fi

echo "✅ pip found"

# Install requirements
echo ""
echo "📦 Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✅ Core dependencies installed"
else
    echo "⚠️ requirements.txt not found. Installing minimal dependencies..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install accelerate transformers diffusers peft datasets pillow tqdm
fi

# Install optional dependencies
echo ""
read -p "Install optional dependencies? (wandb, tensorboard, bitsandbytes) [y/N]: " install_optional
if [[ $install_optional =~ ^[Yy]$ ]]; then
    echo "📦 Installing optional dependencies..."
    pip install wandb tensorboard bitsandbytes
    echo "✅ Optional dependencies installed"
fi

# Configure accelerate
echo ""
echo "⚙️ Configuring Accelerate..."
if command_exists accelerate; then
    echo "Accelerate found. Running configuration..."
    accelerate config
    echo "✅ Accelerate configured"
else
    echo "❌ Accelerate not found. Please check installation."
    exit 1
fi

# Create directory structure
echo ""
echo "📁 Creating directory structure..."
mkdir -p outputs
mkdir -p examples/dataset/images
mkdir -p scripts/logs

echo "✅ Directory structure created"

# Check GPU availability
echo ""
echo "🔍 Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

echo ""
echo "🎉 Setup completed!"
echo ""
echo "📋 Next steps:"
echo "1. Prepare your dataset in examples/dataset/"
echo "2. Run basic training: bash scripts/train_basic.sh"
echo "3. Or advanced training: bash scripts/train_advanced.sh"
echo ""
echo "📚 For more information, check the README.md file" 