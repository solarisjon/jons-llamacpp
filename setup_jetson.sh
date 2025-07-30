#!/bin/bash
# Setup script for NVIDIA Jetson and CUDA-enabled systems

echo "🚀 Setting up llama-cpp with CUDA support..."
echo ""

# Function to check command existence
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "=== Checking Prerequisites ==="

# Check CUDA
if command_exists nvcc; then
    CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    echo "✅ CUDA found: $CUDA_VERSION"
else
    echo "❌ CUDA compiler (nvcc) not found!"
    echo "   Please install CUDA toolkit first."
    exit 1
fi

# Check nvidia-smi
if command_exists nvidia-smi; then
    echo "✅ nvidia-smi found"
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
    echo "   GPU: $GPU_INFO"
else
    echo "❌ nvidia-smi not found!"
    echo "   Please install NVIDIA drivers first."
    exit 1
fi

# Check Python
if command_exists python3; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo "✅ Python found: $PYTHON_VERSION"
else
    echo "❌ Python 3 not found!"
    exit 1
fi

# Check uv
if command_exists uv; then
    echo "✅ uv package manager found"
else
    echo "❌ uv not found! Please install: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo ""
echo "=== Installing Dependencies ==="

# Sync basic dependencies first
echo "Installing base dependencies..."
uv sync

# Remove existing llama-cpp-python if present
echo "Removing existing llama-cpp-python..."
uv remove llama-cpp-python 2>/dev/null || true

# Install with CUDA support
echo "Installing llama-cpp-python with CUDA support..."
echo "This may take 10-15 minutes to compile..."

# Set environment variables for compilation
export CMAKE_ARGS="-DLLAMA_CUBLAS=on"
export CUDACXX=/usr/local/cuda/bin/nvcc

# Install with force reinstall to ensure CUDA compilation
if uv add llama-cpp-python --reinstall; then
    echo "✅ llama-cpp-python installed successfully!"
else
    echo "❌ Installation failed!"
    echo ""
    echo "Try manual installation:"
    echo "  CMAKE_ARGS='-DLLAMA_CUBLAS=on' pip install llama-cpp-python --force-reinstall --no-cache-dir"
    exit 1
fi

echo ""
echo "=== Verification ==="

# Run diagnostic script
if [ -f "check_gpu.py" ]; then
    echo "Running GPU diagnostic..."
    uv run python check_gpu.py
else
    echo "Basic verification..."
    if uv run python -c "from llama_cpp import Llama; print('✅ llama-cpp-python imported successfully')"; then
        echo "✅ Installation appears successful"
    else
        echo "❌ Import failed - installation may have issues"
        exit 1
    fi
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "=== Next Steps ==="
echo "1. Download a GGUF model file to this directory"
echo "2. Update model_path in server.py"
echo "3. Start the server: uv run server.py"
echo "4. Monitor GPU usage: jtop (or nvidia-smi)"
echo ""
echo "=== Troubleshooting ==="
echo "- Run diagnostic: uv run python check_gpu.py"
echo "- Check GPU usage: jtop"
echo "- View server logs for GPU initialization messages"