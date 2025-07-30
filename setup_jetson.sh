#!/bin/bash
# Setup script for NVIDIA Jetson

echo "Setting up llama-cpp with CUDA support for NVIDIA Jetson..."

# Check if CUDA is available
if command -v nvcc &> /dev/null; then
    echo "CUDA found: $(nvcc --version | grep release)"
else
    echo "Warning: CUDA not found. Installing CPU-only version."
fi

# Install dependencies
echo "Installing Python dependencies with CUDA support..."

# Remove existing llama-cpp-python if present
uv remove llama-cpp-python 2>/dev/null || true

# Install with CUDA support
CMAKE_ARGS="-DLLAMA_CUBLAS=on" uv add llama-cpp-python

echo "Setup complete!"
echo ""
echo "To verify GPU support, run:"
echo "  uv run python -c \"from llama_cpp import Llama; print('GPU support available')\""
echo ""
echo "Monitor GPU usage with:"
echo "  watch -n 1 nvidia-smi"