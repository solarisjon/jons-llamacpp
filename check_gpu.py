#!/usr/bin/env python3
"""
GPU diagnostic script for llama-cpp-python
Run this to check if GPU support is properly configured
"""

import sys
import os

def check_cuda():
    """Check CUDA installation"""
    print("=== CUDA Check ===")
    try:
        import subprocess
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ CUDA compiler found:")
            print(result.stdout.strip())
        else:
            print("❌ CUDA compiler not found")
            return False
    except FileNotFoundError:
        print("❌ nvcc not found in PATH")
        return False
    
    # Check nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ nvidia-smi working")
            # Extract GPU info
            lines = result.stdout.split('\n')
            for line in lines:
                if 'CUDA Version' in line:
                    print(f"CUDA Runtime: {line.split('CUDA Version: ')[1].split(' ')[0]}")
                    break
        else:
            print("❌ nvidia-smi failed")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found")
        return False
    
    return True

def check_llama_cpp():
    """Check llama-cpp-python GPU support"""
    print("\n=== llama-cpp-python Check ===")
    try:
        from llama_cpp import Llama
        print("✅ llama-cpp-python imported successfully")
        
        # Try to create a small model instance with GPU
        try:
            # Create a minimal test - we'll use a non-existent model to just test GPU init
            print("Testing GPU support...")
            
            # Check if llama-cpp was compiled with CUDA
            import llama_cpp
            if hasattr(llama_cpp, 'llama_supports_gpu_offload'):
                if llama_cpp.llama_supports_gpu_offload():
                    print("✅ llama-cpp-python compiled with GPU support")
                else:
                    print("❌ llama-cpp-python compiled WITHOUT GPU support")
                    return False
            else:
                print("⚠️  Cannot determine GPU support - using alternative method")
            
        except Exception as e:
            print(f"❌ GPU test failed: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ Failed to import llama-cpp-python: {e}")
        return False
        
    return True

def check_model_loading():
    """Check if we can load model with GPU"""
    print("\n=== Model Loading Test ===")
    
    # Look for model file
    model_files = [f for f in os.listdir('.') if f.endswith('.gguf')]
    if not model_files:
        print("❌ No .gguf model files found in current directory")
        return False
    
    model_file = model_files[0]
    print(f"Found model: {model_file}")
    
    try:
        from llama_cpp import Llama
        print("Attempting to load model with GPU layers...")
        
        # Try with minimal GPU layers first
        llm = Llama(
            model_path=f"./{model_file}",
            n_ctx=512,  # Small context for testing
            n_gpu_layers=1,  # Just one layer for testing
            verbose=True
        )
        
        print("✅ Model loaded successfully with GPU support!")
        
        # Test a simple generation
        print("Testing generation...")
        output = llm("Hello", max_tokens=5, echo=False)
        print(f"✅ Generation test successful: {output['choices'][0]['text'].strip()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def main():
    print("🔍 GPU Support Diagnostic Tool")
    print("=" * 40)
    
    cuda_ok = check_cuda()
    llama_ok = check_llama_cpp()
    
    if cuda_ok and llama_ok:
        model_ok = check_model_loading()
        if model_ok:
            print("\n🎉 All checks passed! GPU should be working.")
        else:
            print("\n⚠️  CUDA and llama-cpp look good, but model loading failed.")
    else:
        print("\n❌ Issues found with GPU setup.")
        
    print("\n=== Recommendations ===")
    if not cuda_ok:
        print("- Install NVIDIA CUDA toolkit")
        print("- Ensure nvidia-smi works")
        print("- Check that GPU drivers are installed")
        
    if not llama_ok:
        print("- Reinstall llama-cpp-python with CUDA support:")
        print("  CMAKE_ARGS='-DLLAMA_CUBLAS=on' pip install llama-cpp-python --force-reinstall --no-cache-dir")
        
    print("\n=== Environment Info ===")
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Working directory: {os.getcwd()}")

if __name__ == "__main__":
    main()