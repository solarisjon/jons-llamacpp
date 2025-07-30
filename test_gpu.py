#!/usr/bin/env python3
"""
Simple GPU test script - loads model and generates text while showing GPU usage
"""

import os
import time
from llama_cpp import Llama

def find_model():
    """Find a GGUF model in current directory"""
    model_files = [f for f in os.listdir('.') if f.endswith('.gguf')]
    if not model_files:
        print("❌ No .gguf model files found!")
        print("Please download a model file first.")
        return None
    return model_files[0]

def main():
    print("🧪 GPU Test Script")
    print("=" * 30)
    
    model_file = find_model()
    if not model_file:
        return
    
    print(f"Using model: {model_file}")
    print("\n📊 Monitor GPU usage with: jtop")
    print("You should see GPU memory and compute usage increase when loading/generating\n")
    
    try:
        print("Loading model with GPU layers...")
        llm = Llama(
            model_path=f"./{model_file}",
            n_ctx=1024,
            n_threads=4,
            n_gpu_layers=-1,  # All layers to GPU
            verbose=True  # Show detailed loading info
        )
        
        print("\n✅ Model loaded! Check jtop for GPU memory usage.")
        print("Starting text generation test...\n")
        
        prompts = [
            "The future of AI is",
            "Once upon a time",
            "Python is a programming language that"
        ]
        
        for i, prompt in enumerate(prompts, 1):
            print(f"Test {i}/3: '{prompt}'")
            print("Generating... (check jtop for GPU compute usage)")
            
            start_time = time.time()
            output = llm(
                prompt,
                max_tokens=50,
                temperature=0.7,
                echo=False
            )
            end_time = time.time()
            
            response = output['choices'][0]['text'].strip()
            print(f"Response: {response}")
            print(f"Time: {end_time - start_time:.2f}s")
            print("-" * 40)
            
            time.sleep(1)  # Brief pause between tests
            
        print("\n🎉 GPU test complete!")
        print("If GPU was used, you should have seen:")
        print("- GPU memory usage during model loading")
        print("- GPU compute spikes during generation")
        print("- Faster generation compared to CPU-only")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("\nTroubleshooting steps:")
        print("1. Run: uv run python check_gpu.py")
        print("2. Ensure CUDA is installed: nvcc --version")
        print("3. Reinstall with GPU support: ./setup_jetson.sh")

if __name__ == "__main__":
    main()