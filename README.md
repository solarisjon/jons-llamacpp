# Llama-cpp Chatbot Server

A web-based chatbot interface for running Llama models locally with llama-cpp-python. Features a modern web UI with dark/light mode toggle and OpenAI-compatible API endpoints.

## Features

- 🚀 FastAPI server with OpenAI-compatible API endpoints
- 🌐 Web-based chat interface with conversation history
- 🌙 Dark/light mode toggle with persistent preferences
- 🎮 GPU acceleration support (CUDA for NVIDIA cards/Jetson)
- 🌍 LAN access for multiple users
- ⚡ Real-time streaming responses
- 🔧 Configurable model parameters (temperature, max tokens, etc.)

## Requirements

- Python 3.8+
- [uv](https://docs.astral.sh/uv/) package manager
- A GGUF format model file
- For GPU: NVIDIA GPU with CUDA support

## Setup

### 1. Clone Repository
```bash
git clone https://github.com/solarisjon/jons-llamacpp.git
cd jons-llamacpp
```

### 2. Install Dependencies

**CPU-only (Mac/Linux/Windows):**
```bash
uv sync
```

**GPU Support (NVIDIA Jetson/Linux with CUDA):**
```bash
./setup_jetson.sh
```

### 3. Download a Model
Download a GGUF model file and place it in the project directory. Examples:
- [Mistral 7B Instruct](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF)
- [Llama 2 7B Chat](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF)

Update the `model_path` in `server.py` to match your model filename:
```python
llm = Llama(
    model_path="./your-model-file.gguf",  # Change this
    # ... other settings
)
```

## Usage

### Start the Server
```bash
uv run server.py
```

The server will start on port 8000 and display:
- Local access: http://localhost:8000
- LAN access: http://YOUR_IP:8000

### Web Interface
Open the URL in any browser to access the chat interface. Features include:
- Real-time chat with conversation history
- Dark/light mode toggle (top-right corner)
- Adjustable model parameters (server URL, max tokens, temperature)
- Loading indicators and error handling

### API Endpoints

The server provides OpenAI-compatible endpoints:

- `GET /` - Web chat interface
- `GET /health` - Health check
- `GET /v1/models` - List available models
- `POST /v1/completions` - Text completions
- `POST /v1/chat/completions` - Chat completions

**Example API usage:**
```bash
# Text completion
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time,",
    "max_tokens": 100,
    "temperature": 0.8
  }'

# Chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 100
  }'
```

## Configuration

Edit `server.py` to adjust model settings:

```python
llm = Llama(
    model_path="./your-model.gguf",
    n_ctx=2048,           # Context window size
    n_threads=4,          # CPU threads
    n_gpu_layers=-1,      # GPU layers (-1=all, 0=CPU only)
    verbose=True          # Show loading info
)
```

### GPU Settings
- `n_gpu_layers=-1`: Offload all layers to GPU (fastest)
- `n_gpu_layers=20`: Offload 20 layers to GPU (partial)
- `n_gpu_layers=0`: CPU only mode

## GPU Monitoring

Monitor GPU usage during inference:
```bash
# NVIDIA GPUs
watch -n 1 nvidia-smi

# Jetson devices
sudo tegrastats
```

## Files

- `main.py` - Simple command-line script
- `server.py` - FastAPI web server
- `chatbot.html` - Web chat interface
- `setup_jetson.sh` - GPU setup script for NVIDIA Jetson
- `pyproject.toml` - Project dependencies

## Troubleshooting

**Model not loading:**
- Ensure the model file path is correct in `server.py`
- Check that the model file is in GGUF format

**GPU not being used:**
- Verify CUDA installation: `nvcc --version`
- Check GPU memory: `nvidia-smi`
- Ensure llama-cpp-python was compiled with CUDA support

**Network access issues:**
- Check firewall settings for port 8000
- Use `ifconfig` or `ip addr` to find your IP address
- Ensure devices are on the same network

## License

MIT License - feel free to modify and distribute.