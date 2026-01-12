# Demucs DGX Spark

AI-powered audio stem separation running on NVIDIA DGX Spark. Separates any song into drums, bass, other, and vocals using Meta's Demucs (htdemucs model).

![Demucs Web UI](https://img.shields.io/badge/GPU-GB10-76b900?style=flat&logo=nvidia) ![PyTorch](https://img.shields.io/badge/PyTorch-2.10-ee4c2c?style=flat&logo=pytorch) ![License](https://img.shields.io/badge/License-MIT-blue)

## Features

- **Web UI** - Drag-and-drop file upload with real-time job status
- **Fast GPU processing** - ~6 minutes for a typical song on GB10
- **Multiple formats** - MP3, M4A, FLAC, WAV, OGG
- **Download options** - Individual stems or all as ZIP
- **Auto cleanup** - TTL-based cleanup (24h) and disk quota (5GB)

## Quick Start

### 1. Create Container

```bash
docker run -d --name demucs --runtime=nvidia --gpus all \
  -v ~/demucs:/workspace/demucs \
  -p 8081:8081 --ipc=host \
  -w /workspace/demucs \
  nvcr.io/nvidia/pytorch:25.12-py3 \
  tail -f /dev/null
```

### 2. Copy Files

```bash
# Clone this repo
git clone https://github.com/YOUR_USERNAME/demucs-dgx-spark.git
cd demucs-dgx-spark

# Copy to DGX Spark
scp -r server.py index.html setup.sh your-dgx:~/demucs/
```

### 3. Run Setup

```bash
# SSH into DGX Spark
ssh your-dgx

# Run setup inside container
docker exec demucs bash /workspace/demucs/setup.sh
```

### 4. Start Server

```bash
docker exec -d demucs bash -c "cd /workspace/demucs && python server.py"
```

Access at **http://your-dgx-ip:8081**

## Why Surgical Install?

NGC containers have a custom PyTorch build optimized for ARM64/GB10. Standard pip packages break CUDA:

```
demucs -> torchaudio -> torch (PyPI) = BROKEN
```

The `setup.sh` script installs packages with `--no-deps` to preserve NGC's PyTorch.

## Files

| File | Description |
|------|-------------|
| `server.py` | FastAPI server with job queue, cleanup, GPU stats |
| `index.html` | Modern web UI with drag-and-drop upload |
| `setup.sh` | Surgical install script preserving NGC PyTorch |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/health` | GET | Server status, GPU stats, queue size |
| `/separate` | POST | Upload file for separation |
| `/jobs` | GET | List recent jobs |
| `/jobs/{id}/download/{stem}.wav` | GET | Download individual stem |
| `/jobs/{id}/download_all` | GET | Download all stems as ZIP |
| `/storage` | GET | Disk usage info |
| `/cleanup` | POST | Manual cleanup trigger |

## Configuration

Edit constants in `server.py`:

```python
SERVER_VERSION = "2.1.0"
MAX_RESULTS_AGE_HOURS = 24    # Auto-delete results older than this
MAX_RESULTS_SIZE_GB = 5.0     # Trigger cleanup when exceeded
MAX_JOB_HISTORY = 50          # Jobs to keep in memory
```

## Environment

- **Hardware:** DGX Spark (GB10 GPU, 120GB unified memory, ARM64)
- **Container:** `nvcr.io/nvidia/pytorch:25.12-py3`
- **Model:** htdemucs (hybrid transformer)
- **Output:** 4 stems (drums, bass, other, vocals) as WAV

## License

MIT
