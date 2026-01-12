#!/bin/bash
# Demucs container setup script
# Run this INSIDE the container after dgx_container_create

set -e

echo "=== Demucs Surgical Install ==="
echo "Preserving NGC PyTorch CUDA support"
echo ""

# Verify we're starting with NGC PyTorch
echo "Step 0: Verify NGC PyTorch..."
python -c "import torch; assert 'nv' in torch.__version__, 'Not NGC PyTorch!'; print(f'  NGC PyTorch: {torch.__version__}')"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'  CUDA: OK')"
echo ""

# Step 1: Install demucs without dependencies
echo "Step 1: Installing demucs (--no-deps)..."
pip install --no-deps demucs -q
echo "  Done"

# Step 2: Install safe dependencies
echo "Step 2: Installing safe dependencies..."
pip install einops julius lameenc pyyaml tqdm soundfile dora-search -q
echo "  Done"

# Step 3: Install torch-dependent packages without deps
echo "Step 3: Installing openunmix and torchaudio (--no-deps)..."
pip install --no-deps openunmix torchaudio -q
echo "  Done"

# Step 4: Install web server dependencies
echo "Step 4: Installing FastAPI and uvicorn..."
pip install fastapi uvicorn python-multipart -q
echo "  Done"

# Step 5: Install ffmpeg for audio format conversion
echo "Step 5: Installing ffmpeg..."
apt-get update -qq && apt-get install -y -qq ffmpeg > /dev/null 2>&1
echo "  Done"

# Verify installation
echo ""
echo "=== Verification ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from demucs.pretrained import get_model; print('demucs.pretrained: OK')"
python -c "from demucs.apply import apply_model; print('demucs.apply: OK')"
python -c "from fastapi import FastAPI; print('fastapi: OK')"

echo ""
echo "=== Setup Complete ==="
echo "To test: python -c \"from demucs.pretrained import get_model; m = get_model('htdemucs'); m.cuda(); print('Model on GPU')\""
