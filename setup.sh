#!/bin/bash
# DJ Gizmo container setup script
# Run this INSIDE the container after dgx_container_create

set -e

echo "=== DJ Gizmo Surgical Install ==="
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
pip install einops julius lameenc pyyaml tqdm soundfile dora-search librosa -q
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

# Step 6: Install Stable Audio Tools for sound effect generation
# NOTE: pip install stable-audio-tools fails due to pandas==2.0.2 pin (no ARM64/Python 3.12 wheel)
# We use surgical install: --no-deps from git + manual deps
echo "Step 6: Installing stable-audio-tools (surgical)..."
pip install --no-deps git+https://github.com/Stability-AI/stable-audio-tools.git -q
pip install einops-exts alias-free-torch local-attention vector-quantize-pytorch k-diffusion -q
pip install huggingface_hub -q
echo "  Done"

# Step 7: HuggingFace authentication for Stable Audio Open
# The model is gated - requires HuggingFace token and license acceptance
echo "Step 7: HuggingFace authentication..."
echo "  NOTE: Stable Audio Open is a gated model."
echo "  Before first use, you must:"
echo "    1. Create account at huggingface.co"
echo "    2. Create access token at huggingface.co/settings/tokens"
echo "    3. Accept license at huggingface.co/stabilityai/stable-audio-open-1.0"
echo "    4. Run: huggingface-cli login"
echo "  (Skipping automatic login - run manually when ready)"
echo "  Done"

# Verify installation
echo ""
echo "=== Verification ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from demucs.pretrained import get_model; print('demucs.pretrained: OK')"
python -c "from demucs.apply import apply_model; print('demucs.apply: OK')"
python -c "import stable_audio_tools; print('stable_audio_tools: OK')"
python -c "from fastapi import FastAPI; print('fastapi: OK')"
python -c "import librosa; print('librosa: OK')"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Test commands:"
echo "  Demucs: python -c \"from demucs.pretrained import get_model; m = get_model('htdemucs'); m.cuda(); print('Demucs on GPU')\""
echo "  Stable Audio: python -c \"from stable_audio_tools import get_pretrained_model; print('Stable Audio import OK')\""
echo ""
echo "IMPORTANT: Before using /generate_effect endpoint:"
echo "  1. Accept license at: https://huggingface.co/stabilityai/stable-audio-open-1.0"
echo "  2. Run: huggingface-cli login"
echo "  3. Enter your HuggingFace access token"
