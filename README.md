# DJ Gizmo

AI-powered audio processing server for DGX Spark. Stem separation, audio analysis, and sound effect generation for DJ/remix applications.

![Demucs Web UI](https://img.shields.io/badge/GPU-GB10-76b900?style=flat&logo=nvidia) ![PyTorch](https://img.shields.io/badge/PyTorch-2.10-ee4c2c?style=flat&logo=pytorch) ![License](https://img.shields.io/badge/License-MIT-blue)

## Features

- **Stem Separation** - Split any song into drums, bass, other, vocals (Demucs htdemucs)
- **Sound Effect Generation** - Text-to-audio AI (Stable Audio Open)
- **Audio Analysis** - BPM, key detection, beat tracking, onset detection, waveform
- **Audio Rendering** - Time-stretch, pitch-shift, cut/trim operations
- **Real-time Updates** - Server-Sent Events for live job status
- **Web UI** - Drag-and-drop file upload with real-time progress
- **Fast GPU processing** - ~30s stems, ~1s effects (small model) on GB10
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
git clone https://github.com/bmdragos/demucs-dgx-spark.git
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

Access at **https://your-dgx-ip:8081**

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
| `/separate` | POST | Upload file for stem separation (queued) |
| `/analyze` | POST | Analyze audio for BPM, beats, onsets (immediate) |
| `/generate_effect` | POST | Generate sound effect from text prompt |
| `/jobs/{id}/render` | POST | Render stem with time-stretch, pitch-shift, cut |
| `/jobs` | GET | List recent jobs |
| `/jobs/stream` | GET | Server-Sent Events for real-time job updates |
| `/jobs/{id}/download/{stem}.wav` | GET | Download individual stem |
| `/jobs/{id}/download_all` | GET | Download all stems as ZIP |
| `/storage` | GET | Disk usage info |
| `/cleanup` | POST | Manual cleanup trigger |

### Audio Analysis

The `/analyze` endpoint returns timing information for DJ/music applications:

```bash
curl -X POST -F "file=@song.mp3" https://your-dgx:8081/analyze
```

```json
{
  "filename": "song.mp3",
  "duration": 187.4,
  "bpm": 128.0,
  "key": "Am",
  "beats": [0.0, 0.469, 0.938, 1.406, ...],
  "downbeats": [0.0, 1.875, 3.75, 5.625, ...],
  "onsets": [0.023, 0.482, 0.951, 1.217, ...],
  "waveform": [0.1, 0.3, 0.8, 0.6, ...],
  "beat_count": 400,
  "onset_count": 1250
}
```

| Field | Description |
|-------|-------------|
| `duration` | Total length in seconds |
| `bpm` | Estimated tempo |
| `key` | Musical key (e.g., "C", "Am", "F#m") |
| `beats` | Every beat position in seconds |
| `downbeats` | First beat of each bar (the "1" in 4/4) |
| `onsets` | Transient/hit positions (drums, notes) |
| `waveform` | RMS envelope (1000 points, normalized 0-1) |

### Sound Effect Generation

Generate sound effects from text descriptions using Stable Audio Open:

```bash
curl -X POST "https://your-dgx:8081/generate_effect?prompt=Air%20horn%20blast.%20High-quality.%20Stereo.&duration=3"
```

Returns a 44.1kHz stereo WAV file. Duration can be 1-47 seconds.

| Parameter | Description |
|-----------|-------------|
| `prompt` | Text description of the sound effect |
| `duration` | Length in seconds (default 5) |
| `model` | `small` (fast ~1s, max 11s) or `full` (slow ~65s, max 47s). Default: `small` |
| `negative_prompt` | What to avoid (default: "low quality, noise, distortion") |

#### Prompting Tips

**Keep it short** - 1-5 words works best for sound effects.

**Add quality boosters** - Dramatically improves output:
- `"High-quality."` - Better fidelity
- `"Stereo."` - Better spatial imaging
- `"44.1kHz"` - Prevents low-fi output

**Example prompts:**
```
"Air horn blast. High-quality. Stereo."
"Vinyl scratch. High-quality. Stereo."
"Cymbal crash. High-quality."
"Laser zap. Stereo."
"Glass shatter"
"Record rewind. High-quality. Stereo."
"Explosion boom. High-quality."
"Whoosh. Stereo."
```

**For rhythmic effects, add BPM:**
```
"Hammering at 120 BPM. High-quality."
"Clicking at 130 BPM"
```

**Model comparison:**
| Model | Generation Time | Max Duration | Best For |
|-------|-----------------|--------------|----------|
| `small` | ~1-2s | 11s | Quick sound effects, iteration |
| `full` | ~65s | 47s | Longer clips, higher quality |

**Known quirks:**
- Some outputs sound MIDI-ish → add `"Live"` or `"Acoustic"`
- Some outputs sound low-fi → add `"44.1kHz high-quality"`

#### Setup Required

The model is gated on HuggingFace. Before first use:
1. Create account at [huggingface.co](https://huggingface.co)
2. Accept license at [stabilityai/stable-audio-open-1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0)
3. Create access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Run inside container: `huggingface-cli login`

### Audio Rendering

Apply transformations to separated stems:

```bash
curl -X POST "https://your-dgx:8081/jobs/{job_id}/render" \
  -H "Content-Type: application/json" \
  -d '{
    "stem": "vocals",
    "speed": 1.2,
    "pitch_semitones": -2,
    "start_time": 10.0,
    "end_time": 30.0
  }'
```

| Parameter | Description |
|-----------|-------------|
| `stem` | Which stem to render (drums, bass, other, vocals) |
| `speed` | Time-stretch factor (0.5 = half speed, 2.0 = double) |
| `pitch_semitones` | Pitch shift (-12 to +12 semitones) |
| `start_time` | Cut start in seconds (optional) |
| `end_time` | Cut end in seconds (optional) |

## Configuration

Edit constants in `server.py`:

```python
SERVER_VERSION = "2.5.0"
MAX_RESULTS_AGE_HOURS = 24    # Auto-delete results older than this
MAX_RESULTS_SIZE_GB = 5.0     # Trigger cleanup when exceeded
MAX_JOB_HISTORY = 50          # Jobs to keep in memory
```

## Environment

- **Hardware:** DGX Spark (GB10 GPU, 120GB unified memory, ARM64)
- **Container:** `nvcr.io/nvidia/pytorch:25.12-py3`
- **Models:**
  - Demucs htdemucs (hybrid transformer) - stem separation
  - Stable Audio Open Small (497M params, ~1s) - fast sound effects
  - Stable Audio Open 1.0 (1.2B params, ~65s) - high quality generation
- **Output:** 44.1kHz stereo WAV

## License

MIT
