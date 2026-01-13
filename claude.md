# DJ Gizmo

AI-powered audio processing server for DJ/remix applications. Runs on DGX Spark (GB10 GPU, 128GB unified memory).

## What This Is

A FastAPI server that provides:
- **Stem separation** - Split songs into drums, bass, other, vocals (~30s)
- **Sound effect generation** - Text-to-audio via Stable Audio Open (~1s small, ~65s full)
- **Audio analysis** - BPM, key, beats, onsets, segments, waveforms
- **Rendering** - Time-stretch, pitch-shift, cut stems on demand

## Architecture

```
server.py          # FastAPI server, job queues, all endpoints
index.html         # Web UI (drag-drop upload, real-time status)
setup.sh           # Surgical install script for NGC containers
stable-audio.md    # Model documentation for Stable Audio Open
```

## Key Technical Details

- **Container:** `nvcr.io/nvidia/pytorch:25.12-py3` on DGX Spark
- **Models:** Demucs htdemucs, Stable Audio Open (small + full)
- **Job queues:** Async queues for separation and effect generation
- **Analysis:** Parallel via ProcessPoolExecutor (5 workers)
- **Output:** 44.1kHz stereo WAV

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/separate` | POST | Upload file, queue stem separation |
| `/generate_effect` | POST | Queue sound effect generation |
| `/analyze` | POST | Immediate audio analysis |
| `/jobs/{id}` | GET | Job status + results + analysis |
| `/jobs/{id}/render` | POST | Render stem with BPM/pitch/cut |
| `/jobs/{id}/download/{stem}.wav` | GET | Download individual stem |
| `/effects/{id}` | GET | Effect job status |
| `/effects/{id}/download` | GET | Download generated effect |

## Effect Generation Models

| Model | Param | Speed | Max Duration |
|-------|-------|-------|--------------|
| `small` (default) | 497M | ~1s | 11s |
| `full` | 1.2B | ~65s | 47s |

## DGX MCP Integration

Project is configured in `~/.dgx/config.json`:
- Service: `demucs` (port 8081, HTTPS)
- Sync: `dgx_sync push/pull` for code deployment
- Restart: `dgx_service_restart demucs`

## Licensing

- **Demucs:** MIT (fully commercial OK)
- **Stable Audio Open:** Free under $1M revenue, enterprise license above

## Target Use Case

Consumer app for making remixes. Goal is simplicity - drop a song, get stems, add effects, export. The software should get out of the way.
