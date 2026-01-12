"""
Demucs Audio Separation Server

PyTorch Demucs inference server for DGX Spark.
Separates audio into stems: drums, bass, other, vocals.
Also provides audio analysis (BPM, beats, onsets).

Endpoints:
  GET  /                        - Web dashboard with drag-and-drop upload
  GET  /health                  - Health check with GPU stats
  GET  /storage                 - Storage usage info
  POST /cleanup                 - Trigger manual cleanup
  POST /separate                - Upload and separate audio
  POST /analyze                 - Analyze audio (BPM, beats, onsets)
  GET  /jobs                    - List jobs
  GET  /jobs/{id}               - Get job status/results
  GET  /jobs/{id}/download/{stem} - Download a stem
  GET  /jobs/{id}/download_all  - Download all stems as ZIP

Cleanup:
  - Startup: Cleans orphaned results from previous sessions
  - TTL: Deletes results older than MAX_RESULTS_AGE_HOURS (24h)
  - Quota: Deletes oldest when exceeding MAX_RESULTS_SIZE_GB (5GB)
  - Manual: POST /cleanup to trigger all cleanups
"""

import os
import io
import uuid
import asyncio
import subprocess
import zipfile
import shutil
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from enum import Enum
from collections import deque

import numpy as np
import torch
import soundfile as sf
import librosa
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
import uvicorn

# Constants
SERVER_VERSION = "2.2.0"
RESULTS_DIR = Path("/workspace/results")
SAMPLE_RATE = 44100
STEMS = ["drums", "bass", "other", "vocals"]

# Cleanup settings
MAX_RESULTS_AGE_HOURS = 24  # Delete results older than this
MAX_RESULTS_SIZE_GB = 5.0   # Delete oldest when exceeding this
MAX_JOB_HISTORY = 50        # Keep this many jobs in memory

# Ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Demucs Separation Server", version=SERVER_VERSION)

# Global model (loaded once at startup)
model = None


def load_demucs_model():
    """Load the Demucs model."""
    global model
    if model is None:
        from demucs.pretrained import get_model
        print("Loading htdemucs model...")
        model = get_model('htdemucs')
        model.cuda()
        model.eval()
        print(f"Model loaded on {next(model.parameters()).device}")
    return model


# --- Job Queue ---

class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    status: JobStatus
    created_at: datetime
    filename: str = ""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    result_path: Optional[str] = None
    progress: float = 0.0
    duration: float = 0.0
    analysis: Optional[dict] = None  # Auto-analysis results


class JobQueue:
    def __init__(self, max_history: int = MAX_JOB_HISTORY):
        self.jobs: dict[str, Job] = {}
        self.queue: deque[str] = deque()
        self.max_history = max_history
        self._lock = asyncio.Lock()

    async def add(self, filename: str) -> Job:
        job_id = str(uuid.uuid4())[:8]
        job = Job(
            id=job_id,
            status=JobStatus.QUEUED,
            created_at=datetime.now(),
            filename=filename
        )
        async with self._lock:
            self.jobs[job_id] = job
            self.queue.append(job_id)
            self._cleanup_old_jobs()
        return job

    async def get(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    async def next(self) -> Optional[Job]:
        async with self._lock:
            if self.queue:
                job_id = self.queue.popleft()
                return self.jobs.get(job_id)
        return None

    async def update(self, job_id: str, **kwargs):
        if job_id in self.jobs:
            for k, v in kwargs.items():
                setattr(self.jobs[job_id], k, v)

    def _cleanup_old_jobs(self):
        """Remove jobs exceeding max_history from memory."""
        completed = [j for j in self.jobs.values()
                     if j.status in (JobStatus.COMPLETED, JobStatus.FAILED)]
        if len(completed) > self.max_history:
            completed.sort(key=lambda x: x.completed_at or x.created_at)
            for job in completed[:-self.max_history]:
                if job.result_path and Path(job.result_path).exists():
                    shutil.rmtree(job.result_path, ignore_errors=True)
                del self.jobs[job.id]

    def list_jobs(self, limit: int = 20) -> list[Job]:
        jobs = sorted(self.jobs.values(),
                      key=lambda x: x.created_at, reverse=True)
        return jobs[:limit]

    def known_job_ids(self) -> set[str]:
        """Return set of job IDs currently tracked."""
        return set(self.jobs.keys())


def get_results_disk_usage() -> float:
    """Get total size of results directory in GB."""
    total = 0
    if RESULTS_DIR.exists():
        for f in RESULTS_DIR.rglob('*'):
            if f.is_file():
                total += f.stat().st_size
    return total / (1024 ** 3)


def cleanup_orphaned_results(known_ids: set[str]) -> int:
    """Delete result folders not tracked by job queue. Returns count deleted."""
    deleted = 0
    if not RESULTS_DIR.exists():
        return 0
    for item in RESULTS_DIR.iterdir():
        if item.is_dir() and item.name not in known_ids:
            shutil.rmtree(item, ignore_errors=True)
            deleted += 1
    return deleted


def cleanup_old_results(max_age_hours: float = MAX_RESULTS_AGE_HOURS) -> int:
    """Delete result folders older than max_age_hours. Returns count deleted."""
    deleted = 0
    if not RESULTS_DIR.exists():
        return 0
    cutoff = time.time() - (max_age_hours * 3600)
    for item in RESULTS_DIR.iterdir():
        if item.is_dir():
            try:
                mtime = item.stat().st_mtime
                if mtime < cutoff:
                    shutil.rmtree(item, ignore_errors=True)
                    deleted += 1
            except OSError:
                pass
    return deleted


def cleanup_by_disk_quota(max_gb: float = MAX_RESULTS_SIZE_GB) -> int:
    """Delete oldest results until under disk quota. Returns count deleted."""
    deleted = 0
    if not RESULTS_DIR.exists():
        return 0

    while get_results_disk_usage() > max_gb:
        # Find oldest result folder
        folders = [(d, d.stat().st_mtime) for d in RESULTS_DIR.iterdir() if d.is_dir()]
        if not folders:
            break
        folders.sort(key=lambda x: x[1])
        oldest = folders[0][0]
        shutil.rmtree(oldest, ignore_errors=True)
        deleted += 1

    return deleted


def run_all_cleanups(known_ids: set[str]) -> dict:
    """Run all cleanup routines. Returns summary."""
    return {
        "orphaned_deleted": cleanup_orphaned_results(known_ids),
        "expired_deleted": cleanup_old_results(),
        "quota_deleted": cleanup_by_disk_quota(),
        "disk_usage_gb": round(get_results_disk_usage(), 2)
    }


def format_size(bytes_size: int) -> str:
    """Format bytes as human-readable size."""
    if bytes_size < 1024:
        return f"{bytes_size}B"
    elif bytes_size < 1024 * 1024:
        return f"{bytes_size / 1024:.1f}KB"
    else:
        return f"{bytes_size / (1024 * 1024):.1f}MB"


def get_job_file_sizes(result_path: str) -> dict:
    """Get file sizes for stems in a job result directory."""
    sizes = {}
    total = 0
    path = Path(result_path)
    if path.exists():
        for stem in STEMS:
            stem_file = path / f"{stem}.wav"
            if stem_file.exists():
                size = stem_file.stat().st_size
                sizes[stem] = size
                total += size
    sizes['total'] = total
    return sizes


job_queue = JobQueue()


# --- Audio Analysis ---

# Krumhansl-Schmuckler key profiles for key detection
KEY_PROFILES = {
    'major': [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    'minor': [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
}
KEY_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def detect_key(y, sr) -> str:
    """Detect musical key using Krumhansl-Schmuckler algorithm."""
    # Get chroma features
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_avg = np.mean(chroma, axis=1)

    # Correlate with major/minor profiles for each root
    best_corr = -1
    best_key = "Unknown"

    for mode, profile in KEY_PROFILES.items():
        for shift in range(12):
            # Rotate profile to match each root note
            rotated = np.roll(profile, shift)
            corr = np.corrcoef(chroma_avg, rotated)[0, 1]
            if corr > best_corr:
                best_corr = corr
                suffix = "" if mode == 'major' else "m"
                best_key = f"{KEY_NAMES[shift]}{suffix}"

    return best_key


def get_waveform_envelope(y, sr, num_points=1000) -> list:
    """Get downsampled RMS envelope for visualization."""
    # Compute RMS energy
    rms = librosa.feature.rms(y=y)[0]

    # Downsample to target number of points
    if len(rms) > num_points:
        step = len(rms) // num_points
        rms = rms[::step][:num_points]

    # Normalize to 0-1 range
    if rms.max() > 0:
        rms = rms / rms.max()

    return [round(float(v), 3) for v in rms]


def analyze_audio_file(file_path: str, include_waveform: bool = True) -> dict:
    """Analyze a single audio file for BPM, beats, onsets, key, waveform."""
    y, sr = librosa.load(file_path, sr=None)
    duration = len(y) / sr

    # Beat tracking
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beats = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    if hasattr(tempo, '__len__'):
        bpm = float(tempo[0]) if len(tempo) > 0 else 0.0
    else:
        bpm = float(tempo)

    # Downbeats (every 4 beats)
    downbeats = beats[::4] if beats else []

    # Onset detection
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onsets = librosa.frames_to_time(onset_frames, sr=sr).tolist()

    # Key detection
    key = detect_key(y, sr)

    result = {
        "duration": round(duration, 3),
        "bpm": round(bpm, 1),
        "key": key,
        "beats": [round(b, 3) for b in beats],
        "downbeats": [round(d, 3) for d in downbeats],
        "onsets": [round(o, 3) for o in onsets],
        "beat_count": len(beats),
        "onset_count": len(onsets)
    }

    # Waveform envelope (optional, can be large)
    if include_waveform:
        result["waveform"] = get_waveform_envelope(y, sr)

    return result


# --- Audio Processing ---

async def process_audio(job: Job, audio_data: bytes):
    """Process audio through Demucs."""
    try:
        await job_queue.update(job.id,
                               status=JobStatus.PROCESSING,
                               started_at=datetime.now())

        # Load model
        demucs_model = load_demucs_model()
        from demucs.apply import apply_model

        # Save temp file for ffmpeg conversion (handles m4a, mp3, etc.)
        temp_input = RESULTS_DIR / f"{job.id}_input.tmp"
        temp_wav = RESULTS_DIR / f"{job.id}_input.wav"

        temp_input.write_bytes(audio_data)

        # Convert to WAV using ffmpeg
        result = subprocess.run([
            'ffmpeg', '-y', '-i', str(temp_input),
            '-ar', str(SAMPLE_RATE), '-ac', '2',
            str(temp_wav)
        ], capture_output=True)

        temp_input.unlink()  # Clean up temp input

        if not temp_wav.exists():
            raise RuntimeError(f"FFmpeg conversion failed: {result.stderr.decode()}")

        await job_queue.update(job.id, progress=0.1)

        # Load audio (keep temp_wav for analysis later)
        audio, sr = sf.read(str(temp_wav))

        # Convert to (channels, samples)
        if audio.ndim == 1:
            audio = np.stack([audio, audio])
        else:
            audio = audio.T

        duration = audio.shape[1] / sr
        await job_queue.update(job.id, duration=duration, progress=0.2)

        # Convert to tensor
        waveform = torch.tensor(audio, dtype=torch.float32, device='cuda')

        # Run separation
        print(f"Separating {duration:.1f}s of audio...")
        with torch.no_grad():
            sources = apply_model(demucs_model, waveform.unsqueeze(0), progress=True)[0]

        await job_queue.update(job.id, progress=0.9)

        # Save results
        result_dir = RESULTS_DIR / job.id
        result_dir.mkdir(parents=True, exist_ok=True)

        for i, stem_name in enumerate(STEMS):
            stem_audio = sources[i].T.cpu().numpy()
            sf.write(str(result_dir / f"{stem_name}.wav"), stem_audio, sr)

        # Pre-create ZIP file (no compression - WAV doesn't compress well)
        base_name = Path(job.filename).stem
        zip_path = result_dir / "stems.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zf:
            for stem_name in STEMS:
                stem_file = result_dir / f"{stem_name}.wav"
                zf.write(stem_file, f"{base_name}_{stem_name}.wav")

        await job_queue.update(job.id, progress=0.92)

        # Auto-analyze full mix and stems
        print(f"Analyzing audio...")
        analysis = {}

        try:
            # Analyze full mix
            analysis["mix"] = analyze_audio_file(str(temp_wav))
        except Exception as e:
            analysis["mix"] = {"error": str(e)}
        finally:
            temp_wav.unlink()  # Clean up temp wav

        await job_queue.update(job.id, progress=0.95)

        # Analyze each stem
        for stem_name in STEMS:
            stem_file = result_dir / f"{stem_name}.wav"
            try:
                analysis[stem_name] = analyze_audio_file(str(stem_file))
            except Exception as e:
                analysis[stem_name] = {"error": str(e)}

        await job_queue.update(job.id,
                               status=JobStatus.COMPLETED,
                               completed_at=datetime.now(),
                               result_path=str(result_dir),
                               analysis=analysis,
                               progress=1.0)

        print(f"Job {job.id} completed: {duration:.1f}s audio -> 4 stems + ZIP + analysis")

    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"Job {job.id} failed: {e}")
        await job_queue.update(job.id,
                               status=JobStatus.FAILED,
                               completed_at=datetime.now(),
                               error=error_msg)


# --- Background Worker ---

async def process_queue():
    """Background worker to process job queue."""
    while True:
        job = await job_queue.next()
        if job:
            temp_path = RESULTS_DIR / f"{job.id}_input.bin"
            if temp_path.exists():
                audio_data = temp_path.read_bytes()
                await process_audio(job, audio_data)
                temp_path.unlink()
        await asyncio.sleep(0.5)


@app.on_event("startup")
async def startup():
    # Run cleanup on startup (orphaned files from previous sessions)
    cleanup_result = run_all_cleanups(job_queue.known_job_ids())
    print(f"Startup cleanup: {cleanup_result}")

    # Pre-load model
    load_demucs_model()

    # Start queue processor
    asyncio.create_task(process_queue())


# --- API Endpoints ---

# Path to index.html (same directory as server.py)
INDEX_HTML_PATH = Path(__file__).parent / "index.html"


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the web dashboard."""
    if INDEX_HTML_PATH.exists():
        return HTMLResponse(content=INDEX_HTML_PATH.read_text())
    return HTMLResponse(content="<h1>index.html not found</h1>", status_code=500)


def get_gpu_info() -> dict:
    """Get GPU stats. Handles GB10 unified memory architecture."""
    info = {}
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            if len(parts) >= 2:
                info['utilization'] = parts[0].strip()
                info['temperature'] = parts[1].strip()

        # Get memory from PyTorch (works on GB10 unified memory)
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            info['memory_used'] = f"{allocated:.1f}GB"
    except:
        pass
    return info


@app.get("/health")
async def health():
    """Health check."""
    gpu = get_gpu_info()
    return {
        "status": "ok",
        "version": SERVER_VERSION,
        "model": "htdemucs",
        "gpu": gpu,
        "queue_size": len(job_queue.queue),
        "cuda_available": torch.cuda.is_available()
    }


@app.post("/cleanup")
async def cleanup():
    """Manually trigger cleanup of old results."""
    result = run_all_cleanups(job_queue.known_job_ids())
    return {
        "status": "ok",
        "cleanup": result
    }


@app.get("/storage")
async def storage_info():
    """Get storage usage info."""
    return {
        "disk_usage_gb": round(get_results_disk_usage(), 2),
        "max_quota_gb": MAX_RESULTS_SIZE_GB,
        "max_age_hours": MAX_RESULTS_AGE_HOURS,
        "results_dir": str(RESULTS_DIR)
    }


@app.post("/separate")
async def separate_audio(file: UploadFile = File(...)):
    """Upload and separate audio into stems."""
    # Validate file extension
    valid_extensions = ('.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac')
    if not file.filename.lower().endswith(valid_extensions):
        raise HTTPException(400, f"Unsupported format. Use: {', '.join(valid_extensions)}")

    # Read file
    audio_data = await file.read()

    # Size limit (500MB)
    if len(audio_data) > 500 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 500MB)")

    # Create job
    job = await job_queue.add(file.filename)

    # Save input temporarily
    temp_path = RESULTS_DIR / f"{job.id}_input.bin"
    temp_path.write_bytes(audio_data)

    return {
        "job_id": job.id,
        "status": job.status.value,
        "message": f"Queued for processing. Check /jobs/{job.id} for status."
    }


@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """Analyze audio for BPM, beats, and onsets. Returns immediately (no queue)."""
    import tempfile

    # Validate file extension
    valid_extensions = ('.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac')
    if not file.filename.lower().endswith(valid_extensions):
        raise HTTPException(400, f"Unsupported format. Use: {', '.join(valid_extensions)}")

    # Read file
    audio_data = await file.read()

    # Size limit (100MB for analysis)
    if len(audio_data) > 100 * 1024 * 1024:
        raise HTTPException(400, "File too large for analysis (max 100MB)")

    try:
        # Save to temp file for librosa/ffmpeg
        with tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix, delete=False) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name

        # Load audio with librosa (handles format conversion via ffmpeg)
        y, sr = librosa.load(tmp_path, sr=None)
        Path(tmp_path).unlink()  # Clean up temp file

        duration = len(y) / sr

        # Beat tracking
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beats = librosa.frames_to_time(beat_frames, sr=sr).tolist()

        # Handle tempo - could be array or scalar depending on librosa version
        if hasattr(tempo, '__len__'):
            bpm = float(tempo[0]) if len(tempo) > 0 else 0.0
        else:
            bpm = float(tempo)

        # Downbeats (every 4 beats, assuming 4/4 time)
        downbeats = beats[::4] if beats else []

        # Onset detection
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onsets = librosa.frames_to_time(onset_frames, sr=sr).tolist()

        return {
            "filename": file.filename,
            "duration": round(duration, 3),
            "bpm": round(bpm, 1),
            "beats": [round(b, 3) for b in beats],
            "downbeats": [round(d, 3) for d in downbeats],
            "onsets": [round(o, 3) for o in onsets],
            "beat_count": len(beats),
            "onset_count": len(onsets)
        }

    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {str(e)}")


@app.get("/jobs/{job_id}/analyze")
async def analyze_job_stems(job_id: str, stem: str = None):
    """
    Get analysis for stems from a completed separation job.
    Returns cached auto-analysis if available, otherwise computes on-demand.

    - No stem param: return all stems + mix analysis
    - stem=drums|bass|other|vocals|mix: return specific analysis
    """
    job = await job_queue.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(400, "Job not complete")

    # If we have cached analysis, use it
    if job.analysis:
        if stem:
            if stem not in job.analysis:
                raise HTTPException(400, f"Invalid stem. Use: mix, {', '.join(STEMS)}")
            return {
                "job_id": job_id,
                "filename": job.filename,
                "stem": stem,
                "analysis": job.analysis[stem]
            }
        return {
            "job_id": job_id,
            "filename": job.filename,
            "analysis": job.analysis
        }

    # Fallback: compute on-demand (for jobs created before auto-analysis)
    result_path = Path(job.result_path)

    # Determine which stems to analyze
    if stem:
        if stem not in STEMS:
            raise HTTPException(400, f"Invalid stem. Use: {', '.join(STEMS)}")
        stems_to_analyze = [stem]
    else:
        stems_to_analyze = STEMS

    results = {}

    for stem_name in stems_to_analyze:
        stem_file = result_path / f"{stem_name}.wav"
        if not stem_file.exists():
            continue

        try:
            y, sr = librosa.load(str(stem_file), sr=None)
            duration = len(y) / sr

            # Beat tracking
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            beats = librosa.frames_to_time(beat_frames, sr=sr).tolist()

            if hasattr(tempo, '__len__'):
                bpm = float(tempo[0]) if len(tempo) > 0 else 0.0
            else:
                bpm = float(tempo)

            # Downbeats (every 4 beats)
            downbeats = beats[::4] if beats else []

            # Onset detection
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            onsets = librosa.frames_to_time(onset_frames, sr=sr).tolist()

            results[stem_name] = {
                "duration": round(duration, 3),
                "bpm": round(bpm, 1),
                "beats": [round(b, 3) for b in beats],
                "downbeats": [round(d, 3) for d in downbeats],
                "onsets": [round(o, 3) for o in onsets],
                "beat_count": len(beats),
                "onset_count": len(onsets)
            }
        except Exception as e:
            results[stem_name] = {"error": str(e)}

    return {
        "job_id": job_id,
        "filename": job.filename,
        "stems_analyzed": list(results.keys()),
        "analysis": results
    }


@app.get("/jobs")
async def list_jobs(limit: int = 20):
    """List recent jobs."""
    jobs = job_queue.list_jobs(limit)
    result = []
    for j in jobs:
        job_data = {
            "id": j.id,
            "filename": j.filename,
            "status": j.status.value,
            "progress": j.progress,
            "duration": j.duration,
            "created_at": j.created_at.isoformat(),
            "error": j.error
        }
        # Add file sizes and BPM for completed jobs
        if j.status == JobStatus.COMPLETED and j.result_path:
            sizes = get_job_file_sizes(j.result_path)
            job_data["sizes"] = {k: format_size(v) for k, v in sizes.items()}
            # Include BPM from analysis if available
            if j.analysis and "mix" in j.analysis:
                job_data["bpm"] = j.analysis["mix"].get("bpm")
                job_data["key"] = j.analysis["mix"].get("key")
        result.append(job_data)
    return {"jobs": result}


@app.get("/jobs/stream")
async def stream_jobs():
    """Server-Sent Events stream for real-time job updates."""
    async def event_generator():
        last_data = None
        while True:
            # Build current jobs data
            jobs = job_queue.list_jobs(20)
            data = []
            for j in jobs:
                job_data = {
                    "id": j.id,
                    "filename": j.filename,
                    "status": j.status.value,
                    "progress": j.progress,
                    "duration": j.duration,
                    "error": j.error
                }
                if j.status == JobStatus.COMPLETED and j.result_path:
                    sizes = get_job_file_sizes(j.result_path)
                    job_data["sizes"] = {k: format_size(v) for k, v in sizes.items()}
                    if j.analysis and "mix" in j.analysis:
                        job_data["bpm"] = j.analysis["mix"].get("bpm")
                data.append(job_data)

            # Only send if data changed
            current = json.dumps(data)
            if current != last_data:
                last_data = current
                yield f"data: {current}\n\n"

            await asyncio.sleep(0.3)  # Check every 300ms

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Get job status and results."""
    job = await job_queue.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    result = {
        "id": job.id,
        "filename": job.filename,
        "status": job.status.value,
        "progress": job.progress,
        "duration": job.duration,
        "created_at": job.created_at.isoformat(),
        "error": job.error
    }

    if job.status == JobStatus.COMPLETED and job.result_path:
        result["stems"] = STEMS
        result["download_urls"] = {
            stem: f"/jobs/{job_id}/download/{stem}.wav"
            for stem in STEMS
        }
        # Include analysis if available
        if job.analysis:
            result["analysis"] = job.analysis

    return result


@app.get("/jobs/{job_id}/download/{filename}")
async def download_stem(job_id: str, filename: str):
    """Download a separated stem."""
    job = await job_queue.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(400, "Job not complete")

    file_path = Path(job.result_path) / filename
    if not file_path.exists():
        raise HTTPException(404, "File not found")

    # Use original filename as base for download
    base_name = Path(job.filename).stem
    stem_name = file_path.stem
    download_name = f"{base_name}_{stem_name}.wav"

    return FileResponse(file_path, filename=download_name)


@app.get("/jobs/{job_id}/download_all")
async def download_all_stems(job_id: str):
    """Download all stems as a zip file (pre-created, no compression overhead)."""
    job = await job_queue.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(400, "Job not complete")

    base_name = Path(job.filename).stem
    result_path = Path(job.result_path)
    zip_path = result_path / "stems.zip"

    if not zip_path.exists():
        raise HTTPException(404, "ZIP file not found")

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"{base_name}_stems.zip"
    )


if __name__ == "__main__":
    # Check for SSL certs
    cert_file = Path(__file__).parent / "192.168.1.159.pem"
    key_file = Path(__file__).parent / "192.168.1.159-key.pem"

    if cert_file.exists() and key_file.exists():
        print("Starting with HTTPS...")
        uvicorn.run(app, host="0.0.0.0", port=8081,
                    ssl_certfile=str(cert_file),
                    ssl_keyfile=str(key_file))
    else:
        print("No SSL certs found, starting with HTTP...")
        uvicorn.run(app, host="0.0.0.0", port=8081)
