"""
Demucs Audio Separation Server

PyTorch Demucs inference server for DGX Spark.
Separates audio into stems: drums, bass, other, vocals.

Endpoints:
  GET  /                        - Web dashboard with drag-and-drop upload
  GET  /health                  - Health check with GPU stats
  GET  /storage                 - Storage usage info
  POST /cleanup                 - Trigger manual cleanup
  POST /separate                - Upload and separate audio
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
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from enum import Enum
from collections import deque

import numpy as np
import torch
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
import uvicorn

# Constants
SERVER_VERSION = "2.1.0"
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

        # Load audio
        audio, sr = sf.read(str(temp_wav))
        temp_wav.unlink()  # Clean up temp wav

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

        await job_queue.update(job.id,
                               status=JobStatus.COMPLETED,
                               completed_at=datetime.now(),
                               result_path=str(result_dir),
                               progress=1.0)

        print(f"Job {job.id} completed: {duration:.1f}s audio -> 4 stems + ZIP")

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
        # Add file sizes for completed jobs
        if j.status == JobStatus.COMPLETED and j.result_path:
            sizes = get_job_file_sizes(j.result_path)
            job_data["sizes"] = {k: format_size(v) for k, v in sizes.items()}
        result.append(job_data)
    return {"jobs": result}


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
    uvicorn.run(app, host="0.0.0.0", port=8081)
