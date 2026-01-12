"""
DJ Gizmo - Audio Separation & Generation Server

PyTorch inference server for DGX Spark.
- Stem separation via Demucs (drums, bass, other, vocals)
- Audio analysis (BPM, key, beats, onsets, segments)
- Sound effect generation via AudioGen (text-to-sound)
- Stem rendering (time-stretch, pitch-shift, cut)

Endpoints:
  GET  /                        - Web dashboard
  GET  /health                  - Health check with GPU stats
  POST /separate                - Upload and separate audio
  POST /analyze                 - Analyze audio (BPM, beats, onsets)
  POST /generate_effect         - Generate sound effect from text
  POST /jobs/{id}/render        - Render stem with effects
  GET  /jobs                    - List jobs
  GET  /jobs/{id}               - Get job status/results
  GET  /jobs/{id}/download/{stem} - Download a stem
  GET  /jobs/{id}/download_all  - Download all stems as ZIP
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
from concurrent.futures import ProcessPoolExecutor
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
import uvicorn

# Process pool for CPU-bound analysis (bypasses GIL)
analysis_pool: ProcessPoolExecutor = None

# Constants
SERVER_VERSION = "2.6.0"
RESULTS_DIR = Path("/workspace/results")
EFFECTS_DIR = Path("/workspace/effects")
SAMPLE_RATE = 44100
STEMS = ["drums", "bass", "other", "vocals"]

# Cleanup settings
MAX_RESULTS_AGE_HOURS = 24  # Delete results older than this
MAX_RESULTS_SIZE_GB = 5.0   # Delete oldest when exceeding this
MAX_JOB_HISTORY = 50        # Keep this many jobs in memory

# Ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EFFECTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="DJ Gizmo Server", version=SERVER_VERSION)

# Global models (Demucs loaded at startup, Stable Audio lazy-loaded)
demucs_model = None
stable_audio_model = None
stable_audio_config = None


def load_demucs_model():
    """Load the Demucs model."""
    global demucs_model
    if demucs_model is None:
        from demucs.pretrained import get_model
        print("Loading htdemucs model...")
        demucs_model = get_model('htdemucs')
        demucs_model.cuda()
        demucs_model.eval()
        print(f"Demucs loaded on {next(demucs_model.parameters()).device}")
    return demucs_model


def load_stable_audio_model():
    """Load the Stable Audio Open model (lazy, on first use)."""
    global stable_audio_model, stable_audio_config
    if stable_audio_model is None:
        import torch
        from stable_audio_tools import get_pretrained_model

        print("Loading Stable Audio Open model (first use, downloading ~3GB)...")
        stable_audio_model, stable_audio_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
        stable_audio_model = stable_audio_model.to("cuda")
        print("Stable Audio Open loaded")
    return stable_audio_model, stable_audio_config


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
    stage: str = "queued"  # queued, loading, separating, saving, analyzing, complete
    num_segments: int = 10  # Number of segment boundaries to detect


@dataclass
class EffectJob:
    id: str
    status: JobStatus
    created_at: datetime
    prompt: str
    duration: float = 5.0
    negative_prompt: str = "low quality, noise, distortion"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    output_path: Optional[str] = None
    progress: float = 0.0
    stage: str = "queued"  # queued, loading, generating, saving, complete


class EffectJobQueue:
    def __init__(self, max_history: int = MAX_JOB_HISTORY):
        self.jobs: dict[str, EffectJob] = {}
        self.queue: deque[str] = deque()
        self.max_history = max_history
        self._lock = asyncio.Lock()

    async def add(self, prompt: str, duration: float = 5.0, negative_prompt: str = "low quality, noise, distortion") -> EffectJob:
        job_id = str(uuid.uuid4())[:8]
        job = EffectJob(
            id=job_id,
            status=JobStatus.QUEUED,
            created_at=datetime.now(),
            prompt=prompt,
            duration=duration,
            negative_prompt=negative_prompt
        )
        async with self._lock:
            self.jobs[job_id] = job
            self.queue.append(job_id)
            self._cleanup_old_jobs()
        return job

    async def get(self, job_id: str) -> Optional[EffectJob]:
        return self.jobs.get(job_id)

    async def next(self) -> Optional[EffectJob]:
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
                if job.output_path and Path(job.output_path).exists():
                    Path(job.output_path).unlink(missing_ok=True)
                del self.jobs[job.id]

    def list_jobs(self, limit: int = 20) -> list[EffectJob]:
        jobs = sorted(self.jobs.values(),
                      key=lambda x: x.created_at, reverse=True)
        return jobs[:limit]

    def known_job_ids(self) -> set[str]:
        """Return set of job IDs currently tracked."""
        return set(self.jobs.keys())


class JobQueue:
    def __init__(self, max_history: int = MAX_JOB_HISTORY):
        self.jobs: dict[str, Job] = {}
        self.queue: deque[str] = deque()
        self.max_history = max_history
        self._lock = asyncio.Lock()

    async def add(self, filename: str, num_segments: int = 10) -> Job:
        job_id = str(uuid.uuid4())[:8]
        job = Job(
            id=job_id,
            status=JobStatus.QUEUED,
            created_at=datetime.now(),
            filename=filename,
            num_segments=num_segments
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


def get_effects_disk_usage() -> float:
    """Get total size of effects directory in GB."""
    total = 0
    if EFFECTS_DIR.exists():
        for f in EFFECTS_DIR.rglob('*'):
            if f.is_file():
                total += f.stat().st_size
    return total / (1024 ** 3)


def cleanup_orphaned_effects(known_ids: set[str]) -> int:
    """Delete effect files not tracked by effect queue. Returns count deleted."""
    deleted = 0
    if not EFFECTS_DIR.exists():
        return 0
    for item in EFFECTS_DIR.iterdir():
        if item.is_file() and item.suffix == '.wav':
            # Extract job_id from filename (format: {job_id}_{prompt}.wav)
            job_id = item.name.split('_')[0]
            if job_id not in known_ids:
                item.unlink(missing_ok=True)
                deleted += 1
    return deleted


def cleanup_old_effects(max_age_hours: float = MAX_RESULTS_AGE_HOURS) -> int:
    """Delete effect files older than max_age_hours. Returns count deleted."""
    deleted = 0
    if not EFFECTS_DIR.exists():
        return 0
    cutoff = time.time() - (max_age_hours * 3600)
    for item in EFFECTS_DIR.iterdir():
        if item.is_file() and item.suffix == '.wav':
            try:
                mtime = item.stat().st_mtime
                if mtime < cutoff:
                    item.unlink(missing_ok=True)
                    deleted += 1
            except OSError:
                pass
    return deleted


def run_all_cleanups(known_ids: set[str], known_effect_ids: set[str] = None) -> dict:
    """Run all cleanup routines. Returns summary."""
    if known_effect_ids is None:
        known_effect_ids = set()
    return {
        "orphaned_deleted": cleanup_orphaned_results(known_ids),
        "expired_deleted": cleanup_old_results(),
        "quota_deleted": cleanup_by_disk_quota(),
        "disk_usage_gb": round(get_results_disk_usage(), 2),
        "effects_orphaned_deleted": cleanup_orphaned_effects(known_effect_ids),
        "effects_expired_deleted": cleanup_old_effects(),
        "effects_disk_usage_gb": round(get_effects_disk_usage(), 2)
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
effect_queue = EffectJobQueue()


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


def detect_segments(y, sr, num_segments=10) -> list:
    """Detect structural segment boundaries using spectral clustering."""
    # Compute chromagram for harmonic content
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

    # Compute MFCCs for timbral content
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Stack features
    features = np.vstack([chroma, mfcc])

    # Compute self-similarity matrix
    rec = librosa.segment.recurrence_matrix(features, mode='affinity', sym=True)

    # Detect boundaries using spectral clustering
    bounds = librosa.segment.agglomerative(features, num_segments)

    # Convert frame indices to times
    bound_times = librosa.frames_to_time(bounds, sr=sr)

    # Add start and end times
    duration = len(y) / sr
    boundaries = [0.0] + [round(float(t), 3) for t in bound_times] + [round(duration, 3)]

    # Remove duplicates and sort
    boundaries = sorted(list(set(boundaries)))

    return boundaries


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


def analyze_audio_file(file_path: str, include_waveform: bool = True, num_segments: int = 10) -> dict:
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

    # Segment boundaries (structural changes)
    segments = detect_segments(y, sr, num_segments)

    result = {
        "duration": round(duration, 3),
        "bpm": round(bpm, 1),
        "key": key,
        "beats": [round(b, 3) for b in beats],
        "downbeats": [round(d, 3) for d in downbeats],
        "onsets": [round(o, 3) for o in onsets],
        "segments": segments,
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
                               started_at=datetime.now(),
                               stage="loading")
        await asyncio.sleep(0)  # Yield to event loop for SSE

        # Load model
        demucs_model = load_demucs_model()
        from demucs.apply import apply_model

        # Save temp file for ffmpeg conversion (handles m4a, mp3, etc.)
        temp_input = RESULTS_DIR / f"{job.id}_input.tmp"
        temp_wav = RESULTS_DIR / f"{job.id}_input.wav"

        temp_input.write_bytes(audio_data)

        # Convert to WAV using ffmpeg (run in thread to not block event loop)
        result = await asyncio.to_thread(subprocess.run, [
            'ffmpeg', '-y', '-i', str(temp_input),
            '-ar', str(SAMPLE_RATE), '-ac', '2',
            str(temp_wav)
        ], capture_output=True)

        temp_input.unlink()  # Clean up temp input

        if not temp_wav.exists():
            raise RuntimeError(f"FFmpeg conversion failed: {result.stderr.decode()}")

        # Load audio (keep temp_wav for analysis later)
        audio, sr = await asyncio.to_thread(sf.read, str(temp_wav))

        # Convert to (channels, samples)
        if audio.ndim == 1:
            audio = np.stack([audio, audio])
        else:
            audio = audio.T

        duration = audio.shape[1] / sr
        await job_queue.update(job.id, duration=duration, stage="separating")
        await asyncio.sleep(0)  # Yield to event loop for SSE

        # Convert to tensor
        waveform = torch.tensor(audio, dtype=torch.float32, device='cuda')

        # Run separation (in thread to allow SSE updates)
        print(f"Separating {duration:.1f}s of audio...")
        def run_separation():
            with torch.no_grad():
                return apply_model(demucs_model, waveform.unsqueeze(0), progress=True)[0]
        sources = await asyncio.to_thread(run_separation)

        await job_queue.update(job.id, stage="saving")
        await asyncio.sleep(0)  # Yield to event loop for SSE

        # Save results
        result_dir = RESULTS_DIR / job.id
        result_dir.mkdir(parents=True, exist_ok=True)

        def save_stems():
            for i, stem_name in enumerate(STEMS):
                stem_audio = sources[i].T.cpu().numpy()
                sf.write(str(result_dir / f"{stem_name}.wav"), stem_audio, sr)
        await asyncio.to_thread(save_stems)

        # Pre-create ZIP file (no compression - WAV doesn't compress well)
        base_name = Path(job.filename).stem
        zip_path = result_dir / "stems.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zf:
            for stem_name in STEMS:
                stem_file = result_dir / f"{stem_name}.wav"
                zf.write(stem_file, f"{base_name}_{stem_name}.wav")

        await job_queue.update(job.id, stage="analyzing")
        await asyncio.sleep(0)  # Yield to event loop for SSE

        # Auto-analyze full mix and stems IN PARALLEL using ProcessPool (bypasses GIL)
        print(f"Analyzing audio (5 files in parallel via multiprocessing)...")
        analysis = {}
        loop = asyncio.get_event_loop()

        # Build list of (name, file_path) to analyze
        files_to_analyze = [
            ("mix", str(temp_wav)),
            ("drums", str(result_dir / "drums.wav")),
            ("bass", str(result_dir / "bass.wav")),
            ("other", str(result_dir / "other.wav")),
            ("vocals", str(result_dir / "vocals.wav")),
        ]

        # Submit all to process pool (true parallelism, bypasses GIL)
        async def analyze_in_process(name: str, file_path: str):
            try:
                result = await loop.run_in_executor(
                    analysis_pool,
                    analyze_audio_file,
                    file_path,
                    True,  # include_waveform
                    job.num_segments
                )
                return (name, result)
            except Exception as e:
                return (name, {"error": str(e)})

        # Run all 5 analyses in parallel across separate processes
        tasks = [analyze_in_process(name, path) for name, path in files_to_analyze]
        results = await asyncio.gather(*tasks)

        # Collect results into analysis dict
        for name, result in results:
            analysis[name] = result

        # Clean up temp wav
        temp_wav.unlink()

        await job_queue.update(job.id,
                               status=JobStatus.COMPLETED,
                               completed_at=datetime.now(),
                               result_path=str(result_dir),
                               analysis=analysis,
                               stage="complete")

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


async def process_effect_queue():
    """Background worker to process effect generation queue."""
    while True:
        job = await effect_queue.next()
        if job:
            await generate_effect_job(job)
        await asyncio.sleep(0.5)


def run_effect_generation(prompt: str, duration: float, negative_prompt: str):
    """Blocking function to run effect generation on GPU. Called via asyncio.to_thread()."""
    from stable_audio_tools.inference.generation import generate_diffusion_cond

    model, model_config = load_stable_audio_model()
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]

    conditioning = [{
        "prompt": prompt,
        "seconds_start": 0,
        "seconds_total": duration
    }]

    print(f"Generating effect: '{prompt}' ({duration}s)...")

    with torch.no_grad():
        output = generate_diffusion_cond(
            model,
            steps=100,
            cfg_scale=7,
            conditioning=conditioning,
            negative_conditioning=[{"prompt": negative_prompt, "seconds_start": 0, "seconds_total": duration}],
            sample_size=sample_size,
            sample_rate=sample_rate,
            device="cuda"
        )

    # output shape is (batch, channels, samples) - get first item
    audio = output[0].cpu().numpy()

    # Trim to requested duration
    target_samples = int(sample_rate * duration)
    audio = audio[:, :target_samples]

    # Transpose: (channels, samples) -> (samples, channels) for soundfile
    audio = audio.T

    # Normalize to prevent clipping
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val * 0.95

    return audio, sample_rate


async def generate_effect_job(job: EffectJob):
    """Process a single effect generation job."""
    try:
        await effect_queue.update(job.id,
                                  status=JobStatus.PROCESSING,
                                  started_at=datetime.now(),
                                  stage="loading")

        await effect_queue.update(job.id, stage="generating", progress=0.1)

        # Run generation in thread to avoid blocking event loop
        audio, sample_rate = await asyncio.to_thread(
            run_effect_generation,
            job.prompt,
            job.duration,
            job.negative_prompt
        )

        await effect_queue.update(job.id, stage="saving", progress=0.9)

        # Build filename from prompt
        safe_prompt = "".join(c if c.isalnum() or c in " -_" else "" for c in job.prompt)
        safe_prompt = safe_prompt.replace(" ", "_")[:50]
        output_filename = f"{job.id}_{safe_prompt}.wav"
        output_path = EFFECTS_DIR / output_filename

        # Write to file
        await asyncio.to_thread(sf.write, output_path, audio, sample_rate)

        print(f"Generated effect: {output_filename} ({sample_rate}Hz)")

        await effect_queue.update(job.id,
                                  status=JobStatus.COMPLETED,
                                  completed_at=datetime.now(),
                                  output_path=str(output_path),
                                  progress=1.0,
                                  stage="complete")

    except Exception as e:
        import traceback
        traceback.print_exc()
        await effect_queue.update(job.id,
                                  status=JobStatus.FAILED,
                                  completed_at=datetime.now(),
                                  error=str(e),
                                  stage="failed")


@app.on_event("startup")
async def startup():
    global analysis_pool

    # Run cleanup on startup (orphaned files from previous sessions)
    cleanup_result = run_all_cleanups(job_queue.known_job_ids(), effect_queue.known_job_ids())
    print(f"Startup cleanup: {cleanup_result}")

    # Pre-load model
    load_demucs_model()

    # Create process pool for parallel analysis (bypasses GIL)
    analysis_pool = ProcessPoolExecutor(max_workers=5)
    print("Analysis process pool ready (5 workers)")

    # Start queue processors
    asyncio.create_task(process_queue())
    asyncio.create_task(process_effect_queue())


@app.on_event("shutdown")
async def shutdown():
    """Clean up process pool to prevent zombie processes."""
    global analysis_pool
    if analysis_pool:
        print("Shutting down analysis process pool...")
        analysis_pool.shutdown(wait=False, cancel_futures=True)
        print("Process pool closed")


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
        "effect_queue_size": len(effect_queue.queue),
        "cuda_available": torch.cuda.is_available()
    }


@app.post("/cleanup")
async def cleanup():
    """Manually trigger cleanup of old results and effects."""
    result = run_all_cleanups(job_queue.known_job_ids(), effect_queue.known_job_ids())
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
async def separate_audio(
    file: UploadFile = File(...),
    num_segments: int = 10
):
    """Upload and separate audio into stems.

    Args:
        file: Audio file to process
        num_segments: Number of segment boundaries to detect (default 10)
    """
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
    job = await job_queue.add(file.filename, num_segments=num_segments)

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
        # Calculate processing time
        if j.completed_at and j.started_at:
            process_time = (j.completed_at - j.started_at).total_seconds()
        elif j.started_at:
            process_time = (datetime.now() - j.started_at).total_seconds()
        else:
            process_time = 0

        job_data = {
            "id": j.id,
            "filename": j.filename,
            "status": j.status.value,
            "stage": j.stage,
            "duration": j.duration,
            "process_time": round(process_time, 1),
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
                # Calculate processing time
                if j.completed_at and j.started_at:
                    process_time = (j.completed_at - j.started_at).total_seconds()
                elif j.started_at:
                    process_time = (datetime.now() - j.started_at).total_seconds()
                else:
                    process_time = 0

                job_data = {
                    "id": j.id,
                    "filename": j.filename,
                    "status": j.status.value,
                    "stage": j.stage,
                    "duration": j.duration,
                    "process_time": round(process_time, 1),
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
        "stage": job.stage,
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


def render_stem(
    file_path: str,
    original_bpm: float,
    target_bpm: float = None,
    pitch_shift: int = None,
    start_time: float = None,
    end_time: float = None
) -> tuple[np.ndarray, int]:
    """
    Render a stem with time-stretch, pitch-shift, and/or cut.
    Returns (audio_array, sample_rate).
    """
    # Load audio
    y, sr = librosa.load(file_path, sr=None, mono=False)

    # Handle stereo vs mono
    is_stereo = y.ndim == 2
    if is_stereo:
        # Process each channel
        channels = [y[0], y[1]]
    else:
        channels = [y]

    processed_channels = []
    for channel in channels:
        processed = channel

        # 1. Cut first (more efficient to process less audio)
        if start_time is not None or end_time is not None:
            start_sample = int((start_time or 0) * sr)
            end_sample = int((end_time or len(processed) / sr) * sr)
            processed = processed[start_sample:end_sample]

        # 2. Time-stretch to target BPM
        if target_bpm and original_bpm and target_bpm != original_bpm:
            # rate > 1 speeds up, rate < 1 slows down
            rate = target_bpm / original_bpm
            processed = librosa.effects.time_stretch(processed, rate=rate)

        # 3. Pitch shift
        if pitch_shift and pitch_shift != 0:
            processed = librosa.effects.pitch_shift(processed, sr=sr, n_steps=pitch_shift)

        processed_channels.append(processed)

    # Reconstruct stereo if needed
    if is_stereo:
        result = np.vstack(processed_channels)
    else:
        result = processed_channels[0]

    return result, sr


@app.post("/jobs/{job_id}/render")
async def render_stem_endpoint(
    job_id: str,
    stem: str = "vocals",
    bpm: float = None,
    pitch: int = None,
    start: float = None,
    end: float = None
):
    """
    Render a stem with time-stretch, pitch-shift, and/or cut.

    Args:
        job_id: Job ID from separation
        stem: Stem to render (drums, bass, other, vocals)
        bpm: Target BPM (time-stretches to this tempo)
        pitch: Semitones to shift (+/- 12 = one octave)
        start: Start time in seconds (cuts before this)
        end: End time in seconds (cuts after this)

    Returns:
        Rendered WAV file
    """
    job = await job_queue.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(400, "Job not complete")

    # Validate stem
    if stem not in STEMS:
        raise HTTPException(400, f"Invalid stem. Use: {', '.join(STEMS)}")

    result_path = Path(job.result_path)
    stem_file = result_path / f"{stem}.wav"

    if not stem_file.exists():
        raise HTTPException(404, f"Stem file not found: {stem}")

    # Get original BPM from analysis
    original_bpm = None
    if job.analysis and "mix" in job.analysis:
        original_bpm = job.analysis["mix"].get("bpm")

    if bpm and not original_bpm:
        raise HTTPException(400, "Cannot time-stretch: original BPM not available")

    # Render in process pool (CPU-intensive)
    loop = asyncio.get_event_loop()
    try:
        audio, sr = await loop.run_in_executor(
            analysis_pool,
            render_stem,
            str(stem_file),
            original_bpm,
            bpm,
            pitch,
            start,
            end
        )
    except Exception as e:
        raise HTTPException(500, f"Render failed: {str(e)}")

    # Write to temp file and return
    base_name = Path(job.filename).stem

    # Build descriptive filename
    parts = [base_name, stem]
    if bpm:
        parts.append(f"{int(bpm)}bpm")
    if pitch:
        sign = "+" if pitch > 0 else ""
        parts.append(f"{sign}{pitch}st")
    if start is not None or end is not None:
        start_str = f"{start:.1f}" if start else "0"
        end_str = f"{end:.1f}" if end else "end"
        parts.append(f"{start_str}-{end_str}s")

    output_filename = "_".join(parts) + ".wav"

    # Write to BytesIO and return
    buffer = io.BytesIO()

    # Handle stereo vs mono for soundfile
    if audio.ndim == 2:
        audio_to_write = audio.T  # soundfile expects (samples, channels)
    else:
        audio_to_write = audio

    sf.write(buffer, audio_to_write, sr, format='WAV')
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="audio/wav",
        headers={"Content-Disposition": f'attachment; filename="{output_filename}"'}
    )


# --- Sound Effect Generation (Stable Audio Open) ---

@app.post("/generate_effect")
async def generate_effect(
    prompt: str,
    duration: float = 5.0,
    negative_prompt: str = "low quality, noise, distortion"
):
    """
    Queue a sound effect generation job using Stable Audio Open.

    Args:
        prompt: Text description of the sound effect (e.g., "Air horn blast. High-quality. Stereo.")
        duration: Duration in seconds (1-47, default 5)
        negative_prompt: What to avoid in generation (default: "low quality, noise, distortion")

    Returns:
        Job info with ID to track status
    """
    # Validate duration (Stable Audio supports up to 47 seconds)
    if duration < 1 or duration > 47:
        raise HTTPException(400, "Duration must be between 1 and 47 seconds")

    # Queue the job
    job = await effect_queue.add(prompt, duration, negative_prompt)

    return {
        "job_id": job.id,
        "status": job.status.value,
        "prompt": job.prompt,
        "duration": job.duration,
        "created_at": job.created_at.isoformat(),
        "message": "Effect generation queued. Poll /effects/{job_id} for status."
    }


@app.get("/effects")
async def list_effects(limit: int = 20):
    """List recent effect generation jobs."""
    jobs = effect_queue.list_jobs(limit)
    return {
        "effects": [
            {
                "job_id": j.id,
                "status": j.status.value,
                "prompt": j.prompt,
                "duration": j.duration,
                "stage": j.stage,
                "progress": j.progress,
                "created_at": j.created_at.isoformat(),
                "completed_at": j.completed_at.isoformat() if j.completed_at else None,
                "error": j.error
            }
            for j in jobs
        ],
        "queue_size": len(effect_queue.queue)
    }


@app.get("/effects/{job_id}")
async def get_effect_status(job_id: str):
    """Get status of an effect generation job."""
    job = await effect_queue.get(job_id)
    if not job:
        raise HTTPException(404, f"Effect job {job_id} not found")

    return {
        "job_id": job.id,
        "status": job.status.value,
        "prompt": job.prompt,
        "duration": job.duration,
        "stage": job.stage,
        "progress": job.progress,
        "created_at": job.created_at.isoformat(),
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "error": job.error,
        "download_url": f"/effects/{job.id}/download" if job.status == JobStatus.COMPLETED else None
    }


@app.get("/effects/{job_id}/download")
async def download_effect(job_id: str):
    """Download a generated sound effect."""
    job = await effect_queue.get(job_id)
    if not job:
        raise HTTPException(404, f"Effect job {job_id} not found")

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(400, f"Effect not ready. Status: {job.status.value}, Stage: {job.stage}")

    if not job.output_path or not Path(job.output_path).exists():
        raise HTTPException(404, "Effect file not found")

    return FileResponse(
        job.output_path,
        media_type="audio/wav",
        filename=Path(job.output_path).name
    )


@app.get("/effects/examples")
async def list_effect_examples():
    """Return example prompts for sound effect generation."""
    return {
        "examples": [
            # DJ / Music
            "air horn blast",
            "vinyl scratch",
            "record rewind",
            "bass drop",
            "synth riser",
            "drum roll",
            "cymbal crash",
            "dj siren",
            # Impacts
            "explosion",
            "thunder crack",
            "glass breaking",
            "door slam",
            # Ambient
            "crowd cheering",
            "applause",
            "rain on window",
            "wind howling",
            # Funny
            "fart sound",
            "cartoon boing",
            "sad trombone",
            "wrong answer buzzer",
        ]
    }


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
