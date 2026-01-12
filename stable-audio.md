# Stable Audio Integration Notes

## Models

### Stable Audio Open 1.0 (Full)
- **HuggingFace:** `stabilityai/stable-audio-open-1.0`
- **Params:** 1.2B
- **Max duration:** 47 seconds
- **Sample rate:** 44.1kHz stereo
- **Steps:** 100 (default)
- **Generation time:** ~65s on GB10
- **Gated:** Yes - requires HuggingFace token + license acceptance

### Stable Audio Open Small
- **HuggingFace:** `stabilityai/stable-audio-open-small`
- **Params:** 497M
- **Max duration:** 11 seconds
- **Sample rate:** 44.1kHz stereo
- **Steps:** 8 (rectified flow, no sampler_type needed)
- **cfg_scale:** 1.0 (not 7.0 like full model)
- **Generation time:** ~1.1-1.2s on GB10
- **Gated:** Yes - same token, separate license acceptance

## Installation

Both models use `stable-audio-tools` library. Surgical install required on NGC containers:

```bash
# Install without deps (avoids pandas==2.0.2 ARM64 issue)
pip install --no-deps git+https://github.com/Stability-AI/stable-audio-tools.git

# Manual deps
pip install einops-exts alias-free-torch local-attention vector-quantize-pytorch k-diffusion huggingface_hub
```

## HuggingFace Authentication

Full model is gated. Inside container:
```bash
huggingface-cli login
# Enter token from https://huggingface.co/settings/tokens
```

Must also accept license at model page before first download.

## Generation Code

### Full Model (1.0)
```python
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

model, config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
model = model.to("cuda")

output = generate_diffusion_cond(
    model,
    steps=100,
    cfg_scale=7,
    conditioning=[{"prompt": "...", "seconds_start": 0, "seconds_total": duration}],
    sample_size=config["sample_size"],  # 2097152
    sample_rate=config["sample_rate"],  # 44100
    device="cuda"
)
```

### Small Model
```python
model, config = get_pretrained_model("stabilityai/stable-audio-open-small")
model = model.to("cuda")

output = generate_diffusion_cond(
    model,
    steps=8,
    cfg_scale=1.0,
    conditioning=[{"prompt": "...", "seconds_start": 0, "seconds_total": duration}],
    sample_size=config["sample_size"],
    sample_rate=config["sample_rate"],
    device="cuda"
    # Note: Small model uses rectified flow, no sampler_type needed
)
```

## Prompting Tips

- Keep prompts short (1-5 words for effects)
- Add quality boosters: `"High-quality."`, `"Stereo."`, `"44.1kHz"`
- For MIDI-sounding output, add `"Live"` or `"Acoustic"`

Good prompts:
```
"Air horn blast. High-quality. Stereo."
"Vinyl scratch. High-quality."
"Cymbal crash. Stereo."
"Explosion boom. High-quality."
```

## API Design

```
POST /generate_effect
  ?prompt=...
  &duration=5.0
  &model=small|full  (default: small)
  &negative_prompt=...
```

- `small`: Fast (~5-10s), max 11s duration
- `full`: Slow (~65s), max 47s duration

## References

- [Stable Audio Open 1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0)
- [Stable Audio Open Small](https://huggingface.co/stabilityai/stable-audio-open-small)
- [stable-audio-tools GitHub](https://github.com/Stability-AI/stable-audio-tools)
- [Prompting Guide - Jordi Pons](https://www.jordipons.me/on-prompting-stable-audio/)
