import os
import glob
import tempfile
import numpy as np
import soundfile as sf
import librosa
from scipy import signal
from audiosr import build_model, super_resolution
from typing import Literal

CHUNK_SEC = 5.12
OVERLAP_SEC = 0.64  # crossfade length between chunks; only matters for files > CHUNK_SEC
WORK_SR = 48000  # AudioSR's internal working rate

def to_numpy(x):
    """Coerce AudioSR's output to a plain numpy.ndarray. Some versions/forks
    return a torch.Tensor (or a numpy subclass) instead of a bare ndarray,
    which breaks librosa's strict type check downstream."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)

def pad(x, target_samples):
    """Fill x up to target_samples with silence instead of looping."""

    if len(x) >= target_samples:
        return x[:target_samples]

    pad_length = target_samples - len(x)

    if x.ndim == 1:
        padding = np.zeros(pad_length, dtype=x.dtype)
    else:
        padding = np.zeros((pad_length, x.shape[1]), dtype=x.dtype)

    return np.concatenate((x, padding), axis=0)

def match_loudness_envelope(orig, orig_sr, output, output_sr, window_sec=0.05,
                             max_gain=4.0, silence_floor=1e-4):
    """Compare the ORIGINAL input's loudness envelope against the finished,
    already-blended output at regular time intervals, and rescale the output
    to track it. Run after crossfading rather than per-chunk, so it also
    smooths out any drift the model introduced inside a chunk or across the
    blend itself, not just a single flat correction per chunk."""

    def windowed_rms(x, sr):
        win = max(1, int(round(window_sec * sr)))
        n = int(np.ceil(len(x) / win))
        rms = np.zeros(n)
        centers = np.zeros(n)
        for i in range(n):
            seg = x[i * win:(i + 1) * win]
            rms[i] = np.sqrt(np.mean(seg.astype(np.float64) ** 2)) if len(seg) else 0.0
            centers[i] = (i + 0.5) * win / sr
        return centers, rms

    orig_centers, orig_rms = windowed_rms(orig, orig_sr)
    _, out_rms = windowed_rms(output, output_sr)

    n = min(len(orig_rms), len(out_rms))
    orig_centers, orig_rms, out_rms = orig_centers[:n], orig_rms[:n], out_rms[:n]

    eps = 1e-9
    gains = np.where(out_rms > eps, orig_rms / (out_rms + eps), 1.0)
    gains = np.clip(gains, 1.0 / max_gain, max_gain)
    # don't chase gain during true silence -- there's nothing meaningful to
    # match, and it risks amplifying the model's residual noise floor
    gains = np.where(orig_rms < silence_floor, 1.0, gains)

    sample_times = np.arange(len(output)) / output_sr
    gain_curve = np.interp(sample_times, orig_centers, gains,
                            left=gains[0], right=gains[-1])
    return output * gain_curve

def stitch_with_crossfade(chunks, starts_sec, overlap):
    """Overlap-add chunks (each at WORK_SR) using a linear crossfade at each seam."""
    starts_48k = [int(round(s * WORK_SR)) for s in starts_sec]
    total_len = max(s + len(c) for s, c in zip(starts_48k, chunks))
    out = np.zeros(total_len, dtype=np.float64)
    overlap_samples = int(round(overlap * WORK_SR))

    for i, (start, chunk) in enumerate(zip(starts_48k, chunks)):
        chunk = chunk.astype(np.float64).copy()
        if i > 0 and overlap_samples > 0:
            ramp_in = np.linspace(0.0, 1.0, overlap_samples)
            ramp_out = 1.0 - ramp_in
            out[start:start + overlap_samples] *= ramp_out
            chunk[:overlap_samples] *= ramp_in
        out[start:start + len(chunk)] += chunk

    return out

def run_single_chunk(chunk_data, sr, model, guidance_scale, ddim_steps):
    """Pad one chunk to exactly CHUNK_SEC, run AudioSR, return the raw 48kHz waveform."""

    target_samples = int(round(CHUNK_SEC * sr))
    if len(chunk_data) < target_samples:
        chunk_data = pad(chunk_data, target_samples)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    sf.write(tmp_path, chunk_data, sr)

    try:
        waveform = super_resolution(
            model,
            tmp_path,
            seed=42,
            guidance_scale=guidance_scale,
            ddim_steps=ddim_steps,
        )
    finally:
        os.remove(tmp_path)

    waveform = to_numpy(waveform).squeeze()
    return waveform

def process_file(path, model, overlap, guidance_scale, ddim_steps):
    input_waveform, input_sr = sf.read(path)

    if input_sr != WORK_SR:
        input_waveform = librosa.resample(input_waveform, orig_sr=input_sr, target_sr=WORK_SR)

    input_sample_count = len(input_waveform)

    chunk_samples = int(round(CHUNK_SEC * WORK_SR))

    if len(input_waveform) <= chunk_samples:
        # short file: single pass, no chunking needed
        output_waveform = run_single_chunk(input_waveform, WORK_SR, model, guidance_scale, ddim_steps)
    else:
        # long file: overlapping windows, crossfaded back together
        overlap_samples = int(round(overlap * WORK_SR))
        hop_samples = chunk_samples - overlap_samples

        starts = list(range(0, len(input_waveform) - chunk_samples + 1, hop_samples))
        if not starts:
            starts = [0]
        # ensure the final chunk is a full, real window ending exactly at
        # the file's end, instead of a hop-derived start that could leave
        # only a sliver of real audio before padding (which breaks AudioSR's
        # cutoff-detection preprocessing on an almost-all-silence buffer)
        last_full_start = len(input_waveform) - chunk_samples
        if starts[-1] != last_full_start:
            starts.append(last_full_start)

        chunks = []
        starts_sec = []
        for start in starts:
            chunk_data = input_waveform[start:start + chunk_samples]
            try:
                chunks.append(run_single_chunk(chunk_data, WORK_SR, model, guidance_scale, ddim_steps))
            except Exception as e:
                raise RuntimeError(
                    f"AudioSR failed on chunk starting at {start / WORK_SR:.2f}s "
                    f"(of {input_sample_count:.2f}s total) in {path}: {e}"
                ) from e
            starts_sec.append(start / WORK_SR)

        output_waveform = stitch_with_crossfade(chunks, starts_sec, overlap)

    # trim back down to the original duration (waveform is at 48kHz)
    output_waveform = output_waveform[:input_sample_count]

    output_waveform = match_loudness_envelope(input_waveform, WORK_SR, output_waveform, WORK_SR)

    return output_waveform, WORK_SR

def upsample(input_folder:str, output_folder:str, model: Literal["basic", "speech"], overlap=0.64, guidance_scale=3.5, ddim_steps=50):
    wav_paths = sorted(set(
        glob.glob(os.path.join(input_folder, "*.wav"))
        + glob.glob(os.path.join(output_folder, "*.WAV"))
    ))

    if not wav_paths:
        print(f"No .wav files found in {input_folder}")
        return

    os.makedirs(output_folder, exist_ok=True)

    model_instance = build_model(model_name=model)

    for path in wav_paths:
        print(f"Processing ({model}): {path}")
        try:
            audio, sr = process_file(path, model_instance, overlap, guidance_scale, ddim_steps)
        except Exception as e:
            print(f"  FAILED: {e}")
            print(f"  Skipping {path}, continuing with the rest of the batch.")
            continue

        out_path = os.path.join(output_folder, os.path.basename(path))
        sf.write(out_path, audio, sr, subtype="PCM_16")

    print("Done.")