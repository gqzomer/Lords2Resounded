import dpdfnet
from scipy import signal
import soundfile as sf
import numpy as np
import glob
import os

def load_wav_at_sr(input_path: str, sample_rate=11025):
    audio, sr = sf.read(input_path, dtype='float32')

    if (sr != sample_rate):
        print(f"input sample rate {sr}Hz differs from requested sample rate. resampling file to {sample_rate}Hz")
        audio = signal.resample_poly(audio, sample_rate, sr)

    # normalize peaks
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = (audio / max_val) * 0.8912

    return audio, sample_rate

def compute_energy_retention(original, enhanced, sr, window_sec=0.05,
                              silence_floor=1e-4, drop_threshold_db=20):
    """Fraction of originally-active windows that survived enhancement
    without a catastrophic drop. A low fraction means the model likely
    collapsed real speech content rather than just removing noise."""
    win = max(1, int(round(window_sec * sr)))
    n = min(len(original), len(enhanced)) // win
    if n == 0:
        return 1.0  # too short to judge meaningfully

    active = 0
    retained = 0
    for i in range(n):
        seg_o = original[i*win:(i+1)*win].astype(np.float64)
        seg_e = enhanced[i*win:(i+1)*win].astype(np.float64)
        rms_o = np.sqrt(np.mean(seg_o**2))
        rms_e = np.sqrt(np.mean(seg_e**2))
        if rms_o > silence_floor:
            active += 1
            drop_db = 20 * np.log10(rms_o / (rms_e + 1e-12))
            if drop_db < drop_threshold_db:
                retained += 1

    return retained / active if active else 1.0

def preprocess_wav(input_path: str, output_path: str, retention_threshold=0.80):
    audio, sr = load_wav_at_sr(input_path=input_path)

    enhanced = dpdfnet.enhance(audio, sample_rate=sr, model="dpdfnet8")

    retention = compute_energy_retention(audio, enhanced, sr)
    if retention < retention_threshold:
        print(f"  {os.path.basename(input_path)}: only {retention:.0%} of "
              f"active content retained, falling back to dpdfnet8_8khz")
        enhanced = dpdfnet.enhance(audio, sample_rate=sr, model="dpdfnet8_8khz")

    sf.write(output_path, samplerate=sr, data=enhanced)

def preprocess_folder(input_folder:str, output_folder:str, retention_threshold:float):
    wav_paths = sorted(set(
        glob.glob(os.path.join(input_folder, "*.wav"))
        + glob.glob(os.path.join(input_folder, "*.WAV"))
    ))

    if not wav_paths:
        print(f"No .wav files found in {input_folder} — skipping.")
        return

    os.makedirs(output_folder, exist_ok=True)

    for path in wav_paths:
        print(f"preprocessing: {path}")
        try:
            out_path = os.path.join(output_folder, os.path.basename(path))
            preprocess_wav(path, out_path, retention_threshold)
        except Exception as e:
            print(f"  FAILED: {e}")
            print(f"  Skipping {path}, continuing with the rest of the batch.")
            continue
