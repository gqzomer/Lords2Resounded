from df import enhance, init_df
from scipy import signal
import soundfile as sf
import numpy as np
import glob
import os
import torch

model, df_state, _ = init_df()

def load_wav_at_sr(input_path: str, sample_rate=48000):
    audio, sr = sf.read(input_path, dtype='float32')

    #upsample audio
    audio = signal.resample_poly(audio, sample_rate, sr)

    #normalize peaks
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = (audio / max_val) * 0.8912

    return audio, sample_rate

def preprocess_wav(input_path: str, output_path: str):
    audio, sr = load_wav_at_sr(input_path=input_path)

    audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()
    enhanced_tensor = enhance(model, df_state, audio_tensor)
    enhanced_audio = enhanced_tensor.squeeze(0).cpu().numpy()

    sf.write(output_path, samplerate=sr, data=enhanced_audio, subtype="PCM_16")

def preprocess_folder(input_folder:str, output_folder:str):
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
            preprocess_wav(path, out_path)
        except Exception as e:
            print(f"  FAILED: {e}")
            print(f"  Skipping {path}, continuing with the rest of the batch.")
            continue
