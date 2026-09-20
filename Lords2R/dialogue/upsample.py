from pathlib import Path
import dpdfnet
from Lords2R.utils import get_lava_model, process_listed_wavs, write_processed_audio

FILE_LIST = Path(__file__).with_name("files.txt")

def process_wav(input_path: str, output_path: str, cutoff: int = 5300):
    lava_model = get_lava_model()
    input_audio, input_sr = lava_model.load_audio(input_path, cutoff=cutoff)

    output_audio = lava_model.enhance(
        input_audio,
        enhance=True,
        denoise=False
    ).cpu().numpy().squeeze()

    output_audio = dpdfnet.enhance(output_audio, sample_rate=48000, model="dpdfnet8_48khz_hr")

    write_processed_audio(output_path, output_audio, input_audio.cpu().numpy().reshape(-1), cutoff)


def upsample(input_folder: str, cutoff: int = 5300):
    process_listed_wavs(input_folder, FILE_LIST, process_wav, cutoff)
