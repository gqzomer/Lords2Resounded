"""Shared audio cleanup, model loading, and filename-list processing."""

from functools import lru_cache
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory

import numpy
import soundfile as sf
from scipy.ndimage import maximum_filter1d, median_filter, uniform_filter1d
from scipy.signal import butter, istft, sosfiltfilt, stft


@lru_cache(maxsize=1)
def get_lava_model():
    from LavaSR.model import LavaEnhance2

    return LavaEnhance2("YatharthS/LavaSR", "cpu")


def remove_peaks(audio, ceiling=0.8912):
    peak = numpy.max(numpy.abs(audio))
    if peak > ceiling:
        audio = audio * (ceiling / peak)
    return audio

def _find_10khz_clicks(audio):
    """
    Return click intervals in samples, for the pipeline's 48 kHz mono audio.
    
    This function was written by Codex after manual spectrogram analysis
    """
    # A 512-sample spectrum every millisecond resolves the narrow 10.1 kHz bursts.
    frequencies, _, spectrum = stft(audio, fs=48000, nperseg=512, noverlap=464)
    power = numpy.abs(spectrum) ** 2
    target = (frequencies >= 9500) & (frequencies <= 10700)
    neighbors = ((frequencies >= 8300) & (frequencies < 9500)) | (
        (frequencies > 10700) & (frequencies <= 11900)
    )
    band_power = power[target].mean(axis=0)
    side_power = power[neighbors].mean(axis=0)
    baseline = median_filter(band_power, size=81, mode="reflect")
    epsilon = numpy.finfo(numpy.float64).tiny
    temporal_ratio = band_power / numpy.maximum(baseline, epsilon)
    spectral_ratio = band_power / numpy.maximum(side_power, epsilon)
    # Convert Hann-window spectral power to mean square; ignore below -75 dBFS.
    audible = (2.0 / 1.5) * power[target].sum(axis=0) >= 10 ** (-75 / 10)

    # Require a 10 dB jump in time and 8 dB above neighboring frequencies.
    peaks = (temporal_ratio >= 10 ** (10 / 10)) & (spectral_ratio >= 10 ** (8 / 10)) & audible
    # Follow quieter tails down to 3 dB, keeping closely spaced clicks separate.
    tails = (temporal_ratio >= 10 ** (3 / 10)) & (spectral_ratio >= 10 ** (3 / 10)) & audible
    edges = numpy.diff(numpy.r_[False, tails, False].astype(numpy.int8))
    starts = numpy.flatnonzero(edges == 1)
    stops = numpy.flatnonzero(edges == -1)
    return [
        (start * 48, stop * 48)
        for start, stop in zip(starts, stops)
        if stop - start <= 30 and peaks[start:stop].any()  # At most 30 ms.
    ]


def _click_fade_mask(length, clicks):
    """
    Pad each click by 2 ms and fade its band attenuation in and out.
    
    this function was written by Codex
    """
    fade = 96  # 2 ms at 48 kHz.
    mask = numpy.zeros(length)
    for start, stop in clicks:
        left = max(0, start - fade)
        right = min(length, stop + fade)
        mask[left:right] = 1.0
        before = numpy.arange(max(0, left - fade), left)
        after = numpy.arange(right, min(length, right + fade))
        fade_in = 0.5 + 0.5 * numpy.cos(numpy.pi * (left - before) / fade)
        fade_out = 0.5 + 0.5 * numpy.cos(numpy.pi * (after - right) / fade)
        mask[before] = numpy.maximum(mask[before], fade_in)
        mask[after] = numpy.maximum(mask[after], fade_out)
    return mask


def suppress_lavasr_10khz_clicks(audio, attenuation_db=24.0):
    """
    Reduce LavaSR clicks in the pipeline's 48 kHz mono output.
    
    This function was written by Codex
    """
    if len(audio) < 512:
        return audio.copy()

    samples = audio.astype(numpy.float64)
    clicks = _find_10khz_clicks(samples)
    if not clicks:
        return audio.copy()

    mask = _click_fade_mask(len(samples), clicks)
    band_filter = butter(4, [9500, 10700], fs=48000, btype="bandpass", output="sos")
    click_band = sosfiltfilt(band_filter, samples)
    gain = 10 ** (-attenuation_db / 20)
    return (samples - (1 - gain) * mask * click_band).astype(audio.dtype)


def suppress_lavasr_hiss(audio, attenuation_db=18.0):
    """
    Reduce added high-frequency hiss in quiet parts of the 48 kHz mono WAVs.

    Use the original speech range to control attenuation above 4.5 kHz.
    This targets gaps and quiet tails; hiss underneath speech remains.
    
    this function was written by Codex
    """
    if len(audio) < 960:
        return audio.copy()

    samples = audio.astype(numpy.float64)
    speech_filter = butter(4, [200, 4500], fs=48000, btype="bandpass", output="sos")
    speech = sosfiltfilt(speech_filter, samples)
    speech_power = uniform_filter1d(speech ** 2, size=960)  # Average over 20 ms.
    speech_db = 10 * numpy.log10(numpy.maximum(speech_power, 1e-12))

    # Begin reducing hiss 25 dB below the clip's normal speech level.
    # Hold open around speech for 25 ms on either side to protect word endings.
    quiet_db = numpy.percentile(speech_db, 90) - 25
    speech_db = maximum_filter1d(speech_db, size=2401)
    quiet = numpy.clip((quiet_db - speech_db) / 12, 0, 1)
    quiet = uniform_filter1d(quiet, size=960)  # Smooth transitions to avoid pumping.

    hiss_filter = butter(4, 4500, fs=48000, btype="highpass", output="sos")
    hiss_band = sosfiltfilt(hiss_filter, samples)
    gain = 10 ** (-attenuation_db * quiet / 20)
    return (samples - (1 - gain) * hiss_band).astype(audio.dtype)


def balance_lavasr_high_band(audio, original, cutoff=5300):
    """
    Limit added highs using the 16 kHz mono input and the preserved output band.

    Compare energy per frequency bin just below the cutoff with the added band.
    Allow 3 dB of headroom and at most 24 dB of reduction. This is a conservative
    blend for these recordings, not an estimate of their lost original highs.

    this function was written by Codex
    """
    if len(audio) < 1536 or len(original) < 512:
        return audio.copy()

    # Both spectra use 32 ms windows and 8 ms steps. PSD scaling makes their
    # energy densities comparable despite the different sample rates.
    frequencies, times, spectrum = stft(
        audio.astype(numpy.float64), 48000, nperseg=1536, noverlap=1152, scaling="psd"
    )
    source_frequencies, source_times, source_spectrum = stft(
        original.astype(numpy.float64), 16000, nperseg=512, noverlap=384, scaling="psd"
    )
    power = numpy.abs(spectrum) ** 2
    source_power = numpy.abs(source_spectrum) ** 2

    def band_energy(frequencies, power, low, high):
        return power[(frequencies >= low) & (frequencies < high)].mean(axis=0)

    source_edge = numpy.interp(times, source_times, band_energy(
        source_frequencies, source_power, cutoff - 1300, cutoff - 300
    ))
    source_speech = numpy.interp(times, source_times, band_energy(
        source_frequencies, source_power, 300, 3000
    ))
    output_speech = band_energy(frequencies, power, 300, 3000)
    active = source_speech > source_speech.max() * 0.01
    if not active.any():
        return audio.copy()
    level_match = numpy.median(output_speech[active] / numpy.maximum(source_speech[active], 1e-20))

    # Estimate the source's noise floor from its quieter frames. Also respect
    # DPDFNet's reduction below the cutoff instead of leaving bright highs.
    noise_floor = numpy.percentile(source_edge, 20)
    source_edge = numpy.maximum(source_edge - 2 * noise_floor, 1e-20) * level_match
    output_edge = band_energy(frequencies, power, cutoff - 1300, cutoff - 300)
    ceiling = numpy.minimum(source_edge, output_edge) * 10 ** (3 / 10)
    ceiling = maximum_filter1d(ceiling, size=5)  # Preserve nearby consonant energy.
    generated = band_energy(frequencies, power, cutoff + 500, 12000)
    reduction_db = numpy.clip(10 * numpy.log10((generated + 1e-20) / (ceiling + 1e-20)), 0, 24)
    reduction_db = uniform_filter1d(reduction_db, size=5)

    # Fade the correction in above the cutoff, leaving the original band alone.
    blend = numpy.clip((frequencies - cutoff) / 700, 0, 1)
    blend = 0.5 - 0.5 * numpy.cos(numpy.pi * blend)
    gain = 10 ** (-blend[:, None] * reduction_db[None, :] / 20)
    _, removed = istft(spectrum * (1 - gain), 48000, nperseg=1536, noverlap=1152, scaling="psd")
    return (audio - removed[:len(audio)]).astype(audio.dtype)


def write_processed_audio(output_path, audio, reference, cutoff):
    """Clean the 48 kHz output using its 16 kHz reference and save it."""
    audio = suppress_lavasr_10khz_clicks(audio)
    audio = balance_lavasr_high_band(audio, reference, cutoff)
    audio = suppress_lavasr_hiss(audio)
    sf.write(file=output_path, data=remove_peaks(audio), samplerate=48000)


def process_listed_wavs(input_folder, file_list, process_wav, cutoff):
    """Back up listed WAVs and replace them with output made from the originals."""
    input_folder = Path(input_folder)
    if not input_folder.is_dir():
        raise NotADirectoryError(f"Input folder does not exist: {input_folder}")

    filenames = list(dict.fromkeys(
        line.strip()
        for line in Path(file_list).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ))
    if not filenames:
        print(f"No filenames in {file_list} — skipping.")
        return

    originals_folder = input_folder / "original_wavs"
    originals_folder.mkdir(exist_ok=True)
    for filename in filenames:
        if Path(filename).name != filename or Path(filename).suffix.lower() != ".wav":
            print(f"  Invalid WAV filename: {filename} — skipping.")
            continue
        path = input_folder / filename
        if not path.is_file():
            print(f"  Missing: {path} — skipping.")
            continue
        print(f"processing: {path}")
        try:
            original_path = originals_folder / filename
            with TemporaryDirectory(prefix=".upsample-", dir=input_folder) as temporary:
                temporary = Path(temporary)
                if not original_path.exists():
                    # Finish the copy before making it the permanent backup.
                    backup_path = temporary / "original.wav"
                    copy2(path, backup_path)
                    backup_path.replace(original_path)

                # Repeated runs always start from the preserved original.
                output_path = temporary / "enhanced.wav"
                process_wav(str(original_path), str(output_path), cutoff)
                # Replace the input only after processing and writing succeed.
                output_path.replace(path)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            print(f"  Skipping {path}, continuing with the rest of the batch.")
