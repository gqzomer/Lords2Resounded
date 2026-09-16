from LavaSR.model import LavaEnhance2
import soundfile as sf
import glob
import os
import dpdfnet
import numpy

from scipy.signal import butter, sosfiltfilt, find_peaks


lava_model = LavaEnhance2("YatharthS/LavaSR", "cpu")


def remove_peaks(audio, ceiling=0.8912):
    peak = numpy.max(numpy.abs(audio))

    if peak > ceiling:
        audio = audio * (ceiling / peak)

    return audio


def suppress_lavasr_10khz_clicks(
    audio,
    sample_rate: int = 48000,
    low_freq: float = 9500.0,
    high_freq: float = 10700.0,
    threshold: float = 0.18,
    min_gap_ms: float = 30.0,
    window_ms: float = 7.0,
    attenuation_db: float = -18.0
):
    """
    Detect short narrowband LavaSR artifacts around ~10.1 kHz and attenuate
    only that narrow frequency component during the transient.

    The original broadband signal is preserved except for the detected
    9.5-10.7 kHz transient component.
    """

    # ---------------------------------------------------------
    # Extract only the suspicious ~10 kHz region.
    # ---------------------------------------------------------

    band_sos = butter(
        N=6,
        Wn=[low_freq, high_freq],
        btype="bandpass",
        fs=sample_rate,
        output="sos"
    )

    suspicious_band = sosfiltfilt(
        band_sos,
        audio
    )

    # ---------------------------------------------------------
    # Build a short smoothed energy envelope.
    # ---------------------------------------------------------

    envelope = numpy.abs(suspicious_band)

    smooth_samples = max(
        1,
        int(sample_rate * 0.0015)
    )

    kernel = numpy.ones(
        smooth_samples,
        dtype=numpy.float32
    ) / smooth_samples

    envelope = numpy.convolve(
        envelope,
        kernel,
        mode="same"
    )

    max_envelope = numpy.max(envelope)

    if max_envelope <= 0:
        return audio.copy()

    normalized_envelope = (
        envelope / max_envelope
    )

    # ---------------------------------------------------------
    # Detect isolated short HF bursts.
    # ---------------------------------------------------------

    min_gap_samples = int(
        sample_rate * min_gap_ms / 1000.0
    )

    peaks, properties = find_peaks(
        normalized_envelope,
        height=threshold,
        distance=min_gap_samples
    )

    print(
        f"  Detected {len(peaks)} "
        f"candidate ~10 kHz transients"
    )

    # ---------------------------------------------------------
    # Construct attenuation mask.
    # ---------------------------------------------------------

    attenuation_gain = 10.0 ** (
        attenuation_db / 20.0
    )

    mask = numpy.ones(
        len(audio),
        dtype=numpy.float32
    )

    half_window = int(
        sample_rate * window_ms / 2000.0
    )

    fade_samples = max(
        1,
        int(sample_rate * 0.0015)
    )

    for peak in peaks:
        start = max(
            0,
            peak - half_window
        )

        end = min(
            len(audio),
            peak + half_window
        )

        if end <= start:
            continue

        print(
            f"    transient at "
            f"{peak / sample_rate:.3f}s"
        )

        # -----------------------------------------------------
        # Fade down.
        # -----------------------------------------------------

        fade_down_end = min(
            start + fade_samples,
            end
        )

        fade_down_length = (
            fade_down_end - start
        )

        if fade_down_length > 0:
            fade_down = numpy.linspace(
                1.0,
                attenuation_gain,
                fade_down_length,
                endpoint=False
            )

            mask[start:fade_down_end] = numpy.minimum(
                mask[start:fade_down_end],
                fade_down
            )

        # -----------------------------------------------------
        # Hold attenuation.
        # -----------------------------------------------------

        fade_up_start = max(
            fade_down_end,
            end - fade_samples
        )

        if fade_up_start > fade_down_end:
            mask[
                fade_down_end:fade_up_start
            ] = numpy.minimum(
                mask[
                    fade_down_end:fade_up_start
                ],
                attenuation_gain
            )

        # -----------------------------------------------------
        # Fade back up.
        # -----------------------------------------------------

        fade_up_length = (
            end - fade_up_start
        )

        if fade_up_length > 0:
            fade_up = numpy.linspace(
                attenuation_gain,
                1.0,
                fade_up_length,
                endpoint=True
            )

            mask[fade_up_start:end] = numpy.minimum(
                mask[fade_up_start:end],
                fade_up
            )

    # ---------------------------------------------------------
    # Attenuate ONLY the narrow suspicious frequency band.
    #
    # audio = everything
    # suspicious_band = only ~9.5-10.7 kHz
    #
    # We replace that band with its attenuated version.
    # ---------------------------------------------------------

    cleaned_audio = (
        audio
        - suspicious_band
        + suspicious_band * mask
    )

    return cleaned_audio


def process_wav(
    input_path: str,
    output_path: str,
    cutoff: int
):
    input_audio, input_sr = lava_model.load_audio(
        input_path,
        cutoff=cutoff
    )

    output_audio = lava_model.enhance(
        input_audio,
        enhance=True,
        denoise=False
    ).cpu().numpy().squeeze()

    # ---------------------------------------------------------
    # TEST:
    # Remove short LavaSR-generated ~10 kHz transients.
    # ---------------------------------------------------------

    # output_audio = suppress_lavasr_10khz_clicks(
    #     output_audio,
    #     sample_rate=48000,
    #     low_freq=9500.0,
    #     high_freq=10700.0,
    #     threshold=0.18,
    #     min_gap_ms=80.0,
    #     window_ms=7.0,
    #     attenuation_db=-18.0
    # )

    # ---------------------------------------------------------
    # Normal DPDFNet stage.
    # ---------------------------------------------------------

    output_audio = dpdfnet.enhance(
        output_audio,
        sample_rate=48000,
        model="dpdfnet8_48khz_hr"
    )

    sf.write(
        file=output_path,
        data=remove_peaks(output_audio),
        samplerate=48000
    )


def preprocess_folder(
    input_folder: str,
    output_folder: str,
    cutoff: int
):
    wav_paths = sorted(set(
        glob.glob(
            os.path.join(
                input_folder,
                "*.wav"
            )
        )
        + glob.glob(
            os.path.join(
                input_folder,
                "*.WAV"
            )
        )
    ))

    if not wav_paths:
        print(
            f"No .wav files found in "
            f"{input_folder} — skipping."
        )
        return

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    for path in wav_paths:
        print(f"processing: {path}")

        try:
            out_path = os.path.join(
                output_folder,
                os.path.basename(path)
            )

            process_wav(
                path,
                out_path,
                cutoff
            )

        except Exception as e:
            print(f"  FAILED: {e}")
            print(
                f"  Skipping {path}, "
                "continuing with the rest of the batch."
            )
            continue


if __name__ == "__main__":
    preprocess_folder(
        "/home/Gersom/Git/R2R/input_wavs/subset",
        "/home/Gersom/Git/R2R/output_no_filtered",
        5300
    )
