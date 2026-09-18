# Lords2 Restoration Project
The Lords2 Restoration project, or L2R for short, aims to restore the audio of [Lords of the Realm II](https://en.wikipedia.org/wiki/Lords_of_the_Realm_II) to its original, uncompressed quality wherever possible. The game's audio was originally produced in **16-bit, 44.1 kHz stereo**. However, due to the limited storage capacity of CDs at the time, as well as the need to keep audio files small enough to load into memory, the final game shipped with heavily downsampled **8-bit, 11.025 kHz mono** WAV files. 

The original uncompressed audio files have most likely been lost or remain buried somewhere in the archives of Sierra On-Line or Rebellion. The conversion to 8-bit, 11.025 kHz audio affected the original recordings in two major ways:

1. Frequencies above approximately **5.5 kHz** were removed.
2. Significant **quantization noise** was introduced.

## Restoration Process
The game's audio can generally be divided into four categories, each requiring a different approach to restoration:

* Background music
* Unit responses
* Dialogue voice-over
* Sound effects

### Background music
The game's original composer, Keith Zizza, has uploaded the original background music from Lords of the Realm II to [Youtube](https://www.youtube.com/playlist?list=PLYb1G8dEQFnW8cwh_1Su1V7Fg8oMNUIGU). With the exception of `setup2.wav`, all of the game's music appears to be available in this collection. As a result, these tracks can be restored using the original high-quality recordings rather than attempting to reconstruct the compressed game files.

The restoration process for this category consists of:

1. Downloading the corresponding high-quality track.
2. Matching its exact duration to the original game file.
3. Removing any added fade-outs so that tracks intended for continuous playback can loop correctly.

### Dialogue voice-over
This category contains the spoken dialogue that accompanies hints, confirmation prompts, and other message windows, for example:

* "Exit the game, my lord?"
* "Slaughter these villagers?"

The original uncompressed recordings are not available for this category. Fortunately, human speech follows relatively predictable spectral patterns, which makes AI-based restoration a viable option.
The restoration process consists of two stages:

1. Reconstructing the missing frequencies above approximately 5.3 kHz using [LavaSR](https://github.com/ysharma3501/LavaSR).
2. Reducing the remaining quantization noise using the 48 kHz model from [DPDFnet](https://github.com/ceva-ip/DPDFNet).

### Unit responses
This category contains the short acknowledgements spoken when units are selected or given commands during real-time battles, for example:

* "Archers, pull!"
* "Bowmen ready!"

Although these files also contain speech, they are considerably shorter than the dialogue voice-over samples. This makes them more difficult to process and requires a slightly different restoration pipeline.As with the dialogue recordings, the original uncompressed files are not currently available. The restoration process therefore relies on AI-based reconstruction, but reverses the order of the processing stages:

1. Reducing quantization noise using the 8 kHz model from [DPDFnet](https://github.com/ceva-ip/DPDFNet).
2. Reconstructing the missing frequencies above approximately 4 kHz using [LavaSR](https://github.com/ysharma3501/LavaSR).

### Sound effects
The final category consists of all non-speech sound effects used by the game, such as the sound played when changing the crop type of a field. Unlike speech, these sounds do not follow a sufficiently predictable structure for AI-based reconstruction to produce reliable results. In practice, attempting to generate the missing high-frequency content often introduces unwanted artifacts or changes the character of the original sound.

For this reason, the preferred approach is to recreate these files from their original source material wherever possible.The main challenge is identifying the sound libraries from which the game's effects were sourced. Fortunately, the use of commercial sound-effect libraries was commonplace in games from this era. The goal is therefore to locate the original source recordings for as many sound effects as possible and document their origins in this repository.