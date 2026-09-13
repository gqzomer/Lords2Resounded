import argparse
from r2r_upsample import upsample


def build_parser():
    parser = argparse.ArgumentParser(
        description="upsample wavs using audioSR"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Folder containing source .wav files to preprocess.",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Folder to write enhanced .wav files to.",
    )
    parser.add_argument(
        "--model", "-m",
        default="speech",
        help=("Which model should be used by audioSR"),
    )
    parser.add_argument(
        "--guidance", "-g",
        type=float,
        default=3.5,
        help=("The guidance factor that should be used by audioSR")
    )
    parser.add_argument(
        "--steps", "-s",
        type=int,
        default=50,
        help=("The number of DDIM steps that should be used by audioSR")
    )
    parser.add_argument(
        "--crossfade", "-c",
        type=float,
        default=0.64,
        help=("The overlap in seconds that should be used for crossfade stitching longer files")
    )
    return parser


def main():
    args = build_parser().parse_args()
    upsample.upsample(
        input_folder=args.input,
        output_folder=args.output,
        model=args.model,
        guidance_scale=args.guidance,
        ddim_steps=args.steps,
        overlap=args.crossfade
    )

if __name__ == "__main__":
    main()