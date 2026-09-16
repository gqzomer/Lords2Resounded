import argparse
from R2R import upsample


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
        "--cutoff", "-c",
        type=int,
        default=5300,
        help=("The overlap in seconds that should be used for crossfade stitching longer files")
    )
    return parser


def main():
    args = build_parser().parse_args()
    upsample.process_folder(
        input_folder=args.input,
        output_folder=args.output,
        cutoff=args.cutoff
    )

if __name__ == "__main__":
    main()