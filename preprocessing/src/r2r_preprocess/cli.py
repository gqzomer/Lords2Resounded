import argparse
from r2r_preprocess import preprocess


def build_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess voice WAVs with DPDFNet before upsampling."
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
        "--retention-threshold", "-t",
        type=float,
        default=0.8,
        help=(
            "If less than this fraction of the original's active content "
            "survives enhancement, fall back to the 8kHz model (default: 0.5)."
        ),
    )
    return parser


def main():
    args = build_parser().parse_args()
    preprocess.preprocess_folder(
        input_folder=args.input,
        output_folder=args.output,
        retention_threshold=args.retention_threshold,
    )


if __name__ == "__main__":
    main()