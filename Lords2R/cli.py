import argparse


def build_parser():
    parser = argparse.ArgumentParser(
        description="Restore listed dialogue WAVs, then listed unit sound WAVs."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Folder to restore in place; originals are backed up in original_wavs.",
    )
    parser.add_argument(
        "--dialogue-cutoff",
        dest="dialogue_cutoff",
        type=int,
        default=5300,
        help="Dialogue reconstruction cutoff in Hz (default: 5300).",
    )
    parser.add_argument(
        "--unit-cutoff",
        type=int,
        default=5300,
        help="Unit sound reconstruction cutoff in Hz (default: 5300).",
    )
    return parser


def main():
    args = build_parser().parse_args()
    from Lords2R.dialogue import upsample as dialogue
    from Lords2R.unit_sounds import upsample as unit_sounds

    print("Processing dialogue")
    dialogue.upsample(
        input_folder=args.input,
        cutoff=args.dialogue_cutoff,
    )
    print("Processing unit sounds")
    unit_sounds.upsample(
        input_folder=args.input,
        cutoff=args.unit_cutoff,
    )

if __name__ == "__main__":
    main()
 