"""Merge one adapter onto its immediate parent; repeat for all three stages."""
import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ["base-model", "adapter", "output", "report", "status"]:
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    if Path(args.output).exists() and any(Path(args.output).iterdir()):
        parser.error("Merge output directory must be empty")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from openperov.merge import main as merge_main
    merge_main()


if __name__ == "__main__":
    main()
