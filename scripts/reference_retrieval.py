"""Reference multivector indexing/retrieval and natural Top40 evidence assembly.

Examples:
  python scripts/reference_retrieval.py index --help
  python scripts/reference_retrieval.py retrieve --help
  python scripts/reference_retrieval.py evidence --help
Inputs are user-provided article/facet/topic records; private data is not bundled.
"""
import argparse
import importlib
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("command", choices=["index", "retrieve", "evidence"], nargs="?")
    if not sys.argv[1:] or sys.argv[1:] == ["--help"]:
        parser.print_help()
        return
    args, remainder = parser.parse_known_args()
    if not args.command:
        parser.error("Choose index, retrieve or evidence")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    name = {"index": "reference_index", "retrieve": "reference_retrieval", "evidence": "reference_evidence"}[args.command]
    module = importlib.import_module("openperov." + name)
    sys.argv = [sys.argv[0]] + remainder
    module.main()


if __name__ == "__main__":
    main()
