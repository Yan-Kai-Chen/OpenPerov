"""Run a DAPT/scientific/style stage on user-supplied data and local weights."""
import argparse
import importlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["dapt", "scientific", "style", "selector", "reranker"], required=True)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    if not args.config.is_file():
        parser.error("Configuration does not exist")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    if args.stage in {"selector", "reranker"}:
        cfg = json.loads(args.config.read_text())
        module = importlib.import_module("openperov.ranker_train")
        command = [sys.argv[0]]
        for key, value in cfg.items():
            if key.startswith("_") or value is None:
                continue
            if isinstance(value, bool):
                if value:
                    command.append("--" + key.replace("_", "-"))
            else:
                command.extend(["--" + key.replace("_", "-"), str(value)])
        sys.argv = command
    else:
        module = importlib.import_module("openperov.training_" + args.stage)
        sys.argv = [sys.argv[0], "--config", str(args.config)]
    module.main()


if __name__ == "__main__":
    main()
