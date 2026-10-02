"""Check the frozen public file manifest without loading model weights."""
from pathlib import Path
import argparse
import hashlib
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "release_manifest.json").read_text(encoding="utf-8"))
    entries = manifest.get("files", [])
    if not entries:
        raise ValueError("A completed public file manifest is required")
    total = 0
    for entry in entries:
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Manifest path leaves the release directory")
        if not path.is_file():
            raise ValueError("Missing release file: " + entry["path"])
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if path.stat().st_size != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
            raise ValueError("File differs from release manifest: " + entry["path"])
        total += entry["bytes"]
    ids = []
    for entry in manifest["datasets"]:
        path = root / entry["path"]
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != entry["records"]:
            raise ValueError("Unexpected question count: " + entry["path"])
        ids.extend(row["id"] for row in rows)
    if len(ids) != len(set(ids)) or len(ids) != 880:
        raise ValueError("Expected 880 unique question/task IDs")
    print(json.dumps({"status": "PASS", "files": len(entries), "bytes": total,
                      "task_records": len(ids), "scope": "listed file integrity and question identity"}, indent=2))


if __name__ == "__main__":
    main()
