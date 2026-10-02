"""Verify the copied frozen code and optional C-MAPSS source files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def verify(root: Path, raw_dir: Path | None = None) -> int:
    manifest = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    entries = [(root / "src" / "rul_tta" / name, expected) for name, expected in manifest["source_sha256"].items()]
    if raw_dir is not None:
        entries.extend((raw_dir / name, expected) for name, expected in manifest["raw_sha256"].items())
    errors = []
    for path, expected in entries:
        if not path.is_file():
            errors.append(f"missing: {path}")
        elif digest(path) != expected:
            errors.append(f"SHA-256 mismatch: {path}")
    if errors:
        raise SystemExit("\n".join(errors))
    return len(entries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path)
    args = parser.parse_args()
    count = verify(ROOT, args.raw_dir)
    print(f"Verified {count} files")


if __name__ == "__main__":
    main()
