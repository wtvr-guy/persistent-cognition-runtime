"""Verify the pinned extraction, distinguishing unchanged and adapted files."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "SOURCE_BASELINE.json").read_text(encoding="utf-8"))
    failures = []
    counts = {"unchanged": 0, "adapted": 0}
    for relative, record in manifest["files"].items():
        path = ROOT / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if actual != record["extracted_sha256"]:
            failures.append(relative)
        kind = "unchanged" if record["source_sha256"] == record["extracted_sha256"] else "adapted"
        counts[kind] += 1
    if failures:
        raise SystemExit("Extraction drift: " + ", ".join(failures))
    print(f"Extraction verified against {manifest['source_commit']}: {counts}")


if __name__ == "__main__":
    main()
