"""Verify source provenance, current extraction digests, additions and retirements."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify_manifest(manifest, root=ROOT):
    failures = []
    counts = {"unchanged": 0, "adapted": 0, "added": 0, "retired": 0}
    extraction_paths = set()
    for relative, record in manifest["files"].items():
        extraction_path = record.get("extraction_path", relative)
        relative_path = Path(extraction_path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            failures.append(f"invalid extraction path: {relative}")
            continue
        if extraction_path in extraction_paths:
            failures.append(f"duplicate extraction path: {extraction_path}")
        extraction_paths.add(extraction_path)
        path = root / relative_path
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        status = record.get("status", "imported")
        if status == "retired":
            if path.exists() or record["extracted_sha256"] is not None or not record.get("reason"):
                failures.append(f"invalid retirement: {relative}")
            if not record["source_sha256"] or not record["source_path"]:
                failures.append(f"retirement lost source provenance: {relative}")
            counts["retired"] += 1
            continue
        if status not in {"imported", "added"}:
            failures.append(f"invalid status: {relative}")
        if actual is None:
            failures.append(f"missing current file: {extraction_path}")
        if actual != record["extracted_sha256"]:
            failures.append(f"modified extraction: {extraction_path}")
        if status == "added":
            if record["source_path"] is not None or record["source_sha256"] is not None or not record.get("reason"):
                failures.append(f"invalid addition: {relative}")
            kind = "added"
        else:
            if not record["source_path"] or not record["source_sha256"]:
                failures.append(f"missing source provenance: {relative}")
            kind = "unchanged" if record["source_sha256"] == record["extracted_sha256"] else "adapted"
        counts[kind] += 1
    for directory in ("src", "tests", "scripts", "benchmarks"):
        for path in (root / directory).rglob("*.py"):
            relative = path.relative_to(root).as_posix()
            if relative not in extraction_paths:
                failures.append(f"unrecorded code: {relative}")
    if failures:
        raise SystemExit("Extraction drift: " + ", ".join(failures))
    return counts


def main():
    manifest = json.loads((ROOT / "SOURCE_BASELINE.json").read_text(encoding="utf-8"))
    counts = verify_manifest(manifest)
    print(f"Extraction verified against {manifest['source_commit']}: {counts}")


if __name__ == "__main__":
    main()
