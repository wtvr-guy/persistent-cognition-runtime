"""Automatically package a complete audit run for review; never overwrite evidence.

Archive creation is intentionally outside the SHA-linked run journal because the
archive contains the journal itself. Each archived file has an independent SHA-256
in BUNDLE_MANIFEST.json; verifier checks exact membership and content.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

ARCHIVE_NAME = "evidence.zip"
BUNDLE_META = "BUNDLE_MANIFEST.json"
BUNDLE_VERSION = "cognitive-effort-bundle/v1"


def _digest(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def package_run(root: Path) -> Path:
    """Archive all ordinary run evidence, including case JSON, without mutating it.

    Raw native output and native_artifacts are intentionally included if present:
    this bundle can contain sensitive data and must be shared deliberately.
    """
    root = root.resolve()
    if not (root / "events.jsonl").is_file():
        raise ValueError("run has no audit event stream")
    target = root / ARCHIVE_NAME
    files = sorted(
        (p for p in root.rglob("*") if p.is_file() and p != target
         and not p.name.endswith(".pending") and not p.is_symlink()),
        key=lambda p: p.relative_to(root).as_posix(),
    )
    entries: dict[str, str] = {}
    for p in files:
        relative = p.relative_to(root).as_posix()
        if relative == BUNDLE_META:
            raise ValueError("reserved bundle metadata filename in run directory")
        entries[relative] = _digest(p.read_bytes())
    manifest = {
        "version": BUNDLE_VERSION,
        "files_sha256": entries,
        "audit_chain": "events.jsonl",
        "warning": "May contain private inputs or native test output; inspect before sharing.",
    }
    temporary = root / (ARCHIVE_NAME + ".pending")
    if temporary.exists():
        raise FileExistsError(f"existing incomplete archive: {temporary}")
    try:
        with zipfile.ZipFile(temporary, mode="x", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6, allowZip64=True) as archive:
            for p in files:
                name = p.relative_to(root).as_posix()
                archive.write(p, arcname=name)
            archive.writestr(
                BUNDLE_META,
                json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
            )
        verify_bundle(temporary)
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def verify_bundle(path: Path) -> dict:
    with zipfile.ZipFile(path, mode="r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate archive entries")
        if BUNDLE_META not in names:
            raise ValueError("missing bundle manifest")
        manifest = json.loads(archive.read(BUNDLE_META))
        if manifest.get("version") != BUNDLE_VERSION:
            raise ValueError("unsupported bundle version")
        expected: dict = manifest["files_sha256"]
        if not isinstance(expected, dict) or set(names) != set(expected) | {BUNDLE_META}:
            raise ValueError("bundle file inventory mismatch")
        for name, digest in expected.items():
            location = Path(name)
            if location.is_absolute() or ".." in location.parts or "\\" in name:
                raise ValueError(f"unsafe bundle filename: {name}")
            if _digest(archive.read(name)) != digest:
                raise ValueError(f"bundle artifact digest mismatch: {name}")
        return {"valid": True, "file_count": len(expected),
                "audit_sha256": expected.get("events.jsonl")}
