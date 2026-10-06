"""Explicitly freeze Hugging Face revisions and download into the local HF cache.

Run with the separate inference Python. Downloads need internet and, for Gemma,
an accepted model license plus a locally configured Hugging Face credential.
"""
import argparse
import json
from pathlib import Path


def main():
    from huggingface_hub import HfApi, snapshot_download
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=Path("benchmarks/specialists/profiles.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", action="append", help="Only freeze/download these HF model IDs")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; use a new filename to preserve the frozen matrix")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    matrix = json.loads(args.matrix.read_text())
    identities = {}
    errors = []
    for profile in matrix["profiles"]:
        model = profile["model"]
        if profile["backend"] != "hf" or profile.get("blocked_reason"):
            continue
        if args.model and model not in args.model:
            continue
        if model not in identities:
            try:
                revision = profile.get("revision") or HfApi().model_info(model).sha
                snapshot_download(model, revision=revision,
                                  allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.model", "merges.txt", "*.txt"])
                identities[model] = revision
            except Exception as exc:
                errors.append({"model": model, "error_type": type(exc).__name__, "message": str(exc)})
                identities[model] = None
        profile["revision"] = identities[model]
    matrix["preparation_errors"] = errors
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(matrix, handle, indent=2)
        handle.write("\n")
    print(args.output)
    for error in errors:
        print(json.dumps(error))
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
