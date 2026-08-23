"""Get RoBERTa-CNN's ~500MB checkpoint through git despite GitHub's 100MB
per-file limit, by committing it as a set of smaller parts instead of the
raw file. `split` (run once, before committing a new checkpoint) is the
other half of this — see its own docstring for what it writes.

Cloning the repo gets a teammate the parts, not the checkpoint PyTorch
actually needs. `restore` is the command that turns those parts back into
that file — meant to run once, right after cloning or pulling one down.
"""

import argparse
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNK_SIZE = 95 * 1024 * 1024  # comfortably under GitHub's 100MB hard limit


def split_file(path: Path, chunk_size: int = CHUNK_SIZE) -> Path:
    """Split `path` into <chunk_size> parts under a sibling `<name>.parts/` folder."""
    parts_dir = path.parent / f"{path.name}.parts"
    parts_dir.mkdir(exist_ok=True)

    digest = hashlib.sha256()
    chunk_count = 0
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
            (parts_dir / f"part-{chunk_count:03d}").write_bytes(chunk)
            chunk_count += 1

    manifest = {
        "filename": path.name,
        "size": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "chunk_count": chunk_count,
    }
    (parts_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Split {path} ({manifest['size']:,} bytes) into {chunk_count} parts under {parts_dir}")
    return parts_dir


def join_parts(parts_dir: Path) -> Path:
    """Reassemble a `<name>.parts/` folder back into its original file, sha256-verified."""
    manifest = json.loads((parts_dir / "manifest.json").read_text(encoding="utf-8"))
    output_path = parts_dir.parent / manifest["filename"]

    digest = hashlib.sha256()
    with output_path.open("wb") as out:
        for index in range(manifest["chunk_count"]):
            chunk = (parts_dir / f"part-{index:03d}").read_bytes()
            digest.update(chunk)
            out.write(chunk)

    if digest.hexdigest() != manifest["sha256"]:
        output_path.unlink()
        raise ValueError(
            f"Checksum mismatch reassembling {output_path.name} — a part is missing or "
            "corrupted. Delete the .parts folder and re-fetch it, then try again."
        )
    print(f"Restored {output_path} ({manifest['size']:,} bytes, checksum verified)")
    return output_path


def restore_all() -> None:
    """Reconstruct every checkpoint under data/models/ whose target file is missing
    or doesn't match its manifest's recorded size — skips ones already restored."""
    parts_dirs = sorted((PROJECT_ROOT / "data" / "models").glob("**/*.parts"))
    if not parts_dirs:
        print("No split checkpoints found under data/models/.")
        return

    for parts_dir in parts_dirs:
        manifest = json.loads((parts_dir / "manifest.json").read_text(encoding="utf-8"))
        target = parts_dir.parent / manifest["filename"]
        if target.exists() and target.stat().st_size == manifest["size"]:
            print(f"{target} already present, skipping.")
            continue
        join_parts(parts_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    split_parser = subparsers.add_parser("split", help="split one checkpoint into parts")
    split_parser.add_argument("path", type=Path)

    subparsers.add_parser("restore", help="reassemble every split checkpoint under data/models/")

    args = parser.parse_args()
    if args.command == "split":
        split_file(args.path)
    elif args.command == "restore":
        restore_all()


if __name__ == "__main__":
    main()
