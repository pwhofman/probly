"""Normalize tracked-file mtimes from content checksums for incremental builds.

A fresh CI checkout stamps every file with the current time, making Sphinx
treat everything as outdated. Run after restoring the docs build cache:
files whose checksum matches the cached manifest get a fixed old mtime;
changed or new files keep their fresh mtime and are rebuilt. Without a
manifest (cold build) a full build runs.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "docs" / "build" / "file-checksums.json"

# 2000-01-01 UTC; anything older than the cached environment's read times works.
OLD_MTIME = 946684800


def main() -> None:
    """Reset mtimes of files whose checksum matches the cached manifest."""
    listing = subprocess.run(
        ["git", "ls-files", "-z"],  # noqa: S607
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    )
    tracked = [name for name in listing.stdout.decode().split("\0") if name]
    previous: dict[str, str] = {}
    if MANIFEST.is_file():
        previous = json.loads(MANIFEST.read_text())
    current: dict[str, str] = {}
    unchanged = 0
    for name in tracked:
        path = REPO_ROOT / name
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        current[name] = digest
        if previous.get(name) == digest:
            os.utime(path, (OLD_MTIME, OLD_MTIME))
            unchanged += 1
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(current, sort_keys=True, indent=0) + "\n")
    fresh = len(current) - unchanged
    print(f"Reset mtimes of {unchanged} unchanged file(s); {fresh} kept fresh.")  # noqa: T201


if __name__ == "__main__":
    main()
