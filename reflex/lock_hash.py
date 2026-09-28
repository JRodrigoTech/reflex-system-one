"""Stable hashes for dependency locks across Windows and Unix checkouts."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


def _normalized_lock_bytes(raw: bytes) -> bytes:
    """Return lock content with platform line endings represented as LF."""
    return raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def lock_sha256(path: Path) -> str:
    """Hash lock content independently of the checkout's line-ending policy."""
    return hashlib.sha256(_normalized_lock_bytes(path.read_bytes())).hexdigest()


def lock_hash_matches(candidate: object, path: Path) -> bool:
    """Accept canonical hashes and hashes written by older Reflex versions."""
    if not isinstance(candidate, str):
        return False
    raw = path.read_bytes()
    normalized = _normalized_lock_bytes(raw)
    crlf = normalized.replace(b"\n", b"\r\n")
    accepted = {
        hashlib.sha256(raw).hexdigest(),
        hashlib.sha256(normalized).hexdigest(),
        hashlib.sha256(crlf).hexdigest(),
    }
    return candidate.lower() in accepted


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m reflex.lock_hash <lock-file>", file=sys.stderr)
        return 2
    try:
        print(lock_sha256(Path(sys.argv[1])))
    except OSError as error:
        print(f"Unable to hash dependency lock: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
