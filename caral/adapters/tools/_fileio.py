"""
Safe text-file I/O for tools: strict UTF-8 handling (refuse rather than corrupt), CRLF and BOM
preservation, symlink-aware atomic writes that keep file permissions.

- NotTextError: file is binary or not valid UTF-8.
- TextFile: text (LF-normalized), crlf (bool), bom (bool).
- read_text_file(path) -> TextFile: strict read; raises NotTextError / OSError.
- write_text_file(path, text, crlf=False, bom=False) -> None: atomic replace (temp file + fsync + rename).
- looks_binary(data) -> bool: NUL byte in the first 8 KiB.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path


class NotTextError(Exception):
    pass


@dataclass
class TextFile:
    text: str
    crlf: bool = False
    bom: bool = False


def looks_binary(data: bytes) -> bool:
    return b"\x00" in data[:8192]


def read_text_file(path: Path) -> TextFile:
    data = path.read_bytes()
    if looks_binary(data):
        raise NotTextError(f"{path} looks like a binary file.")
    bom = data.startswith(b"\xef\xbb\xbf")
    try:
        raw = data[3:].decode("utf-8") if bom else data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise NotTextError(
            f"{path} is not valid UTF-8 (byte {e.start}); refusing to edit it because that would corrupt it."
        ) from e
    crlf = "\r\n" in raw
    return TextFile(raw.replace("\r\n", "\n") if crlf else raw, crlf, bom)


def write_text_file(path: Path, text: str, crlf: bool = False, bom: bool = False) -> None:
    if crlf:
        text = text.replace("\r\n", "\n").replace("\n", "\r\n")
    data = (b"\xef\xbb\xbf" if bom else b"") + text.encode("utf-8")
    target = Path(os.path.realpath(path))  # write through symlinks instead of replacing them
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".caral-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if target.exists():
            shutil.copymode(target, tmp)
        else:
            os.chmod(tmp, 0o666 & ~_umask())
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _umask() -> int:
    current = os.umask(0)
    os.umask(current)
    return current
