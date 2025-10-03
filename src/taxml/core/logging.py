# taxml/core/logging.py
from __future__ import annotations

import logging
logging.getLogger(__name__)

from logging.handlers import MemoryHandler
from pathlib import Path
from typing import Optional

# module-level stash so we can flush later
_BUFFER: Optional[MemoryHandler] = None
_FILE: Optional[logging.Handler] = None

def setup_logging(
    *,
    console_level: int = logging.INFO,
    file_path: Optional[Path] = None,          # if given at boot, we attach immediately
    file_level: int = logging.DEBUG,
    root_level: int = logging.DEBUG,
    capture_warnings: bool = True,
    reconfigure: bool = True,
    buffer_early: bool = False,                # <— NEW: buffer early logs until file is attached
    buffer_capacity: int = 10_000,
    fmt: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> None:
    global _BUFFER, _FILE

    if capture_warnings:
        logging.captureWarnings(True)

    root = logging.getLogger()
    if reconfigure:
        for h in list(root.handlers):
            root.removeHandler(h)
    root.setLevel(root_level)

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # console
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # optional immediate file
    if file_path is not None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(file_path, encoding="utf-8")
        fh.setLevel(file_level)
        fh.setFormatter(formatter)
        root.addHandler(fh)
        _FILE = fh

    # optional early buffer (only if no file yet)
    if buffer_early and _FILE is None:
        _BUFFER = MemoryHandler(capacity=buffer_capacity, flushLevel=logging.CRITICAL + 1)
        _BUFFER.setLevel(logging.DEBUG)
        _BUFFER.setFormatter(formatter)
        root.addHandler(_BUFFER)

    root.debug(
        "Logging configured (console=%s, file=%s, buffered=%s)",
        logging.getLevelName(console_level),
        getattr(_FILE, "baseFilename", None),
        _BUFFER is not None,
    )

def attach_file_logger(
    file_path: Path,
    *,
    file_level: int = logging.DEBUG,
    fmt: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    flush_buffer: bool = True,
    keep_buffer: bool = False,  # keep buffering for a while if you really want to
) -> None:
    """Attach a file handler later and optionally flush any early buffered logs."""
    global _BUFFER, _FILE
    root = logging.getLogger()

    # idempotent: if already attached to same path, do nothing
    if _FILE and getattr(_FILE, "baseFilename", None) == str(file_path):
        return

    file_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(file_path, encoding="utf-8")
    fh.setLevel(file_level)
    fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    root.addHandler(fh)
    _FILE = fh

    if flush_buffer and _BUFFER is not None:
        _BUFFER.setTarget(fh)
        _BUFFER.flush()
        if not keep_buffer:
            root.removeHandler(_BUFFER)
            _BUFFER.close()
            _BUFFER = None

def reset_logging() -> None:
    """Convenience for tests: remove all handlers and clear buffer/file state."""
    global _BUFFER, _FILE
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
    _BUFFER = None
    _FILE = None
