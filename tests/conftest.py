"""Shared fixtures. Every test runs against a temp data dir — never ~/.hermes
or the user's real library."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch):
    """Point config + db at a temp directory and reset module caches."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("VDJ_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VDJ_DB_PATH", str(data_dir / "test.sqlite3"))

    from app import config, db

    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "CACHE_DIR", data_dir / "cache")
    monkeypatch.setattr(config, "DJ_CACHE_DIR", data_dir / "cache" / "dj")
    monkeypatch.setattr(config, "VOICES_DIR", data_dir / "voices")
    monkeypatch.setattr(config, "CONFIG_PATH", data_dir / "config.json")
    monkeypatch.setattr(config, "DB_PATH", data_dir / "test.sqlite3")
    monkeypatch.setattr(config, "_CACHE", None, raising=False)

    db.close()
    config.ensure_dirs()
    db.init_db()
    yield data_dir
    db.close()


def make_mp3(path: Path, seconds: float = 1.0, freq: int = 440,
             tags: dict[str, str] | None = None) -> Path:
    """Render a real, tiny mp3 with ffmpeg so tests exercise real decoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={seconds}",
        "-c:a", "libmp3lame", "-b:a", "64k", "-ar", "44100", "-ac", "2",
    ]
    for key, value in (tags or {}).items():
        cmd += ["-metadata", f"{key}={value}"]
    cmd.append(str(path))
    subprocess.run(cmd, check=True, capture_output=True, timeout=60)
    return path


@pytest.fixture
def music_dir(tmp_path):
    root = tmp_path / "music"
    make_mp3(root / "a.mp3", tags={"title": "Alpha", "artist": "Band One",
                                   "album": "First", "genre": "Rock",
                                   "date": "1999"})
    make_mp3(root / "sub" / "b.mp3", freq=660,
             tags={"title": "Beta", "artist": "Band Two", "genre": "Pop"})
    # Untagged: exercises the filename-guess path.
    make_mp3(root / "Band Three - Gamma.mp3", freq=880)
    return root


@pytest.fixture
def has_ffmpeg():
    if subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0:
        pytest.skip("ffmpeg not installed")
    return True


os.environ.setdefault("VDJ_LOG_LEVEL", "WARNING")
