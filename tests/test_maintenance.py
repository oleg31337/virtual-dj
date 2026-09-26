"""Housekeeping: temporary data must clean itself up (user report).

The DJ audio cache had no eviction at all — every rendered break stayed forever —
and an interrupted voice download or a killed process could leave a partial file
with nothing to collect it. These tests pin the sweep's rules:

* the newest ``cache.dj_keep_files`` breaks survive, older/bigger ones go;
* a break the queue is ABOUT to play is never removed;
* stale partial files go, fresh ones (a download in flight) stay;
* voice MODELS are never touched;
* ``cache.enabled: false`` means no periodic sweep, but the on-demand sweep still
  works (the "Clean now" button is an explicit instruction).
"""

from __future__ import annotations

import os
import time

import pytest

from app import config, maintenance


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """App pointed at a temp data dir, with the real cache layout."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "DJ_CACHE_DIR", tmp_path / "cache" / "dj")
    monkeypatch.setattr(config, "VOICES_DIR", tmp_path / "voices")
    config.ensure_dirs()
    monkeypatch.setattr(config, "_CACHE", {
        "cache": {"enabled": True, "cleanup_interval_minutes": 10,
                  "dj_keep_files": 3, "dj_max_age_hours": 48, "dj_max_mb": 0,
                  "tmp_grace_minutes": 60},
        "logging": {"to_file": False},
    })
    return tmp_path


def _break(name: str, *, age_s: float = 0, size: int = 1024):
    path = config.DJ_CACHE_DIR / name
    path.write_bytes(b"x" * size)
    old = time.time() - age_s
    os.utime(path, (old, old))
    return path


def test_keeps_newest_and_prunes_the_rest(data_dir):
    keep = [_break(f"new{i}.mp3", age_s=i) for i in range(3)]     # newest three
    old = [_break(f"old{i}.mp3", age_s=3600 + i) for i in range(4)]
    report = maintenance.clean()
    assert report["removed_files"] == 4
    assert all(p.exists() for p in keep)
    assert not any(p.exists() for p in old)


def test_never_removes_a_break_the_queue_is_about_to_play(data_dir, monkeypatch):
    protected = _break("playing-next.mp3", age_s=10 * 24 * 3600)   # ancient, but queued
    monkeypatch.setattr(maintenance, "protected_names", lambda: {protected.name})
    newer = _break("newer.mp3", age_s=60)
    older = _break("older.mp3", age_s=3600)
    monkeypatch.setattr(config, "_CACHE", {
        "cache": {"enabled": True, "dj_keep_files": 1, "dj_max_age_hours": 0,
                  "dj_max_mb": 0, "tmp_grace_minutes": 60},
    })
    report = maintenance.clean()
    assert protected.exists(), "a prepared break must survive the sweep"
    assert protected.name in report["protected"]
    assert newer.exists(), "the newest unprotected break is kept"
    assert not older.exists(), "the surplus unprotected break is pruned"


def test_age_limit_removes_even_within_the_count_budget(data_dir, monkeypatch):
    fresh = _break("fresh.mp3", age_s=60)
    stale = _break("stale.mp3", age_s=49 * 3600)
    monkeypatch.setattr(config, "_CACHE", {
        "cache": {"enabled": True, "dj_keep_files": 100, "dj_max_age_hours": 48,
                  "dj_max_mb": 0, "tmp_grace_minutes": 60},
    })
    maintenance.clean()
    assert fresh.exists() and not stale.exists()


def test_size_budget_is_enforced_newest_first(data_dir, monkeypatch):
    monkeypatch.setattr(config, "_CACHE", {
        "cache": {"enabled": True, "dj_keep_files": 0, "dj_max_age_hours": 0,
                  # 1 MB budget: ~2.5 of these 400 KB files fit
                  "dj_max_mb": 1, "tmp_grace_minutes": 60},
    })
    paths = [_break(f"b{i}.mp3", age_s=i, size=400 * 1024) for i in range(6)]
    maintenance.clean()
    survivors = [p for p in paths if p.exists()]
    assert 0 < len(survivors) <= 3
    assert survivors[0] == paths[0], "the newest file must be the one kept"


def test_stale_partials_go_but_an_in_flight_download_stays(data_dir):
    fresh = data_dir / "voices" / "en_US-amy-medium.onnx.part"
    fresh.parent.mkdir(parents=True, exist_ok=True)
    fresh.write_bytes(b"partial")
    stale = data_dir / "voices" / "ru_RU-irina-medium.onnx.part"
    stale.write_bytes(b"partial")
    old = time.time() - 3 * 3600
    os.utime(stale, (old, old))
    cfg = dict(config._CACHE["cache"])
    config._CACHE["cache"] = cfg
    maintenance.clean()
    assert fresh.exists(), "a download in progress must not be swept"
    assert not stale.exists(), "an abandoned partial must be"


def test_voice_models_are_never_touched(data_dir):
    model = config.VOICES_DIR / "en_US-amy-medium.onnx"
    model.write_bytes(b"model")
    old = time.time() - 365 * 24 * 3600
    os.utime(model, (old, old))
    maintenance.clean()
    assert model.exists(), "voice models are downloads, not cache"


def test_disabled_cache_skips_the_interval_sweep_but_not_the_button(data_dir, monkeypatch):
    config._CACHE["cache"]["enabled"] = False
    _break("keep-me.mp3", age_s=10 * 24 * 3600)
    report = maintenance.clean()
    assert report["removed_files"] == 0 and "disabled" in report["skipped"]
    assert (config.DJ_CACHE_DIR / "keep-me.mp3").exists()
    forced = maintenance.clean(force=True)
    assert forced["removed_files"] >= 1


def test_stats_report_the_footprint(data_dir):
    _break("a.mp3", size=2048)
    _break("b.mp3", size=2048)
    stats = maintenance.cache_stats()
    assert stats["dj_files"] == 2 and stats["dj_bytes"] == 4096
    assert stats["enabled"] is True and stats["keep_files"] == 3


def test_maintainer_thread_starts_and_stops(data_dir):
    m = maintenance.Maintainer()
    m.start()
    try:
        assert m._thread is not None and m._thread.is_alive()
        assert m._thread.daemon, "housekeeping must never hold the process open"
    finally:
        m.stop()
    assert m._thread is None


def test_startup_sweep_runs_on_start(data_dir):
    _break("ancient.mp3", age_s=10 * 24 * 3600)
    m = maintenance.Maintainer()
    m.start()
    try:
        deadline = time.time() + 5
        while time.time() < deadline and (config.DJ_CACHE_DIR / "ancient.mp3").exists():
            time.sleep(0.05)
        assert not (config.DJ_CACHE_DIR / "ancient.mp3").exists(), "no startup sweep"
    finally:
        m.stop()
