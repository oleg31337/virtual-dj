"""Loudness normalization: gain math, real measurement, analyzer, encoding.

The policy (target, true-peak cap, boost/cut limits) is pure math and is
asserted exhaustively; the measurement and the analyzer are exercised against
REAL ffmpeg-generated files, and the API surface is driven through TestClient.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from app import config, db, library, loudness
from app.stream import BROADCASTER, Broadcaster
from tests.conftest import make_mp3

# --- gain policy (pure) -----------------------------------------------------


def test_gain_follows_the_target():
    # -16 target, a track 4 dB quieter gets +4 dB back.
    assert loudness.gain_for(-20.0, None) == 4.0
    # The gain is always relative to the configured target.
    assert loudness.gain_for(-20.0, None, target_lufs=-14.0) == 6.0
    assert loudness.gain_for(-10.0, None) == -6.0


def test_true_peak_cap_prevents_clipping():
    # A quiet track that is already peaking hot cannot take the full boost:
    # +4 dB would push a -2.0 dBFS peak over the -1.5 ceiling, so the boost is
    # limited to +0.5 dB.
    assert loudness.gain_for(-20.0, -2.0) == 0.5
    # With real headroom the target wins.
    assert loudness.gain_for(-20.0, -10.0) == 4.0
    # A loud, already-clipped source is only ever attenuated.
    assert loudness.gain_for(-7.0, 1.5) == -9.0


def test_boost_and_cut_caps():
    # Never shout: a very quiet track is capped at max_boost_db.
    assert loudness.gain_for(-40.0, None) == 6.0
    # Never fully bury a track: capped at min_gain_db.
    assert loudness.gain_for(0.0, None) == -12.0


def test_unmeasurable_inputs_yield_no_gain():
    for bad in (None, float("nan"), float("-inf"), float("inf"), -80.0, -120.0):
        assert loudness.gain_for(bad) is None
    assert loudness.gain_for("nonsense") is None  # type: ignore[arg-type]


def test_settings_are_clamped_against_a_silly_config():
    config.save_config({"loudness": {"target_lufs": -500, "workers": 999,
                                     "max_boost_db": "loud", "window_seconds": -5}})
    s = loudness.settings()
    assert -30.0 <= s["target_lufs"] <= -6.0
    assert 1 <= s["workers"] <= 16
    assert s["max_boost_db"] == 6.0          # junk falls back to the default
    assert s["window_seconds"] >= 0


# --- measurement (real ffmpeg) ---------------------------------------------


def test_measure_real_file_with_head_window(has_ffmpeg, tmp_path):
    path = make_mp3(tmp_path / "tone.mp3", seconds=3.0)
    result = loudness.measure(path, window_seconds=2)
    assert result is not None
    assert result["seconds"] == 2                    # the head window
    assert -70.0 < result["lufs_i"] < 0.0
    assert result["true_peak_dbfs"] is not None


def test_measure_rejects_broken_and_missing_files(has_ffmpeg, tmp_path):
    broken = tmp_path / "broken.mp3"
    broken.write_bytes(b"this is not audio at all" * 50)
    assert loudness.measure(broken, window_seconds=5) is None
    assert loudness.measure(tmp_path / "nope.mp3", window_seconds=5) is None


def test_measure_tolerates_non_utf8_filenames(has_ffmpeg, tmp_path):
    """Real libraries contain names whose bytes are not valid UTF-8; the
    analyzer must not die decoding ffmpeg's stderr."""
    name = tmp_path / "b\xe9rk - junk.mp3"      # latin-1 bytes, not UTF-8
    path = make_mp3(name, seconds=3.0)
    assert loudness.measure(path, window_seconds=2) is not None


# --- analyzer (real files, real DB) ----------------------------------------


@pytest.fixture
def analyzed_library(music_dir, has_ffmpeg):
    """Scan a small library and drain the loudness queue with one worker."""
    config.save_config({"music_dir": str(music_dir)})
    library.scan_library(str(music_dir))
    assert loudness.ANALYZER.queue_size() == 3

    loudness.ANALYZER.start(workers=1)
    deadline = time.time() + 60
    while time.time() < deadline and loudness.ANALYZER.running:
        time.sleep(0.2)
    loudness.ANALYZER.stop()
    return music_dir


def test_analyzer_measures_every_track(analyzed_library):
    assert not loudness.ANALYZER.running
    assert loudness.ANALYZER.analyzed == 3
    assert loudness.ANALYZER.failed == 0
    assert loudness.ANALYZER.queue_size() == 0      # nothing left to do

    rows = db.connect().execute(
        "SELECT lufs_i, true_peak_dbfs, gain_db, loudness_analyzed_at, "
        "loudness_algo FROM tracks"
    ).fetchall()
    assert len(rows) == 3
    for row in rows:
        assert row["loudness_analyzed_at"] is not None
        assert row["loudness_algo"] == loudness.algo_for(120)
        if row["lufs_i"] is not None:               # measured
            assert -12.0 <= row["gain_db"] <= 6.0
        else:                                       # unmeasurable
            assert row["gain_db"] is None


def test_measured_gain_reaches_the_ui_query(analyzed_library):
    tracks = library.query_tracks(limit=10)
    assert len(tracks) == 3
    assert all("gain_db" in t and "lufs_i" in t for t in tracks)


def test_changing_the_window_requeues_the_library(analyzed_library):
    assert loudness.ANALYZER.queue_size() == 0
    # The analyzed window is part of the stored algo id: changing it must make
    # every track pending again instead of silently mixing two estimates.
    config.save_config({"loudness": {"window_seconds": 60}})
    assert loudness.ANALYZER.queue_size() == 3
    config.save_config({"loudness": {"window_seconds": 120}})


def test_reset_forgets_every_measurement(analyzed_library):
    assert loudness.ANALYZER.queue_size() == 0
    assert loudness.ANALYZER.reset() == 3
    assert loudness.ANALYZER.queue_size() == 3
    row = db.connect().execute(
        "SELECT COUNT(*) AS n FROM tracks WHERE gain_db IS NOT NULL"
    ).fetchone()
    assert row["n"] == 0


# --- broadcast encoding -----------------------------------------------------


def test_encode_args_use_static_gain_when_measured():
    args = Broadcaster()._encode_args("/tmp/x.mp3", gain_db=-4.2)
    filters = args[args.index("-af") + 1]
    assert filters == "volume=-4.20dB"
    assert "loudnorm" not in filters


def test_encode_args_fall_back_to_dynamic_loudnorm():
    # No measurement yet -> the legacy dynamic chain (unchanged behaviour).
    args = Broadcaster()._encode_args("/tmp/x.mp3")
    filters = args[args.index("-af") + 1]
    assert filters.startswith("loudnorm=")
    assert "I=-16.0" in filters


def test_disabled_normalization_ignores_measured_gains():
    config.save_config({"loudness": {"enabled": False}})
    args = Broadcaster()._encode_args("/tmp/x.mp3", gain_db=-4.2)
    assert "loudnorm" in args[args.index("-af") + 1]


def test_encode_args_tolerate_a_junk_gain():
    args = Broadcaster()._encode_args("/tmp/x.mp3", gain_db="not-a-number")  # type: ignore[arg-type]
    assert "loudnorm" in args[args.index("-af") + 1]


def test_play_file_uses_the_track_gain_only_for_music(music_dir, has_ffmpeg,
                                                      monkeypatch):
    b = Broadcaster()
    seen: dict[str, object] = {}
    original = b._encode_args

    def spy(path, extra_filters=None, gain_db=None):
        seen["gain_db"] = gain_db
        return original(path, extra_filters=extra_filters, gain_db=gain_db)

    monkeypatch.setattr(b, "_encode_args", spy)
    b.add_listener()

    music = {"path": str(music_dir / "a.mp3"), "title": "Alpha", "gain_db": -3.5}
    b._play_file(music["path"], "track", {"track": music})
    assert seen["gain_db"] == -3.5

    # The DJ voice keeps the dynamic chain (one Piper model -> consistent).
    b._play_file(music["path"], "dj",
                 {"track": music, "dj_text": "hello"})
    assert seen["gain_db"] is None


# --- HTTP API ---------------------------------------------------------------


@pytest.fixture
def loudness_client(music_dir, has_ffmpeg, monkeypatch):
    """Real app, broadcaster stubbed, loudness autostart OFF for determinism."""
    config.save_config({"music_dir": str(music_dir),
                        "loudness": {"autostart": False}})
    library.scan_library(str(music_dir))

    from app import server
    from app.scheduler import SCHEDULER

    monkeypatch.setattr(BROADCASTER, "start", lambda: None)
    monkeypatch.setattr(BROADCASTER, "stop", lambda: None)
    monkeypatch.setattr(SCHEDULER, "start", lambda: None)
    SCHEDULER.clear()
    with TestClient(server.app) as c:
        yield c
    loudness.ANALYZER.stop()


def test_loudness_status_endpoint(loudness_client):
    body = loudness_client.get("/api/loudness/status").json()
    assert body["running"] is False
    assert body["enabled"] is True
    assert body["queue"] == 3
    assert body["settings"]["target_lufs"] == -16.0
    # The UI's main status payload carries it too.
    assert "loudness" in loudness_client.get("/api/status").json()


def test_loudness_config_roundtrip(loudness_client):
    r = loudness_client.put("/api/config", json={
        "loudness": {"target_lufs": -14.0, "max_boost_db": 3.0, "workers": 2}})
    assert r.status_code == 200
    cfg = loudness_client.get("/api/config").json()["loudness"]
    assert cfg["target_lufs"] == -14.0
    assert cfg["max_boost_db"] == 3.0
    assert cfg["workers"] == 2
    assert cfg["window_seconds"] == 120            # untouched defaults survive


def test_analyze_stop_reset_endpoints(loudness_client, has_ffmpeg):
    started = loudness_client.post("/api/loudness/analyze",
                                   json={"workers": 1}).json()
    assert started["enabled"] is True
    # Wait for the pass to finish (three 1-second files).
    deadline = time.time() + 60
    while time.time() < deadline:
        status = loudness_client.get("/api/loudness/status").json()
        if not status["running"] and status["queue"] == 0:
            break
        time.sleep(0.3)
    status = loudness_client.get("/api/loudness/status").json()
    assert status["analyzed"] == 3
    assert status["queue"] == 0

    # Measured gains are visible to the library list the UI renders.
    tracks = loudness_client.get("/api/library/tracks").json()
    assert all(t["gain_db"] is not None for t in tracks)

    loudness_client.post("/api/loudness/stop")
    assert loudness_client.get("/api/loudness/status").json()["running"] is False

    reset = loudness_client.post("/api/loudness/reset").json()
    assert reset["reset"] == 3
    assert reset["status"]["queue"] == 3
