"""The four housekeeping follow-ups: artist identity, config keys, enrichment, logs.

Each block covers one suggestion implemented after the validation sweep:

* unidentifiable artists share ONE identity, so the no-long-runs rule applies to
  a wall of untagged tracks instead of treating each as a fresh band;
* a config write with a key nothing reads is refused (422) instead of being
  stored and silently ignored;
* Wikipedia lookups are language-aware and remember their misses (Cyrillic names
  were probing English five times per track and always 404ing);
* logging goes to a bounded rotating file as well as the console.
"""

from __future__ import annotations

import logging
import time

import pytest

from app import config, enrich, library


# --- artist identity ---------------------------------------------------------

@pytest.mark.parametrize("artist", [
    None, "", "   ", "Unknown", "UNKNOWN ARTIST", "-", "?",
    "Неизвестный исполнитель", "неизвестный артист", "Various Artists", "V.A.",
    "07-Dva koncerta II. akustika",     # folder/track prefix that leaked into the tag
    "01 - something", "12_track",
])
def test_unidentifiable_artists_share_one_identity(artist):
    assert library.artist_key({"artist": artist}) == library._UNKNOWN_ARTIST_KEY


@pytest.mark.parametrize("artist", [
    "Pink Floyd", "30 Seconds to Mars", "Сектор Газа", "Blink-182", "Ноль",
    "The 1975", "65daysofstatic",
])
def test_real_artists_keep_their_own_identity(artist):
    key = library.artist_key({"artist": artist})
    assert key != library._UNKNOWN_ARTIST_KEY
    assert key == artist.strip().lower()


def test_unknown_artists_are_now_capped_like_any_other_artist():
    """Three untagged tracks in a row must not pass: they are 'the same group'."""
    tracks = [{"id": i, "artist": ""} for i in range(5)]
    ordered = library.interleave_artists(tracks, 2, None, 0, 5)
    assert len(ordered) == 2, "the cap applies to unidentifiable artists too"


# --- config key validation ---------------------------------------------------

def test_unknown_paths_flags_a_typo():
    assert config.unknown_paths({"playback": {"repeat_windows": 50}}) == ["playback.repeat_windows"]
    assert config.unknown_paths({"playback": {"repeat_window": 50}}) == []
    assert config.unknown_paths({"loudness": {"target": -14}}) == ["loudness.target"]
    assert config.unknown_paths({"loudness": {"target_lufs": -14}}) == []


def test_unknown_paths_walks_into_nested_dicts():
    bad = config.unknown_paths({"playback": {"program": {"sizes": 6, "size": 6}}})
    assert bad == ["playback.program.sizes"]


def test_unknown_paths_respects_allow_prefixes():
    assert config.unknown_paths({"legacy": {"gone": 1}},
                                allow_prefixes=("legacy",)) == []


def test_save_config_warns_but_still_stores(caplog, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "vdj.sqlite3")
    monkeypatch.setattr(config, "_CACHE", None)
    with caplog.at_level(logging.WARNING, logger="virtual_dj.config"):
        config.save_config({"playback": {"nope": 1}})
    assert any("unknown key" in r.message for r in caplog.records)


# --- enrichment: language-aware + negative cache -----------------------------

class _FakeResponse:
    def __init__(self, status, payload=None):
        self.status_code = status
        self._payload = payload or {}

    def json(self):
        return self._payload


class _FakeClient:
    """Records every URL, answers only for the Russian host."""

    def __init__(self, answer_for=("ru.wikipedia.org",)):
        self.calls: list[str] = []
        self.answer_for = answer_for

    def get(self, url):
        self.calls.append(url)
        if any(h in url for h in self.answer_for):
            return _FakeResponse(200, {"extract": "Сектор Газа — советская рок-группа.",
                                       "title": "Сектор Газа"})
        return _FakeResponse(404)


def test_cyrillic_names_are_looked_up_on_the_russian_wikipedia_first():
    client = _FakeClient()
    got = enrich._wikipedia(client, "Сектор_Газа")
    assert got["summary"].startswith("Сектор Газа")
    assert client.calls[0].startswith(enrich.WIKI_ROOT_RU)
    # The English-only disambiguators are skipped for Cyrillic terms: they 404 on
    # the real library and cost a request each.
    assert not any("_(band)" in c or "_(singer)" in c for c in client.calls)


def test_latin_names_still_use_the_english_wikipedia():
    client = _FakeClient(answer_for=("en.wikipedia.org",))
    enrich._wikipedia(client, "Queen")
    assert client.calls[0].startswith(enrich.WIKI_ROOT)


def test_a_miss_is_remembered_and_not_retried(monkeypatch):
    monkeypatch.setattr(enrich, "_WIKI_MISSES", {})
    client = _FakeClient(answer_for=())          # everything 404s
    assert enrich._wikipedia(client, "Никто_Никогда") == {}
    first_pass = len(client.calls)
    assert first_pass > 0
    # Second call for the same term: served from the miss cache, no requests.
    assert enrich._wikipedia(client, "Никто_Никогда") == {}
    assert len(client.calls) == first_pass


def test_a_miss_expires(monkeypatch):
    monkeypatch.setattr(enrich, "_WIKI_MISSES", {})
    client = _FakeClient(answer_for=())
    enrich._wikipedia(client, "Никто_Никогда")
    before = len(client.calls)
    enrich._WIKI_MISSES["Никто_Никогда"] = time.time() - enrich._WIKI_MISS_TTL_S - 1
    enrich._wikipedia(client, "Никто_Никогда")
    assert len(client.calls) > before, "an expired miss must be retried"


# --- rotating file log -------------------------------------------------------

def test_logging_writes_a_rotating_file(tmp_path, monkeypatch):
    import importlib

    import app as app_pkg

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "_CACHE", {"logging": {"to_file": True, "max_mb": 1, "backups": 2}})
    root = logging.getLogger()
    before = list(root.handlers)
    try:
        importlib.reload(app_pkg)          # re-runs the module-level setup
        logging.getLogger("virtual_dj.test").warning("hello from the log file")
        logfile = tmp_path / "logs" / "virtual-dj.log"
        assert logfile.exists(), "a rotating file log should be created"
        assert "hello from the log file" in logfile.read_text()
    finally:
        for handler in list(root.handlers):
            if handler not in before:
                root.removeHandler(handler)


def test_file_logging_can_be_switched_off(tmp_path, monkeypatch):
    import importlib

    import app as app_pkg

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "_CACHE", {"logging": {"to_file": False}})
    root = logging.getLogger()
    before = list(root.handlers)
    try:
        importlib.reload(app_pkg)
        from logging.handlers import RotatingFileHandler
        added = [h for h in root.handlers if h not in before]
        assert not any(isinstance(h, RotatingFileHandler) for h in added), \
            "logging.to_file: false must not add a file handler"
    finally:
        for handler in list(root.handlers):
            if handler not in before:
                root.removeHandler(handler)
