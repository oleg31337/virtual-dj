"""Tests for local-AI metadata helpers and the corrupt-vs-nonlatin policy."""

from __future__ import annotations

import app.ai_meta as ai_meta
from app import textq


def test_cyrillic_names_are_usable_not_corrupt():
    # A clean Cyrillic title/artist must be playable now.
    assert textq.is_usable("Би-2")
    assert textq.is_usable("Звери")
    assert not textq.is_corrupt("Мумий Тролль")


def test_mojibake_is_corrupt_but_not_plain_nonlatin():
    # Mis-decoded Cyrillic bytes are corrupt and should be rejected.
    mojibake = "Ñèíàãîãàëüíàÿ"
    assert textq.is_corrupt(mojibake)
    assert not textq.is_usable(mojibake)
    # A clean Greek title is fine.
    assert textq.is_usable("Αδελφότητα")


def test_rejection_reason_corrupt_not_nonlatin():
    assert textq.rejection_reason("Ñèíàãîãàëüíàÿ", "Title") == "corrupt"
    assert textq.rejection_reason("Artist", None) == "no_title"
    # Cyrillic alone is NOT a rejection reason.
    assert textq.rejection_reason("Би-2", "Полковнику никто") is None


def test_infer_genre_picks_canonical(monkeypatch):
    # The local LLM returns a free-form genre; we canonicalise it.
    monkeypatch.setattr(ai_meta, "_ollama_chat",
                        lambda *a, **k: "synth-pop, obviously")
    assert ai_meta.infer_genre("Depeche Mode", "Enjoy the Silence") == "Pop"


def test_infer_genre_caches(monkeypatch):
    calls = []

    def fake(system, user, **k):
        calls.append(user)
        return "Metal"

    monkeypatch.setattr(ai_meta, "_ollama_chat", fake)
    # Cache disabled so we exercise both the call and the cache path is tested
    # by the second identical call returning the same without a new LLM hit.
    g1 = ai_meta.infer_genre("Metallica", "One", use_cache=True)
    g2 = ai_meta.infer_genre("Metallica", "One", use_cache=True)
    assert g1 == g2 == "Metal"
    # Two identical inputs must hit the model once when cached.
    assert len(calls) == 1


def test_recover_names_splits_and_romanises(monkeypatch):
    payload = (
        '{"artist": "Basta", "title": "Sansara", "album": null, '
        '"language": "cyrillic", "confident": true}'
    )
    monkeypatch.setattr(ai_meta, "_ollama_chat", lambda *a, **k: payload)
    res = ai_meta.recover_names("01bastasansara", None, path="/x/01bastasansara.mp3")
    assert res["artist"] == "Basta"
    assert res["title"] == "Sansara"
    assert res["confident"] is True


def test_recover_names_handles_bad_json(monkeypatch):
    monkeypatch.setattr(ai_meta, "_ollama_chat",
                        lambda *a, **k: "I cannot parse this track")
    res = ai_meta.recover_names("garbage", "track 01")
    assert res["artist"] is None
    assert res["confident"] is False


def test_health_when_llm_disabled(monkeypatch):
    monkeypatch.setattr(ai_meta.config, "get",
                        lambda dot, d=None: False if dot == "llm.enabled" else d)
    assert ai_meta.health()["enabled"] is False


def test_infer_genre_none_when_disabled(monkeypatch):
    monkeypatch.setattr(ai_meta.config, "get",
                        lambda dot, d=None: False if dot == "ai.free_text_genre" else d)
    assert ai_meta.infer_genre("X", "Y") is None
