"""Unit tests for the pieces that do not need model weights."""

from __future__ import annotations

import pytest

from ttssvc import config, schema
from ttssvc.engines.piper import PiperEngine
from ttssvc.engines.supertonic import Supertonic3Engine

SCHEMA = [
    {"key": "speed", "type": "float", "min": 0.5, "max": 2.0, "default": 1.0},
    {"key": "steps", "type": "int", "min": 2, "max": 8, "default": 5},
    {"key": "mode", "type": "enum", "default": "a",
     "choices": [{"value": "a"}, {"value": "b"}]},
    {"key": "loud", "type": "bool", "default": False},
]


def test_defaults_cover_every_param():
    assert schema.defaults(SCHEMA) == {
        "speed": 1.0, "steps": 5, "mode": "a", "loud": False}


def test_resolve_clamps_and_reports():
    out = schema.resolve(SCHEMA, {"speed": -1, "steps": 99})
    assert out.values["speed"] == 0.5
    assert out.values["steps"] == 8
    assert out.ignored == []
    assert sorted(out.clamped) == ["speed", "steps"]


def test_resolve_ignores_unknown_keys():
    out = schema.resolve(SCHEMA, {"noise_scale": 0.7})
    assert out.values["speed"] == 1.0
    assert out.ignored == ["noise_scale"]


def test_resolve_coerces_strings_and_ints():
    out = schema.resolve(SCHEMA, {"speed": "1.25", "steps": "6.4"})
    assert out.values["speed"] == 1.25
    assert out.values["steps"] == 6


def test_resolve_enum_and_bool():
    out = schema.resolve(SCHEMA, {"mode": "b", "loud": "true"})
    assert out.values["mode"] == "b"
    assert out.values["loud"] is True
    out = schema.resolve(SCHEMA, {"mode": "zzz", "loud": 0})
    assert out.values["mode"] == "a"
    assert out.values["loud"] is False
    assert out.ignored == ["mode"]


def test_resolve_rejects_wrong_type():
    with pytest.raises(schema.ParamError):
        schema.resolve(SCHEMA, {"speed": {"nope": 1}})
    with pytest.raises(schema.ParamError):
        schema.resolve(SCHEMA, {"speed": "fast"})
    with pytest.raises(schema.ParamError):
        schema.resolve(SCHEMA, {"loud": "maybe"})


def test_none_values_leave_the_default():
    out = schema.resolve(SCHEMA, {"speed": None})
    assert out.values["speed"] == 1.0
    assert out.clamped == []


# --- engine metadata ------------------------------------------------------

def test_supertonic_declares_ten_voices_over_en_and_ru():
    engine = Supertonic3Engine(config)
    voices = engine.voices()
    assert [v["id"] for v in voices] == [
        "M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"]
    assert all(v["languages"] == ["en", "ru"] for v in voices)
    assert [lang["code"] for lang in engine.languages()] == ["en", "ru"]
    params = {p["key"]: p for p in engine.params()}
    assert set(params) == {"speed", "sentence_pause"}
    assert params["speed"]["min"] < params["speed"]["default"] < params["speed"]["max"]
    # steps are a deployment setting, never a user parameter
    assert "steps" not in params
    assert engine.text_support()["stress_marks"] is False


def test_supertonic_sanitizes_characters_it_cannot_index():
    engine = Supertonic3Engine(config)
    # A tiny stand-in indexer: mark the characters the model does know.
    indexer = [-1] * 65536
    for ch in "abcxyz-., !/?0123456789":
        indexer[ord(ch)] = 5
    for ch in "абвгдежзиклмнопрстуфхцчшщъыьэюя":
        indexer[ord(ch)] = 7
    for ch in "\u0306\u0308":  # breve and diaeresis: NFKD parts of й and ё
        indexer[ord(ch)] = 8
    engine._indexer = indexer

    out = engine._sanitize("Далее — трек\u2026 ёлка, йод!")
    assert "\u2014" not in out and "\u2026" not in out
    # em dash became a hyphen-like separator, ellipsis a full stop
    assert "-" in out and "..." not in out
    assert "ёлка".replace("ё", "е")[:1] in out  # sanity: text survived
    assert "  " not in out
    # a character with no embedding is dropped, not passed through as -1
    indexer = [-1] * 65536
    indexer[ord("a")] = 5
    indexer[ord("b")] = 5
    engine._indexer = indexer
    assert engine._sanitize("a\u2603b") == "a b"


def test_piper_voice_ids_split_by_language():
    engine = PiperEngine(config)
    voices = {v["id"]: v for v in engine.voices()}
    assert voices["en_GB-alan-medium"]["languages"] == ["en"]
    assert voices["ru_RU-irina-medium"]["languages"] == ["ru"]
    assert [lang["code"] for lang in engine.languages()] == ["en", "ru"]
    assert set(p["key"] for p in engine.params()) == {
        "speed", "expressiveness", "rhythm", "sentence_pause"}
    assert engine.text_support()["stress_marks"] is True
    assert engine.model_ready()[0] is False


def test_piper_urls_follow_the_huggingface_layout():
    from ttssvc.engines.piper import _voice_urls

    onnx, sidecar = _voice_urls("ru_RU-irina-medium")
    assert onnx.endswith("/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx")
    assert sidecar.endswith(".onnx.json")


def test_steps_are_clamped_by_config(monkeypatch):
    assert config.steps() == 5
    monkeypatch.setenv("TTS_STEPS", "1")
    assert config.steps() == 2
    monkeypatch.setenv("TTS_STEPS", "99")
    assert config.steps() == 8
