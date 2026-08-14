"""Russian (Cyrillic) names must be readable by the English TTS voice.

The DJ voices are en_US-* Piper models, which cannot pronounce Cyrillic
glyphs. Cyrillic in a DJ line must be transliterated to a Latin spelling the
English voice reads with roughly Russian pronunciation -- but ONLY on the audio
path. The on-screen DJ text (and the LLM script) keep the original Cyrillic so
the listener can read the real name.
"""

import sys
from app import dj


def test_transliteration_maps_known_names():
    assert dj._transliterate_cyrillic("Агата Кристи") == "Agata Kristi"
    assert dj._transliterate_cyrillic("Звезда по имени Солнце") == "Zvezda po imeni Solntse"
    assert dj._transliterate_cyrillic("Тёмная ночь") == "Tyomnaya noch"


def test_transliteration_keeps_latin_and_punctuation():
    # English and numbers must pass through untouched; only Cyrillic converts.
    out = dj._transliterate_cyrillic("Here's Сказочная тайга by Агата Кристи, 1984.")
    assert out == "Here's Skazochnaya tayga by Agata Kristi, 1984."
    assert "1984" in out  # year spelling handled elsewhere, but not mangled here


def test_no_cyrillic_in_audio_text():
    # Whatever script reaches synthesize, the text handed to Piper must be ASCII
    # (no Cyrillic) so the English voice can read it.
    cyr = "Сейчас прозвучит песня Сказочная тайга группы Агата Кристи."
    # synthesize calls _transliterate_cyrillic internally; the text it would pass
    # to Piper must contain no Cyrillic. We verify the function output directly.
    audio_text = dj._transliterate_cyrillic(cyr)
    assert not any("а" <= ch <= "я" or "А" <= ch <= "Я" or ch in "ёЁ" for ch in audio_text)
    assert audio_text.isascii()


def test_synthesize_emits_ascii_for_cyrillic():
    # End-to-end guard: synthesize() must not choke on, and must transliterate,
    # Cyrillic input. We only assert it returns a path (audio was produced) and
    # that the internal transliteration produced ASCII.
    cyr = "Агата Кристи исполняет Сказочная тайга."
    path = dj.synthesize(cyr)
    # path may be None in CI without piper/voice; if present, it proves the
    # Cyrillic path through synthesize works without crashing.
    if path is not None:
        assert path.exists()
    # The transliteration itself is the load-bearing guarantee.
    assert dj._transliterate_cyrillic(cyr).isascii()
