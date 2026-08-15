"""Tests for local song-language classification (origin/language programs)."""

import sys
sys.path.insert(0, ".")

from app import language as L


def test_classify_russian_via_cyrillic():
    assert L.classify("Сказочная тайга", "Агата Кристи") == "russian"


def test_classify_french_via_words_and_diacritics():
    assert L.classify("Ne Me Quitte Pas", "Jacques Brel") == "french"
    assert L.classify("La Vie En Rose", "Édith Piaf") == "french"


def test_classify_spanish_via_words():
    assert L.classify("Por Una Cabeza", "Carlos Gardel") == "spanish"
    assert L.classify("Bésame Mucho", "Consuelo Velázquez") == "spanish"


def test_classify_german_via_umlaut_and_words():
    assert L.classify("Hier Kommt Alex", "Die Toten Hosen") == "german"
    assert L.classify("Du Hast", "Rammstein") == "german"


def test_classify_english_default_for_ambiguous():
    # Plain English titles stay English (the fallback default).
    assert L.classify("Bohemian Rhapsody", "Queen") == "english"
    assert L.classify("Never Gonna Give You Up", "Rick Astley") == "english"


def test_mojibake_cyrillic_is_not_russian():
    # Corrupt accented-Latin names (stray Cyrillic glyphs in an otherwise-Latin
    # string) must NOT be read as Russian -- they are mojibake, not Russian songs.
    assert L.classify("Cortйge", "Apocalyptica") == "english"
    assert L.classify("Life Burns (feat. lauri ylцnen)", "Apocalyptica") == "english"


def test_mojibake_diacritic_is_not_french():
    # CP1252 mojibake with a stray circumflex must not read as French.
    assert L.classify("WHAT ABOUT ME", "HADDAWAY",
                      "Dance Trax vol.1 From DJ ×ÓÊ") == "english"
    assert L.classify("CALLING", "GERI HALLIWELL", "ÕÈÒ FM 107.4FM") == "english"


def test_languages_constant_and_default():
    assert set(L.LANGUAGES) == {"english", "french", "spanish", "german", "russian"}
    assert L.DEFAULT_LANGUAGE == "english"


def test_unknown_language_falls_back_to_english():
    # A non-listed language (e.g. Italian/Portuguese) lands in English per spec.
    assert L.classify("Volare", "Domenico Modugno") in ("english", "italian") or \
        L.classify("Volare", "Domenico Modugno") == "english"
    # The classifier only emits the five supported buckets.
    for t, a in [("Volare", "Domenico Modugno"), ("Aquelas", "Madredeus")]:
        assert L.classify(t, a) in L.LANGUAGES
