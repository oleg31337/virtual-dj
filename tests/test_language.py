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


def test_cyrillic_album_on_latin_song_is_not_russian():
    # A Cyrillic ALBUM next to a Latin title is a Russian-label compilation,
    # not a Russian song. The title is what matters for language origin.
    assert L.classify("How You Remind Me", "Nickelback", "Открытое Радио") == "english"


def test_proper_noun_umlauts_in_artist_are_not_german():
    # Umlauts in the ARTIST name (Icelandic/Finnish/Swedish proper nouns like
    # "Björk", "Mötley Crüe", "Ylönen") must not classify an English song as
    # German. Diacritic evidence comes from the TITLE only.
    assert L.classify("Play Dead", "Björk & David Arnold") == "english"
    assert L.classify("You're All I Need", "Mötley Crüe") == "english"


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


def test_language_from_country():
    assert L.language_from_country("FR") == "french"
    assert L.language_from_country("fr") == "french"
    assert L.language_from_country("france") == "french"
    assert L.language_from_country("ES") == "spanish"
    assert L.language_from_country("DE") == "german"
    assert L.language_from_country("RU") == "russian"
    assert L.language_from_country("UA") == "russian"
    # Non-mapped / unknown -> None (keeps English default).
    assert L.language_from_country("US") is None
    assert L.language_from_country("GB") is None
    assert L.language_from_country("") is None
    assert L.language_from_country(None) is None
    # Brazil/Portugal are NOT Spanish.
    assert L.language_from_country("BR") is None
    assert L.language_from_country("PT") is None


def test_artist_origin_tiebreaker_upgrades_country():
    # Téléphone: French diacritics in the name but a plain title -> text is
    # inconclusive (English). The artist's French origin must upgrade it.
    assert L.classify("Cendrillon", "Téléphone", artist_country="FR") == "french"
    # Without the country hint it would fall back to English.
    assert L.classify("Cendrillon", "Téléphone") == "english"
    # A confident text decision is NOT overridden by country.
    assert L.classify("Ne Me Quitte Pas", "Jacques Brel",
                      artist_country="US") == "french"
    # Unknown country keeps the English default.
    assert L.classify("Just Another Song", "Some Band",
                      artist_country="US") == "english"
