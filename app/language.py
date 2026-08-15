"""Local, dependency-free song-language classification.

Used to group the library into "origin/language" programs (English, French,
Spanish, German, Russian). The five buckets are fixed by product decision; any
track that does not clearly match one of them falls back to ``english`` (the
default), so there is no sprawling long tail of one-off languages.

The classifier is intentionally cheap and deterministic -- no network, no LLM.
Reliability comes mostly from *script* detection (Cyrillic => Russian) and from
highly distinctive diacritics / function words for the Latin-script languages.
It is good enough to keep a Russian, French, Spanish or German run coherent;
English is the safe default for anything ambiguous.

Only the title + artist + album are considered (the human-readable, usually
localised fields). Path fragments are ignored because folder names are often
not in the song's language.
"""

from __future__ import annotations

import re
from typing import Iterable

# The five supported buckets. English is the fallback default.
LANGUAGES = ("english", "french", "spanish", "german", "russian")
DEFAULT_LANGUAGE = "english"

_CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")

# Distinctive diacritics per language, split by reliability:
#  * STRONG  -- a single occurrence is an unambiguous marker of that language
#    (Spanish ñ, German ä/ö/ü/ß, French ç/æ/œ). One is enough to decide.
#  * WEAK    -- circumflex vowels (ê/â/î/ô/û). They occur in French but ALSO in
#    Italian and, crucially, in CP1252 mojibake (a stray "Ê" in "Dance Trax
#    vol.1 From DJ ×ÓÊ"). A lone weak diacritic therefore needs word support.
# é/á/í/ó/ú and à/è/ù are deliberately omitted: they are shared across
# French/Spanish/Portuguese/Italian and are common mojibake, so they are useless
# as a signal here.
_FRENCH_STRONG = set("çæœÇÆŒ")
_FRENCH_WEAK = set("êâîôûÊÂÎÔÛ")
_SPANISH_STRONG = set("ñ¿¡Ñ¿¡")
_GERMAN_STRONG = set("äöüßÄÖÜẞ")

# High-frequency function words (lowercased, matched as whole tokens). These
# catch tracks whose diacritics were stripped by bad tags but whose text is
# still clearly FR/ES/DE. Lists are generous on the most common, unambiguous
# words; rarer/shared words are intentionally omitted to avoid false positives.
_FRENCH_WORDS = {
    "le", "la", "les", "un", "une", "des", "du", "au", "aux", "et", "est",
    "dans", "pour", "avec", "sur", "par", "que", "qui", "je", "tu", "il",
    "elle", "nous", "vous", "ils", "elles", "mon", "ton", "son", "ma", "ta",
    "sa", "ce", "cet", "cette", "oui", "non", "mais", "entre", "sous", "vers",
    "depuis", "pas", "rien", "tout", "toute", "autre", "chez", "ne",
    "bien", "fait", "vous", "leur", "sans", "ou", "bon", "belle",
    "petit", "petite", "grand", "jolie", "amour", "coeur", "mer", "lune",
    "nuit", "jour", "vie", "mort", "monde", "temps", "guerre", "paix",
    "lumiere", "lumière", "soleil", "femme", "homme", "enfant", "rue",
    "maison", "ville", "pays", "roi", "reine", "dieu", "en",
}
_SPANISH_WORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "y", "es", "en",
    "de", "del", "que", "por", "para", "con", "mi", "tu", "su", "mis", "tus",
    "sus", "pero", "muy", "como", "mas", "más", "dia", "día", "año", "cada",
    "nos", "os", "me", "te", "se", "no", "si", "lo", "le", "amor", "corazon",
    "corazón", "senor", "señor", "mujer", "hombre", "noche", "vida", "mundo",
    "tiempo", "cielo", "mar", "sol", "luna", "madre", "padre", "fuego", "agua",
    "tierra", "alma", "paz", "guerra", "luz", "flor", "estrella", "camino",
    "sueno", "sueño", "beso", "mano", "guitarra", "bailar", "cantar", "son",
    "eres", "soy", "era", "eran", "cuando", "donde", "siempre", "nada",
    "todo", "algo", "quiero", "mucho", "bésame", "besame", "consuelo", "corazones",
    "amores", "querido", "querida", "bella", "bello", "pobre", "libre", "frio",
    "frío", "calor", "lluvia", "viento", "playa", "monte", "río", "rio", "sangre",
    "pueblo", "ciudad", "barco", "camisa", "sombra", "eco", "silencio", "risa",
    "lágrima", "lagrima", "sueños", "suenos", "corazón", "cariño", "mirada",
    "animal", "naturaleza", "verde", "oscuro", "claro", "dulce", "fuerte",
    "feliz", "triste", "lejos", "cerca", "tarde", "mañana", "manana", "nunca",
    "siempre", "ayer", "hoy", "aquí", "aqui", "alli", "allí", "contigo", "conmigo",
    "nuestro", "nuestra", "tuyo", "tuya", "mío", "mia", "míos", "vos", "usted",
}
_GERMAN_WORDS = {
    "der", "die", "das", "und", "ich", "ein", "eine", "einer", "nicht",
    "ist", "auf", "mit", "fur", "für", "dich", "du", "wir", "sie", "er", "es",
    "auch", "sehr", "wie", "was", "warum", "weil", "uber", "über", "unter",
    "nach", "von", "im", "am", "dem", "den", "des", "einem", "einen", "schon",
    "noch", "nur", "schone", "schöne", "schoner", "schöner", "liebe", "herz",
    "welt", "nacht", "zeit", "leben", "traum", "kind", "mensch", "seele",
    "komm", "kommt", "hier", "tot", "toten", "hast", "habe", "hat", "mein",
    "dein", "sein", "sind", "war", "gibt", "nichts", "alles", "licht",
    "feuer", "wasser", "erde", "himmel", "stimme", "lied", "blick", "weg",
    "ende", "anfang", "morgen", "abend", "stadt", "haus", "freund", "freiheit",
    "krieg", "frieden", "sonne", "mond", "tier", "wort", "wahrheit", "blut",
    "stahl", "winter", "sommer", "schlag", "schlagt", "deine", "meine", "unsere",
    "eure", "euer", "ihr", "ihre", "sein", "seine", "klein", "gross", "groß",
    "schwarz", "weiss", "weiß", "blau", "rot", "grun", "grün", "tag", "jahre",
    "jahr", "wege", "wege", "tief", "hoch", "warm", "kalt", "neu", "alt",
    "jung", "frei", "stark", "sanft", "laut", "leise", "schnell", "langsam",
}

_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ¿¡]+")


def _token_scores(text: str) -> dict[str, int]:
    """Return per-language word-hit scores for the given Latin text.

    Longer, more distinctive words (>=4 letters) count double so that a couple
    of tiny ambiguous syllables (``me``, ``la``) in an otherwise-English title
    don't flip the classification, while a real French/Spanish sentence still
    accumulates a clear lead.
    """
    low = text.lower()
    tokens = set(_WORD_RE.findall(low))
    scores = {"french": 0, "spanish": 0, "german": 0}
    for tok in tokens:
        for lang, words in (("french", _FRENCH_WORDS), ("spanish", _SPANISH_WORDS),
                            ("german", _GERMAN_WORDS)):
            if tok in words:
                scores[lang] += 2 if len(tok) >= 4 else 1
    return scores


def classify(title: str = "", artist: str = "", album: str = "") -> str:
    """Classify a track's language into one of ``LANGUAGES``.

    Cyrillic anywhere in the text => Russian -- UNLESS the text is mostly Latin,
    which means the Cyrillic is almost certainly mojibake (a corrupt accented
    Latin name like "Cortйge" for "Cortège"), not a Russian song. In that case
    the stray Cyrillic is ignored and the Latin heuristics decide.

    Otherwise Latin text is scored by distinctive diacritics and function words;
    the highest-scoring language wins only if it clears a threshold AND leads
    the runner-up, else English (the default).
    """
    parts = [title or "", artist or "", album or ""]
    joined = " ".join(p for p in parts if p)

    cyr = _CYRILLIC_RE.findall(joined)
    if cyr:
        # Treat as Russian only when Cyrillic is a substantial part of the text;
        # a lone stray glyph in an otherwise-Latin string is mojibake noise.
        latin = re.findall(r"[A-Za-zÀ-ÿ¿¡]", joined)
        if len(cyr) >= max(2, len(latin) * 0.25):
            return "russian"
        # else: fall through to Latin heuristics (ignore the mojibake glyph)

    diac = {"french": 0, "spanish": 0, "german": 0}
    strong = {"french": 0, "spanish": 0, "german": 0}
    weak = {"french": 0, "spanish": 0, "german": 0}
    for ch in joined:
        if ch in _FRENCH_STRONG:
            diac["french"] += 1; strong["french"] += 1
        elif ch in _FRENCH_WEAK:
            diac["french"] += 1; weak["french"] += 1
        if ch in _SPANISH_STRONG:
            diac["spanish"] += 1; strong["spanish"] += 1
        if ch in _GERMAN_STRONG:
            diac["german"] += 1; strong["german"] += 1

    words = _token_scores(joined)
    # Weight diacritics heavily (they are near-unambiguous) and words by length.
    scores = {
        "french": diac["french"] * 3 + words["french"],
        "spanish": diac["spanish"] * 3 + words["spanish"],
        "german": diac["german"] * 3 + words["german"],
    }
    best = max(scores, key=lambda k: scores[k])
    # A language wins only with a real lead (>=3 and strictly above every other
    # language) AND with supporting evidence:
    #   * a strong (unambiguous) diacritic of that language, OR
    #   * >=2 word-points (so short function words like "ne", "por" still count), OR
    #   * >=2 weak diacritics of that language (mojibake rarely repeats one cleanly).
    # This keeps a single mojibake circumflex ("Ê" in "Dance Trax ... ×ÓÊ") from
    # being read as French when the text has no other French signal. English is
    # the fallback.
    if scores[best] >= 2 and scores[best] > max(
        v for k, v in scores.items() if k != best
    ) and (strong[best] >= 1 or words[best] >= 2 or weak[best] >= 2):
        return best
    return DEFAULT_LANGUAGE


def classify_track(track: dict) -> str:
    """Classify a track dict (keys: title/artist/album)."""
    return classify(
        title=(track.get("title") or ""),
        artist=(track.get("artist") or ""),
        album=(track.get("album") or ""),
    )


def tally(tracks: Iterable[dict]) -> dict[str, int]:
    """Count tracks per language (for the programs/themes panel)."""
    out = {lang: 0 for lang in LANGUAGES}
    for tr in tracks:
        lang = (tr.get("language") or DEFAULT_LANGUAGE)
        out[lang] = out.get(lang, 0) + 1
    return out
