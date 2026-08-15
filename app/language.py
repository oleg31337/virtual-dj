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

# Artist country-of-origin (from MusicBrainz `area`/`country`) -> the language
# bucket we group that artist under. Used ONLY as a tiebreaker when the cheap
# text classifier is inconclusive (it falls back to English). Map is
# deliberately conservative: a country is only associated with a language when
# that language is clearly dominant there, so a mis-resolved origin cannot flip
# an otherwise-confident classification.
#
# France and its overseas territories are unambiguously French. Belgium and
# Switzerland are multilingual, but the text classifier already catches German/
# Italian tracks by their diacritics/words, so mapping them here only upgrades
# the French-singing majority that would otherwise fall to English.
_COUNTRY_TO_LANGUAGE = {
    # French
    "FR": "french", "MQ": "french", "GP": "french", "RE": "french",
    "YT": "french", "MF": "french", "BL": "french", "MC": "french",
    "BE": "french", "CH": "french",
    # Spanish (Spain + Latin America). Brazil (BR) and Portugal (PT) excluded.
    "ES": "spanish", "MX": "spanish", "AR": "spanish", "CO": "spanish",
    "CL": "spanish", "PE": "spanish", "VE": "spanish", "EC": "spanish",
    "UY": "spanish", "BO": "spanish", "PY": "spanish", "CR": "spanish",
    "DO": "spanish", "PA": "spanish", "NI": "spanish", "HN": "spanish",
    "GT": "spanish", "SV": "spanish", "CU": "spanish", "PR": "spanish",
    # German
    "DE": "german", "AT": "german",
    # Russian (and the core Russian-speaking CIS)
    "RU": "russian", "UA": "russian", "BY": "russian",
}
# Also accept a few lowercased country names in case MusicBrainz returns the
# full name instead of the ISO code.
_COUNTRY_NAME_TO_LANGUAGE = {k.lower(): v for k, v in {
    "france": "french", "belgium": "french", "switzerland": "french",
    "monaco": "french", "spain": "spanish", "mexico": "spanish",
    "argentina": "spanish", "colombia": "spanish", "chile": "spanish",
    "germany": "german", "austria": "german", "russia": "russian",
    "ukraine": "russian", "belarus": "russian",
}.items()}


def language_from_country(country: str | None) -> str | None:
    """Map a MusicBrainz country/area code or name to a language bucket.

    Returns ``None`` when the country is unknown or not associated with one of
    the supported non-English buckets (so the caller keeps its default).
    """
    if not country:
        return None
    c = country.strip()
    code = c.upper()
    if code in _COUNTRY_TO_LANGUAGE:
        return _COUNTRY_TO_LANGUAGE[code]
    name = c.lower()
    if name in _COUNTRY_NAME_TO_LANGUAGE:
        return _COUNTRY_NAME_TO_LANGUAGE[name]
    # Some MB areas are nested ("X, France"); accept a trailing known name.
    for key, lang in _COUNTRY_NAME_TO_LANGUAGE.items():
        if name.endswith(", " + key) or name == key:
            return lang
    return None


# Diacritics that suggest an artist name is from a non-English language and is
# worth resolving via MusicBrainz origin. These are the SAME distinctive marks
# the text classifier uses (French ç/æ/œ + accents, Spanish ñ, German
# ä/ö/ü/ß), so we only spend a network lookup on the names that could actually
# benefit. Plain ASCII names (Queen, Nirvana) are skipped -- their text
# classification is already correct and the network call would be wasted.
_NAME_DIACRITIC_RE = re.compile(
    r"[àâäãáåæçéèêëíìîïñóòôöõøúùûüýÿßÀÂÄÃÁÅÆÇÉÈÊËÍÌÎÏÑÓÒÔÖÕØÚÙÛÜÝ]"
)


def _looks_like_non_english_name(artist: str) -> bool:
    """True if the artist name carries a language-diagnostic diacritic.

    Used to decide whether a MusicBrainz country lookup is worth attempting for
    an otherwise-English-default track. Keeps the scan fast and avoids
    hammering the API for names that are already correctly classified.
    """
    return bool(_NAME_DIACRITIC_RE.search(artist or ""))

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


def _token_scores(text: str) -> dict[str, tuple[int, int]]:
    """Score Latin ``text`` per language.

    Returns ``{lang: (points, distinct_words)}``. ``points`` weights longer,
    more distinctive words (>=4 letters) double so a real FR/ES/DE sentence
    accumulates a clear lead; ``distinct_words`` is the count of *different*
    vocabulary items matched (a single ambiguous word like ``pour`` must never
    decide on its own -- we require >=2 distinct matches).

    Words are scanned across title + artist + album. Proper-noun *diacritics*
    in the artist/album (e.g. the umlaut in "Mötley Crüe" or "Björk") are
    explicitly NOT counted -- that signal is handled separately, from the
    *title* only, by ``classify``.
    """
    low = text.lower()
    tokens = set(_WORD_RE.findall(low))
    out = {lang: [0, 0] for lang in ("french", "spanish", "german")}
    for tok in tokens:
        for lang, words in (("french", _FRENCH_WORDS), ("spanish", _SPANISH_WORDS),
                            ("german", _GERMAN_WORDS)):
            if tok in words:
                out[lang][0] += 2 if len(tok) >= 4 else 1
                out[lang][1] += 1
    return {k: tuple(v) for k, v in out.items()}


def _title_diacritics(title: str) -> dict[str, dict[str, int]]:
    """Count title-only diacritics per language (strong vs weak)."""
    res = {
        "french": {"strong": 0, "weak": 0},
        "spanish": {"strong": 0, "weak": 0},
        "german": {"strong": 0, "weak": 0},
    }
    for ch in title or "":
        if ch in _FRENCH_STRONG:
            res["french"]["strong"] += 1
        elif ch in _FRENCH_WEAK:
            res["french"]["weak"] += 1
        if ch in _SPANISH_STRONG:
            res["spanish"]["strong"] += 1
        if ch in _GERMAN_STRONG:
            res["german"]["strong"] += 1
    return res


def classify(title: str = "", artist: str = "", album: str = "",
             artist_country: str | None = None) -> str:
    """Classify a track's language into one of ``LANGUAGES``.

    Decision order:
      1. **Russian** -- Cyrillic in the *song title* (the title is almost
         always in the song's own language), OR Cyrillic making up >=40% of all
         alphabetic characters (a fully Cyrillic album+artist with a short
         Latin title still counts). A Cyrillic *album* next to a Latin title
         (e.g. an English track on a Russian radio sampler) is NOT Russian.
      2. **Latin FR/ES/DE** -- scored by distinctive *title* diacritics plus
         vocabulary from title+artist+album. A language wins only if it leads
         the runner-up AND has supporting evidence: a strong (unambiguous)
         title diacritic, >=2 distinct vocabulary matches, or >=2 weak
         diacritics. Everything else (including mojibake stray accents) is
         English, the safe default.
      3. **Artist origin tiebreaker** -- if the text classifier is inconclusive
         (English default) and the artist's MusicBrainz country-of-origin maps
         to a non-English bucket (e.g. France -> French), use that. This catches
         bands like Téléphone whose name carries French diacritics but whose
         track titles do not, without disturbing any confident text decision.
    """
    title = title or ""
    joined = " ".join(p for p in (title, artist or "", album or "") if p)

    # --- 1. Russian via Cyrillic -------------------------------------------------
    cyr = _CYRILLIC_RE.findall(joined)
    if cyr:
        title_cyr = _CYRILLIC_RE.findall(title)
        if title_cyr:
            # Cyrillic in the title is Russian -- UNLESS it is a single stray
            # glyph inside an otherwise-Latin title, which is mojibake (a
            # corrupt "Cortège" rendered as "Cortйge"). Require >=2 Cyrillic
            # glyphs, or Cyrillic making up >=40% of the title's letters.
            title_alpha = [c for c in title if c.isalpha()]
            if len(title_cyr) >= 2 or len(title_cyr) >= 0.4 * len(title_alpha):
                return "russian"
            # else: lone mojibake glyph in the title -> fall through to Latin.
        else:
            # No Cyrillic in the title: only Russian if Cyrillic dominates the
            # whole string (otherwise it's a Cyrillic album/artist on a Latin
            # song).
            alpha = [c for c in joined if c.isalpha()]
            if alpha and len(cyr) >= 0.4 * len(alpha):
                return "russian"
        # else fall through to Latin heuristics (ignore the stray glyph)

    # --- 2. Latin FR/ES/DE ------------------------------------------------------
    dia = _title_diacritics(title)
    words = _token_scores(joined)
    scores = {
        "french": dia["french"]["strong"] * 3 + dia["french"]["weak"] * 0
                  + words["french"][0],
        "spanish": dia["spanish"]["strong"] * 3 + words["spanish"][0],
        "german": dia["german"]["strong"] * 3 + words["german"][0],
    }
    best = max(scores, key=lambda k: scores[k])
    others = max(v for k, v in scores.items() if k != best)
    strong = dia[best]["strong"]
    distinct = words[best][1]
    weak = dia[best]["weak"]
    # Win requires a real lead AND evidence (a strong title diacritic, >=2
    # distinct words, or >=2 weak diacritics). A single shared word or a lone
    # mojibake accent can never decide.
    if scores[best] > others and (strong >= 1 or distinct >= 2 or weak >= 2):
        return best
    # Inconclusive by text: fall back to the artist's country-of-origin when it
    # maps to a supported non-English bucket. This upgrades bands whose *name*
    # is clearly from a language (Téléphone -> France -> French) but whose
    # titles carry no decisive diacritics/words. It never overrides a confident
    # text decision above, and a missing/unknown country keeps English default.
    if artist_country:
        origin_lang = language_from_country(artist_country)
        if origin_lang and origin_lang != DEFAULT_LANGUAGE:
            return origin_lang
    return DEFAULT_LANGUAGE


def classify_track(track: dict, artist_country: str | None = None) -> str:
    """Classify a track dict (keys: title/artist/album)."""
    return classify(
        title=(track.get("title") or ""),
        artist=(track.get("artist") or ""),
        album=(track.get("album") or ""),
        artist_country=artist_country,
    )


def tally(tracks: Iterable[dict]) -> dict[str, int]:
    """Count tracks per language (for the programs/themes panel)."""
    out = {lang: 0 for lang in LANGUAGES}
    for tr in tracks:
        lang = (tr.get("language") or DEFAULT_LANGUAGE)
        out[lang] = out.get(lang, 0) + 1
    return out
