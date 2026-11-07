"""The DJ brain: turns track metadata into a spoken intro.

Pipeline per track:
    metadata (+ enrichment facts) -> LLM script -> TTS service -> mp3 segment

Both stages degrade gracefully: with no LLM we fall back to a templated line,
and if the TTS service fails the stream simply skips the DJ break rather than
stalling. Speech synthesis itself lives in the separate ``tts`` container; this
module owns the words and the audio mastering, never an engine.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx

from . import config, db, enrich, ru_numbers, tts_client

log = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Piper (and most TTS) read bare digits as individual numerals — "1984" comes
# out as "one nine eight four" instead of a year. Years are converted to words
# before synthesis so the voice model reads them as dates, not digits.
_YEAR_RE = re.compile(r"\b(19|20)(\d{2})\b")

_UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
          "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
          "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]


def _int_to_words(n: int) -> str:
    if n < 20:
        return _UNITS[n]
    if n < 100:
        t, r = divmod(n, 10)
        return _TENS[t] + (f"-{_UNITS[r]}" if r else "")
    if n < 1000:
        h, r = divmod(n, 100)
        return _UNITS[h] + " hundred" + (f" {_int_to_words(r)}" if r else "")
    if 1900 <= n < 2000:
        # 19xx reads as "nineteen" + the two-digit tail (e.g. 1984 -> "nineteen eighty-four").
        return f"nineteen {_int_to_words(n - 1900)}"
    if n < 2100:
        # 20xx: colloquial "twenty twenty-three" (2023), but tails below ten
        # would read as "twenty seven" (sounding like 27) — say "two thousand
        # seven" for those so the year is unambiguous.
        tail = n - 2000
        if tail == 0:
            return "two thousand"
        if tail < 10:
            return f"two thousand {_int_to_words(tail)}"
        return f"twenty {_int_to_words(tail)}"
    # Fallback for anything else.
    return " ".join(_int_to_words(int(d)) for d in str(n))


# Stress-mark characters a language model may write. Only U+0301 (COMBINING
# ACUTE) placed immediately AFTER the stressed vowel is obeyed by espeak-ng.
# Measured on the bundled Russian voice: "замо\u0301к" -> zˈɑmʌk (correct) while
# the plain acute "замо\u00b4к" is ignored, and U+02B9/U+02BC are read ALOUD as
# their names ("zˈɛ a ɪdʒˈɛktɪv ˈɛm ʌ k" for "за\u02b9мок") — an audible disaster
# that the old prompt actively invited by asking for "´".
_COMBINING_ACUTE = "\u0301"
_STRESS_MARKS = "\u00b4\u2019\u02bc\u02b9\u0060\u2032\u0301"
_CYRILLIC_VOWELS = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"


def _normalize_stress_marks(text: str) -> str:
    """Rewrite whatever stress mark the model wrote into the form espeak obeys.

    Every mark is attached to its adjacent Cyrillic vowel as U+0301, moved to
    the far side of the vowel when the model put it in front, and dropped when
    there is no vowel to attach to. ``замо\u00b4к`` -> ``замо\u0301к`` (obeyed),
    ``з\u00b4амок`` -> ``за\u0301мок``, ``за\u02b9мок`` -> ``замок``.
    """
    if not text:
        return text
    out: list[str] = []
    index = 0
    while index < len(text):
        ch = text[index]
        if ch not in _STRESS_MARKS:
            out.append(ch)
            index += 1
            continue
        if ch == _COMBINING_ACUTE:
            # Already the obeyed form; it is meaningless unless a vowel precedes.
            if out and out[-1][0] in _CYRILLIC_VOWELS:
                out[-1] = out[-1][0] + _COMBINING_ACUTE
            index += 1
            continue
        following = text[index + 1] if index + 1 < len(text) else ""
        if following in _CYRILLIC_VOWELS:
            out.append(following + _COMBINING_ACUTE)      # mark written first
            index += 2
            continue
        if out and out[-1][0] in _CYRILLIC_VOWELS:
            out[-1] = out[-1][0] + _COMBINING_ACUTE       # attach to the vowel
            index += 1
            continue
        index += 1                                        # nothing to mark: drop
    return "".join(out)


def _strip_stress_marks(text: str) -> str:
    """Drop stress marks from non-Russian text (meaningless to English voices)."""
    return "".join(ch for ch in (text or "") if ch not in _STRESS_MARKS)


def _spell_dates(text: str) -> str:
    """Convert bare years in speech text to spoken words.

    ``1984`` -> ``nineteen eighty-four``; ``2023`` -> ``twenty twenty-three``.
    Keeps the digits out of the TTS input so the voice says the year, not a
    string of numerals. Numbers that are clearly not years (counts, etc.) are
    left as-is to avoid mangling things like "top 10".
    """
    return _YEAR_RE.sub(lambda m: _int_to_words(int(m.group(0))), text)


# ---------------------------------------------------------------------------
# Russian (Cyrillic) transliteration for TTS.
#
# The bundled English voices (en_US-*) cannot pronounce Cyrillic glyphs, so a
# Russian artist/song name fed to them verbatim comes out as garbled noise.
# We transliterate to a Latin spelling that an ENGLISH voice reads with
# roughly Russian pronunciation, so the DJ says something like "Agata Kristi"
# / "Zemfira" instead of mangling the characters.
#
# This is applied ONLY on the audio path (synthesize) and ONLY when the chosen
# voice is English: a Russian voice (ru_RU-*) reads Cyrillic natively, so the
# text is passed through verbatim for it.
#
# WHY THIS MAP (and not scholarly/ISO-9 transliteration):
#   Scholarly transliteration (ISO 9 / Library of Congress) is built for
#   *reversible* one-letter-to-one-letter mapping for linguists and catalogues.
#   Fed to an English TTS engine it is actively misread -- "Zhukov" comes out
#   with an English J (/dʒ/), "Lermontov" with an English R, and "sch" (from
#   the combo сщ) is voiced as /sk/ ("school"). The romanization explicitly
#   designed to be "intuitive for Anglophones to read and pronounce" is
#   BGN/PCGN, so this map follows BGN/PCGN for the single letters and then adds
#   a handful of multi-letter COMBINATION rules that are the difference
#   between an English voice saying "shch" vs "s-ch" or "sk". No dictionary,
#   no network -- just deterministic rules good enough for intelligible,
#   Russian-flavoured speech.
_CYR_TO_LAT = {
    # single letters (BGN/PCGN style)
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
    "А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Е": "E", "Ё": "Yo",
    "Ж": "Zh", "З": "Z", "И": "I", "Й": "Y", "К": "K", "Л": "L", "М": "M",
    "Н": "N", "О": "O", "П": "P", "Р": "R", "С": "S", "Т": "T", "У": "U",
    "Ф": "F", "Х": "Kh", "Ц": "Ts", "Ч": "Ch", "Ш": "Sh", "Щ": "Shch",
    "Ъ": "", "Ы": "Y", "Ь": "", "Э": "E", "Ю": "Yu", "Я": "Ya",
}

# Multi-letter clusters an English TTS otherwise mangles. Applied BEFORE the
# single-letter pass so e.g. "сч" becomes "sch" (read like "school") rather
# than "s-ch", and "зж" becomes a single "zh" rather than "z-zh".
_COMBOS = [
    ("сч", "sch"), ("зч", "shch"), ("жч", "shch"),
    ("сш", "ssh"), ("зш", "ssh"),
    ("зж", "zh"), ("тс", "ts"),
]

_CYR_VOWELS = set("аеёиоуыэюяъь")
_CYR_RE = re.compile(r"[А-Яа-яЁё]")


def _transliterate_cyrillic(text: str) -> str:
    """Transliterate any Cyrillic in ``text`` to a Latin, TTS-readable spelling.

    Non-Cyrillic characters (Latin words, digits, punctuation, spaces) pass
    through unchanged, so an English intro containing a Russian name keeps its
    English and only the name is converted. The result contains only ASCII,
    which every engine handles cleanly — an English voice cannot read Cyrillic.

    ``е`` is rendered ``ye`` at the start of a word or after a vowel / soft or
    hard sign (its Russian "y+e" onset); elsewhere it is a plain ``e``.
    """
    if not _CYR_RE.search(text):
        return text
    # 1) collapse multi-letter clusters an English voice would mispronounce.
    for cyr, lat in _COMBOS:
        text = text.replace(cyr, lat).replace(cyr.upper(), lat.capitalize())
    # 2) single-letter pass, with the е/Е -> ye rule.
    out = []
    for i, ch in enumerate(text):
        if ch in ("е", "Е"):
            prev = text[i - 1].lower() if i > 0 else ""
            if i == 0 or prev in _CYR_VOWELS:
                out.append("Ye" if ch == "Е" else "ye")
                continue
        out.append(_CYR_TO_LAT.get(ch, ch))
    return "".join(out)

SYSTEM_PROMPT = (
    "You are the voice of a radio station. You introduce songs on air. "
    "Write ONLY the words to be spoken aloud: no stage directions, no track "
    "listings, no markdown, no emoji, no quotation marks around the whole "
    "reply, and never mention that you are an AI. Keep it natural, spoken "
    "English. Every intro stands on its own: talk about the song you are "
    "introducing (you may briefly nod to the one that just played), never about "
    "the station's schedule or how the music is organised. "
    "STRICT FACT RULE: state ONLY facts that appear verbatim in the "
    "information block you are given. Never guess or embellish a song's "
    "history, chart position, recording story, meaning, or origin. If the "
    "information block is thin, say something short and atmospheric about the "
    "title, artist and genre instead of inventing history. A short accurate "
    "intro is always better than a longer invented one."
)

# How the line has to be WRITTEN so the voice reads it correctly: years as words,
# stylised names in their spoken form, punctuation the engine actually pauses on.
# These are style rules, so they ride in the system message with the others.
#
# The examples are called illustrations TWICE on purpose. A small model reads a
# bare example list as content when it sits next to the song facts: the examples
# used to live in the user prompt and nemotron-3-nano:4b opened with "Mindfields
# by Pink, the Prodigy" — an artist that appears in no field of that track
# (measured: 1/3 of lines, from the "P!nk is Pink" example alone), and it also
# recited the rules' own "two songs" phrasing. The Russian rule block used to
# carry "Агата Кристи" as its stress example, which invites the same mistake for
# Russian lines, so its example names are gone too.
SPEECH_RULES = (
    "\nHOW TO WRITE THE LINE (style rules - apply them, do not recite them):\n"
    "- Write every year and every number as words, never as digits: "
    "\"nineteen ninety-nine\", not \"1999\"; \"two songs\", not \"2 songs\".\n"
    "- Write artist and song names the way they are spoken: normalise stylised "
    "spellings and drop symbols that are not pronounced. The three examples "
    "below ILLUSTRATE the rule only; they are NOT this song and must never be "
    "said unless the information block lists them: P!nk is \"Pink\", "
    "Ke$ha is \"Kesha\", deadmau5 is \"dead mouse\". Keep every name "
    "recognisable and never invent a different one.\n"
    "- Say only the artists, titles and albums listed in the information "
    "block. Never add an artist name of your own.\n"
    "- Never announce the genre and never read the style tags out: they are "
    "context for the tone of the line, not something to state.\n"
    "- Keep the sentences short, and punctuate for breath: a comma is a small "
    "pause, an em dash a longer one."
)

# The Russian rule set has to carry the whole style contract, not just the
# pronunciation aids: the English block used to ride along and its first sentence
# was "write it the way it will be SPOKEN, not as a catalogue listing". Splitting
# the two by language silently dropped that, and half the Russian lines came back
# as bare lists ("Fly Away, Lenny Kravitz, rock" / "Наш дом, Mashina Vremeni,
# 1993, Rock") — measured, 3 of 6. The speech-framing rules are therefore
# repeated here in Russian.
RUSSIAN_RULES = (
    "\nОБЯЗАТЕЛЬНО пиши свой ответ ПО-РУССКИ (на русском языке). "
    "Названия песен и имена исполнителей — на русском, сохраняй их "
    "смысл и форму, не переводи и не транслитерируй.\n"
    "Чтобы голосовой движок правильно выговаривал слова, выполни три "
    "обязательных правила:\n"
    "1) Ставь знак ударения ´ СРАЗУ ПОСЛЕ ударной гласной каждого "
    "слова, например: приве´т, музыка´льный, золото´й. "
    "Используй ТОЛЬКО ´ — никакие другие знаки (ʼ, ʹ, `, ‘, ’) не "
    "ставь: движок их либо не видит, либо читает вслух как название "
    "знака.\n"
    "2) Всегда пиши букву Ё (ё) там, где она действительно есть в "
    "слове, никогда не заменяй её на Е (е).\n"
    "3) Годы и числа оставляй цифрами (например: \"1984\") — русский "
    "голос читает их правильно сам. Словами их писать НЕ нужно.\n"
    "Убирай из названий символы и стилизации, которые не читаются вслух, "
    "но не заменяй имя исполнителя другим. Говори только о песне, которую "
    "представляешь: называй лишь те имена и названия, что перечислены ниже, "
    "и не добавляй других исполнителей.\n"
    "Пиши так, как это звучит вслух, а не как список из каталога: живая фраза "
    "ведущего, а не перечисление «название, исполнитель, жанр».\n"
    "Пиши короткими фразами и расставляй знаки для дыхания: запятая — "
    "короткая пауза, тире — длинная.\n"
    "Не называй жанр и не перечисляй стилевые теги: они нужны только для "
    "настроения фразы."
)


def effective_voice(voice: str | None = None) -> str:
    """The voice that will actually speak: the request, else a usable default.

    A voice only counts when the TTS service reports it *installed* — an id the
    engine merely lists cannot speak, and returning it would make `tts_problem()`
    report a problem for a station that has a perfectly good voice. Otherwise the
    service's resolution wins (see ``tts_client.default_voice``): a config
    carried over from a previous engine therefore lands on the station's own pick
    instead of leaving the DJ silent.
    """
    wanted = (voice or config.get("dj.voice") or "").strip()
    if wanted and wanted in tts_client.installed_voice_ids():
        return wanted
    fallback = tts_client.default_voice()
    if wanted and wanted != fallback:
        log.debug("voice %r is not installed on the TTS service; using %r",
                  wanted, fallback)
    return fallback


def language_for(voice: str | None = None,
                 language: str | None = None) -> str:
    """The DJ's language as a code (``en``/``ru``).

    The language is no longer a property of the voice id: an engine may speak ten
    styles across many languages, so the choice is explicit (``dj.language``) and
    validated against the selected voice's declared languages — an English-only
    voice cannot be asked to speak Russian, so the answer falls back to what the
    voice actually supports.

    When the TTS service cannot be reached there are no declared languages to
    validate against; a legacy Piper-shaped voice id (``ru_RU-*``) then supplies
    the last hint, so text shaping stays sane during an outage.
    """
    voice_id = effective_voice(voice)
    entry = tts_client.voice(voice_id)
    hint = language if language is not None else str(config.get("dj.language") or "")
    if entry is not None:
        return tts_client.resolve_language(entry, hint)
    if hint:
        return hint.strip().lower()
    legacy = (voice_id or voice or "").lower()
    return "ru" if legacy.startswith(("ru_", "ru-")) else "en"


def voice_catalogue() -> list[dict[str, Any]]:
    """The voice catalogue, as the TTS service declares it.

    Nothing about voices is hardcoded in the app any more: whatever the running
    engine offers (ids, names, genders, languages, install state) is what the DJ
    card shows.
    """
    return tts_client.voices()


def available_voices() -> list[str]:
    """Ids of voices that can speak right now."""
    return tts_client.installed_voice_ids()


def params_schema() -> list[dict[str, Any]]:
    """Synthesis parameters the engine declares (rendered by the DJ card)."""
    return tts_client.params()


def _clean_script(text: str, max_sentences: int,
                  language: str | None = None) -> str:
    text = _THINK_RE.sub("", text or "")
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"[*_#`>\[\]]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().strip('"').strip()
    if not text:
        return ""
    sentences = [s for s in _SENT_SPLIT.split(text) if s.strip()]
    text = " ".join(sentences[: max(1, max_sentences)]).strip()
    if (language or "en") == "ru":
        # Whatever accent character the model wrote becomes the combining acute
        # an espeak-based engine obeys, attached to the right vowel (see
        # _normalize_stress_marks); an engine without stress support gets it
        # stripped later, on the audio path.
        text = _normalize_stress_marks(text)
        # Russian numerals must be words: the neural voice invents numbers when
        # it has to read digits ("в 1989 году" came out as 979/1089/1972 in
        # different takes), and the spelled year was correct every single time.
        text = ru_numbers.spell_numbers_ru(text)
    else:
        text = _strip_stress_marks(text)
        text = _spell_dates(text)
    return text


def prepare_script(text: str, voice: str | None = None,
                   max_sentences: int = 99,
                   language: str | None = None) -> str:
    """The display-level prep every spoken line goes through.

    Called for LLM output (via ``generate_script``) and for text handed straight
    to ``/api/dj/preview``, so the text a user sees is the text the voice is
    given: years and numbers written as words, and Russian stress marks
    rewritten into the character espeak-ng obeys. Deliberately does NOT
    transliterate — Cyrillic stays Cyrillic on screen.

    ``language`` is the language already resolved by the caller. When it is not
    given, it comes from ``dj.language`` validated against the selected voice —
    the on-air path never repeats it down the call chain.
    """
    return _clean_script(text, max_sentences,
                         language=language or language_for(voice))


def fallback_script(track: dict[str, Any],
                    language: str | None = None) -> str:
    """Template intro, used when the LLM is off or unreachable.

    The themed *program* a track belongs to is deliberately absent here — and
    from the prompt (see :func:`generate_script`). The DJ introduces songs, not
    the station's scheduling: no "we are switching to a new program", no
    "next up, a run of Rock". The program still drives the playlist; it is just
    never spoken.
    """
    artist = (track.get("artist") or "").strip()
    title = (track.get("title") or "").strip() or "this next one"
    year = (track.get("year") or "").strip()
    base = ""
    if (language or "en") == "ru":
        if artist and year:
            base = f"Далее в эфире: {title}, в исполнении {artist}, {year} года."
        elif artist:
            base = f"А сейчас — {title}, {artist}."
        else:
            base = f"Далее — {title}."
    else:
        if artist and year:
            base = f"Coming up next: {title}, by {artist}, from {year}."
        elif artist:
            base = f"Here's {title}, by {artist}."
        else:
            base = f"Up next, {title}."
    # Same final shaping as an LLM line, so the template reads (and sounds like)
    # the same thing: years as words, stress marks in the form the engine wants.
    return _clean_script(base, 99, language=language or "en")


# Metadata that is a placeholder rather than a value: a ripper (or a failed
# tagger) wrote it when it had nothing to write. Saying it on air is worse than
# saying nothing — the DJ announced "Genre: Default" and "...their Best Hits
# 200% album" for tracks whose tags are simply empty. Measured in the live
# library: 150 tracks tagged genre "None", 95 "Default", 47 "Genre", 22
# "Unknown"; 11 tracks on an album called "Je Veux (2010) mp3", plus
# "Ripped by Buko's MP3 Lair". Both lists are meant to be extended.
_JUNK_META = {
    "none", "null", "nil", "default", "unknown", "untitled", "n/a", "na",
    "genre", "album", "track", "audio", "music", "mp3", "flac", "cd", "cd1",
    "cd2", "disc 1", "disc 2", "disc1", "disc2", "various", "various artists",
    "va", "no album", "no genre", "not set", "-",
    # "Other"/"Misc" are buckets, not genres: the Russian voice said «этот трек
    # из жанра Other», which is noise on air. 371 + 74 tracks in the live library.
    "other", "misc",
}
_JUNK_META_RE = re.compile(
    # A format/rip marker as a word of its own ...
    r"(?:^|[\s(\[])(?:mp3|flac|wav|cbr|vbr|320kbps|ripped|rip|download)"
    r"(?:$|[\s)\].])"
    # ... or a URL / domain, which is where those markers usually come from.
    r"|www\.|https?://|\b[a-z0-9-]+\.(?:com|net|org|ru|info|biz|me)\b",
    re.I)


def _is_junk_meta(value: str) -> bool:
    """True when a tag holds a placeholder instead of a real value."""
    text = (value or "").strip()
    return text.lower() in _JUNK_META or bool(_JUNK_META_RE.search(text))


def _facts_block(track: dict[str, Any], facts: dict[str, Any]) -> str:
    lines = []
    unknown = []
    for label, key in (("Title", "title"), ("Artist", "artist"),
                       ("Album", "album"), ("Genre", "genre"), ("Year", "year")):
        value = (track.get(key) or "").strip() if track.get(key) else ""
        if value and label != "Title" and _is_junk_meta(value):
            # Treated as missing, so it also lands in the UNKNOWN list below and
            # the model knows not to guess it.
            value = ""
        if value:
            lines.append(f"{label}: {value}")
        elif label != "Title":
            unknown.append(label.lower())
    # NOTE: neither MusicBrainz "first-release-date" nor "release_title" are
    # surfaced to the model. Both describe one arbitrary *recording/release*
    # row (often a later compilation or live cut), not the song's origin, and
    # small models confidently restate them as "originally released in ...".
    # The ID3 year above is far more trustworthy for a personal collection.
    # Three tags, not all eight MusicBrainz returns: the model reads a long tag
    # list as something to recite ("...mixing rock electronic pop alternative
    # rock industrial drum and bass dubstep hardcore").
    tags = [str(t).strip() for t in (facts.get("tags") or [])]
    tags = [t for t in tags if t and not _is_junk_meta(t)][:3]
    if tags:
        lines.append("Style tags: " + ", ".join(tags))
    if facts.get("credited_artists"):
        lines.append("Credited artists: " + ", ".join(facts["credited_artists"]))
    if facts.get("artist_summary"):
        lines.append("About the artist: " + facts["artist_summary"][:800])
    if unknown:
        lines.append(
            "UNKNOWN (do not mention, do not guess): " + ", ".join(unknown)
        )
    return "\n".join(lines)


# Names the prompt offers ONLY as illustrations of the spelling rule. A small
# model copies them into the line it writes: for "Mindfields" by Prodigy — which
# carries none of these names — nemotron-3-nano:4b opened with "Mindfields by
# Pink, the Prodigy", then "Mindfields by Pink and Keesha", and once recited
# "two songs" out of the number rule. Moving the examples into the system message
# helped nothing; only a guard on the output catches it. Russian spellings are
# listed too: a Russian line said the same names in Cyrillic.
_EXAMPLE_NAME_RES = (
    re.compile(r"\bP(?:ink|!nk)\b"),                      # the "P!nk is Pink" example
    # "Kesha" / "Keesha": the model misspells it, so the extra e is allowed.
    re.compile(r"\bK(?:e\$ha|e+sha)\b", re.I),
    re.compile(r"\bdeadmau5\b|\bdead mouse\b", re.I),
    re.compile(r"\bPanic!? at the Disco\b", re.I),
    re.compile(r"\bBlink-182\b", re.I),
    re.compile(r"\bone eighty-two\b", re.I),
    re.compile(r"\bПинк\b|\bКеша\b|\bДэдмаус\b"),
)

# The other thing a 4B model does to a DJ line: loop. One measured line was
# "да-да-да-да-…" eighty times over; others repeated a word or a phrase.
_LOOP_RES = (
    re.compile(r"(.{2,5})\1{5,}"),                          # да-да-да-да-да-да
    re.compile(r"\b(\w{2,})(?:\s+\1\b){4,}", re.I),         # yes yes yes yes yes
    re.compile(r"\b(\w+\s+\w+\s+\w+)(?:\s+\1){2,}", re.I),  # phrase x3
)


def _usable_script(script: str, track: dict[str, Any]) -> bool:
    """False when a generated line is unusable on air.

    Two failure modes the prompt cannot prevent, both measured: naming an artist
    that exists only in the prompt's examples, and degenerate repetition. A
    rejected line is asked for again once, then replaced by the template.
    """
    fields = " ".join(
        str(track.get(key) or "") for key in ("title", "artist", "album")
    ).lower()
    for pattern in _EXAMPLE_NAME_RES:
        # Capitalised on purpose: "a pink neon glow" is imagery, "Pink" is the
        # artist from the example list. A name the track really carries (Pink
        # Floyd introducing Pink Floyd) is not a leak.
        match = pattern.search(script)
        if match and match.group(0).lower() not in fields:
            log.info("rejecting a DJ line that named a prompt example (%s)",
                     match.group(0))
            return False
    for pattern in _LOOP_RES:
        if pattern.search(script):
            log.info("rejecting a looping DJ line (%s)", pattern.pattern)
            return False
    return True


def _ask_llm(system_prompt: str, user_prompt: str) -> str:
    """One chat completion from the configured LLM, retrying transport failures.

    A transient Ollama blip (restart, first load of a model) must not drop the
    line to the templated fallback, so the call is retried with a short backoff.
    Returns "" when every attempt failed.
    """
    base_url = str(config.get("llm.base_url", "")).rstrip("/")
    model = config.get("llm.model", "qwen3.5:9b")
    timeout = float(config.get("llm.timeout_s", 120))
    retries = int(config.get("llm.retries", 2))
    for attempt in range(retries + 1):
        try:
            resp = httpx.post(
                f"{base_url}/api/chat",
                timeout=timeout,
                json={
                    "model": model,
                    "stream": False,
                    "think": False,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "options": {
                        "temperature": float(config.get("llm.temperature", 0.7)),
                        "num_predict": 220,
                    },
                },
            )
            resp.raise_for_status()
            return (resp.json().get("message") or {}).get("content", "")
        except Exception as exc:  # connection refused, timeout, 5xx, empty body
            if attempt < retries:
                log.warning("LLM attempt %d failed (%s); retrying", attempt + 1, exc)
                time.sleep(1.0 * (attempt + 1))
            else:
                log.warning("LLM call failed (%s)", exc)
    return ""


def generate_script(track: dict[str, Any], previous: dict[str, Any] | None = None,
                    voice: str | None = None,
                    language: str | None = None) -> str:
    """Ask the local LLM for an on-air intro; fall back to a template.

    The DJ's language follows the configured language + the chosen voice: pass
    ``language`` (or leave None for ``dj.language`` resolved against the voice)
    and the script is generated in that language.

    The *program* a track belongs to is never part of the prompt. It used to be:
    a track that opened a themed block told the model to welcome listeners into
    that vibe, and the model duly said things like "we are switching to a new
    program". The program is a playlist device — it decides what plays — not
    something the DJ talks about.
    """
    max_sentences = config.randint_range(
        "dj.sent_min", "dj.sent_max",
        config.DEFAULTS["dj"]["sent_min"], config.DEFAULTS["dj"]["sent_max"])
    facts: dict[str, Any] = {}
    try:
        facts = enrich.enrich_track(track)
    except Exception as exc:
        log.debug("enrichment failed: %s", exc)

    language = language or language_for(voice)
    is_russian = language == "ru"

    if not config.get("llm.enabled", True):
        return fallback_script(track, language=language)

    style = config.get("dj.style", "warm, witty late-night radio host")
    prev_line = ""
    if previous and (previous.get("title") or previous.get("artist")):
        prev_line = (
            f"\nThe song that just finished was "
            f"{previous.get('title') or 'unknown'} by "
            f"{previous.get('artist') or 'unknown'}. You may briefly reference it."
        )

    # The model is a small one (see llm.model) and reads the rule examples as
    # song data when they sit next to the facts, so the rules ride in the SYSTEM
    # message and the user message carries only this song. One rule set, never
    # both: the English rules want years as words, the Russian ones want them as
    # digits (that is what the Russian engine needs), and handing over both left
    # the model to pick.
    rules = RUSSIAN_RULES if is_russian else SPEECH_RULES
    system_prompt = SYSTEM_PROMPT + rules

    user_prompt = (
        f"Persona: {style}.\n"
        f"Write at most {max_sentences} sentences introducing the next song.\n"
        f"Mention one genuinely interesting fact ONLY if the information block "
        f"below directly supports it. Do NOT state any release date or year "
        f"that is not listed below."
        f"{prev_line}\n\n"
        f"Information about the next song:\n"
        f"{_facts_block(track, facts)}\n"
        f"Say only the names listed above.\n"
    )

    def ask() -> str:
        content = _ask_llm(system_prompt, user_prompt)
        if not content:
            return ""
        return prepare_script(content, voice=voice,
                              max_sentences=max_sentences,
                              language=language)

    script = ask()
    if script and not _usable_script(script, track):
        # Ask again before giving up on the model: the guards fire on maybe one
        # line in six, and the second pass is usually clean. A second bad line
        # falls back to the template, which is plain but always correct.
        script = ask()
    if script and _usable_script(script, track):
        return script
    if not script:
        log.warning("no usable DJ line from the LLM; using the template")
    return fallback_script(track, language=language)


def _cache_path(text: str, voice: str, language: str,
                params: dict[str, Any], engine: str) -> Path:
    """Where a render is cached.

    The key covers everything that changes the audio — engine, voice, language,
    every parameter value and the exact text (already number-spelled) — because
    the engines are stochastic: two runs of the same request differ, so a hit is
    only valid for an identical request.
    """
    digest = hashlib.sha256(
        f"{engine}|{voice}|{language}|"
        f"{json.dumps(params, sort_keys=True, default=str)}|{text}"
        .encode("utf-8")
    ).hexdigest()[:32]
    return config.DJ_CACHE_DIR / f"{digest}.mp3"


def _shape_for_engine(text: str, language: str,
                      stress_marks: bool) -> str:
    """Text shaping that depends on what the engine understands.

    * Russian: stress marks are the espeak convention — an engine without that
      notion would speak the mark as a glyph, so it is stripped for those; and
      numbers are written out (see app.ru_numbers) because a neural voice
      invents them when it reads digits.
    * English: years become words ("nineteen eighty-four") and Cyrillic names
      are transliterated, because an English voice cannot read Cyrillic.
    """
    if language == "ru":
        text = _normalize_stress_marks(text) if stress_marks \
            else _strip_stress_marks(text)
        return ru_numbers.spell_numbers_ru(text)
    text = _strip_stress_marks(text)
    text = _spell_dates(text)
    return _transliterate_cyrillic(text)


def synthesize(text: str, voice: str | None = None,
               language: str | None = None,
               params: dict[str, Any] | None = None) -> Path | None:
    """Render ``text`` to the mp3 the stream plays. Returns None on failure.

    Speech comes from the standalone TTS service (``app.tts_client``); this
    function owns the two things that are the station's own: what the text must
    look like before it is spoken, and how the audio is mastered (loudnorm +
    gain + mp3 at the stream's format) so a DJ break sits level with the music.
    """
    text = (text or "").strip()
    if not text:
        return None
    voice = effective_voice(voice)
    caps = tts_client.capabilities() or {}
    entry = tts_client.voice(voice)
    if entry is None:
        log.warning("the TTS service offers no voice %r", voice)
        return None
    if not entry.get("installed"):
        log.warning("voice %r is not installed on the TTS service", voice)
        return None
    lang = language_for(voice, language)
    stress_marks = bool((caps.get("text") or {}).get("stress_marks"))
    engine_id = str((caps.get("engine") or {}).get("id") or "")

    text = _shape_for_engine(text, lang, stress_marks)
    resolved = tts_client.resolve_params(params)
    out_path = _cache_path(text, voice, lang, resolved, engine_id)
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    try:
        wav_bytes, meta = tts_client.speech(text, voice, lang, resolved)
    except tts_client.TTSError as exc:
        log.error("TTS service could not render the line: %s", exc)
        return None
    if not wav_bytes:
        log.error("TTS service returned empty audio")
        return None
    if (meta.get("params_ignored") or "[]") not in ("[]", ""):
        log.debug("engine ignored parameters: %s", meta.get("params_ignored"))

    config.DJ_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "dj.wav"
        wav_path.write_bytes(wav_bytes)
        gain_db = float(config.get("dj.gain_db", 0.0))
        filters = ["loudnorm=I=-16:TP=-1.5:LRA=11"]
        if abs(gain_db) > 0.01:
            filters.append(f"volume={gain_db}dB")
        tmp_mp3 = Path(tmpdir) / "dj.mp3"
        ff = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(wav_path), "-af", ",".join(filters),
             "-ar", str(config.get("stream.sample_rate", 44100)),
             "-ac", str(config.get("stream.channels", 2)),
             "-b:a", f"{int(config.get('stream.bitrate_kbps', 128))}k",
             "-write_xing", "0", "-id3v2_version", "0",
             str(tmp_mp3)],
            capture_output=True, timeout=180, check=False,
        )
        if ff.returncode != 0 or not tmp_mp3.exists():
            log.error("ffmpeg dj encode failed: %s",
                      ff.stderr.decode("utf-8", "replace")[:400])
            return None
        shutil.move(str(tmp_mp3), str(out_path))
    log.info("DJ line rendered: %s (%s, %.2fs audio, %s ms)",
             voice, lang, meta.get("duration") or 0.0, meta.get("render_ms"))
    return out_path


def audio_duration(path: Path) -> float:
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, timeout=30, check=False,
        )
        return float(proc.stdout.decode().strip() or 0.0)
    except (ValueError, OSError, subprocess.TimeoutExpired):
        return 0.0


def prepare_break(track: dict[str, Any],
                  previous: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Generate script + audio for one DJ break. Returns None if unavailable."""
    # Check the voice BEFORE the LLM call: without it the TTS service cannot
    # speak, so generating a script only burns the (GPU-backed) LLM and then
    # drops the break silently. One clear warning, then stay quiet until fixed.
    global _tts_warned
    problem = tts_problem()
    if problem:
        if not _tts_warned:
            _tts_warned = True
            log.warning(
                "DJ breaks are OFF: %s. The station keeps playing music. Fix it "
                "in the web UI's DJ card (pick a voice marked installed, or "
                "install one), or run `python -m app.voices --list` to see what "
                "the TTS service offers (configured voice: %s).",
                problem, config.get("dj.voice", ""))
        return None
    _tts_warned = False
    script = generate_script(track, previous)
    if not script:
        return None
    audio = synthesize(script)
    if audio is None:
        return None
    duration = audio_duration(audio)
    try:
        conn = db.connect()
        conn.execute(
            "INSERT INTO dj_scripts(track_id, text, audio_path, duration) "
            "VALUES(?,?,?,?)",
            (track.get("id"), script, str(audio), duration),
        )
        conn.commit()
    except Exception as exc:
        log.debug("could not persist dj script: %s", exc)
    return {"text": script, "audio_path": str(audio), "duration": duration}


def llm_health() -> dict[str, Any]:
    base_url = str(config.get("llm.base_url", "")).rstrip("/")
    if not base_url:
        return {"ok": False, "error": "no base_url configured"}
    try:
        resp = httpx.get(f"{base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m.get("name") for m in resp.json().get("models", [])]
        wanted = config.get("llm.model")
        return {"ok": True, "models": models, "model_present": wanted in models}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def list_ollama_models(base_url: str | None = None) -> dict[str, Any]:
    """Query an Ollama instance for its available models.

    Uses ``base_url`` if given (so the UI can probe a URL the user just typed
    before saving), otherwise the configured ``llm.base_url``. Returns
    ``{"ok": bool, "models": [name, ...], "error": str|None}``. Never raises.
    """
    url = str(base_url or config.get("llm.base_url", "")).rstrip("/")
    if not url:
        return {"ok": False, "models": [], "error": "no base_url configured"}
    try:
        resp = httpx.get(f"{url}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m.get("name") for m in resp.json().get("models", [])]
        return {"ok": True, "models": models, "error": None}
    except Exception as exc:
        return {"ok": False, "models": [], "error": str(exc)}


def test_llm(base_url: str | None = None, model: str | None = None,
             prompt: str | None = None) -> dict[str, Any]:
    """Make a real chat completion to verify the LLM works as the DJ.

    Uses the supplied ``base_url``/``model`` (so the UI can test before saving)
    or the configured ones. Performs one real ``/api/chat`` call — the same
    shape ``generate_script`` uses — and returns the generated text. Never
    raises; returns ``{"ok", "text", "model", "error"}``.
    """
    url = str(base_url or config.get("llm.base_url", "")).rstrip("/")
    model = model or config.get("llm.model", "qwen3.5:9b")
    timeout = float(config.get("llm.timeout_s", 120))
    if not url:
        return {"ok": False, "text": "", "model": model, "error": "no base_url configured"}
    try:
        resp = httpx.post(
            f"{url}/api/chat",
            timeout=timeout,
            json={
                "model": model,
                "stream": False,
                "think": False,
                "messages": [
                    {"role": "system",
                     "content": "You are the on-air DJ for a late-night radio station."},
                    {"role": "user",
                     "content": prompt or "Say hello to the listeners in one short, warm sentence."},
                ],
                "options": {"temperature": 0.7, "num_predict": 120},
            },
        )
        resp.raise_for_status()
        text = (resp.json().get("message") or {}).get("content", "")
        return {"ok": True, "text": text, "model": model, "error": None}
    except Exception as exc:
        return {"ok": False, "text": "", "model": model, "error": str(exc)}


# One warning per outage, not one per break: the queue would otherwise repeat
# the same line every few minutes forever.
_tts_warned = False


def tts_problem(voice: str | None = None) -> str | None:
    """Why the DJ cannot speak right now, or None when everything is in place.

    Single source of truth for "is the voice usable", so the boot log, the
    health payload, the web UI and ``prepare_break`` all describe the same
    state. The answer now comes from the TTS service (unreachable, model not
    loaded, no voice installed, configured voice missing) rather than from a
    model file on local disk — the app owns no speech engine any more.
    """
    problem = tts_client.problem()
    if problem:
        return problem
    # Judge the voice that will actually speak (see tts_client.problem): a dead
    # id from a previous engine must not silence a station that has a usable one.
    available = tts_client.installed_voice_ids()
    wanted = voice or config.get("dj.voice")
    effective = effective_voice(wanted)
    if effective and effective in available:
        return None
    installed = ", ".join(available) or "none"
    return (f"the configured voice {wanted!r} is not installed "
            f"(have: {installed})")


def tts_health() -> dict[str, Any]:
    """What the DJ card and ``/api/health`` show about speech."""
    problem = tts_problem()
    service = tts_client.health()
    caps = tts_client.capabilities() or {}
    return {
        "ok": problem is None,
        "problem": problem,
        "service": service,
        "engine": caps.get("engine"),
        "binary": None,  # kept for older clients/GUI: there is no local binary
        # The voice/language that will actually speak, plus the raw setting so
        # the UI can explain a substitution instead of showing a dead id as if
        # it were live.
        "voice": effective_voice(),
        "configured_voice": config.get("dj.voice"),
        "language": language_for(),
        "params": tts_client.resolve_params(),
        "voices_available": tts_client.installed_voice_ids(),
        "base_url": tts_client.base_url(),
    }


__all__ = [
    "generate_script", "synthesize", "prepare_break", "fallback_script",
    "available_voices", "voice_catalogue", "language_for", "effective_voice",
    "params_schema", "llm_health", "tts_health", "tts_problem",
    "audio_duration", "_spell_dates",
]

# There is deliberately no voice catalogue in this file any more. Voices,
# languages and synthesis parameters are whatever the TTS service declares in
# GET /v1/capabilities (see app/tts_client.py), so a new engine needs no change
# here — the DJ card renders the service's own answer.
