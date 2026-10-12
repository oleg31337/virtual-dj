"""The DJ brain: turns track metadata into a spoken intro.

Pipeline per track:
    metadata (+ enrichment facts) -> LLM script -> Piper TTS -> mp3 segment

Both stages degrade gracefully: with no LLM we fall back to a templated line,
and if TTS fails the stream simply skips the DJ break rather than stalling.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx

from . import config, db, enrich

log = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Piper (and most TTS) read bare digits as individual numerals — "1984" comes
# out as "one nine eight four" instead of a year. Years and numbers the DJ
# should *speak* are converted to words before synthesis so the voice model
# reads them as dates/quantities, not digits.
_YEAR_RE = re.compile(r"\b(19|20)(\d{2})\b")
_NUM_RE = re.compile(r"\b\d+\b")

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
    if n < 2000:
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
    which Piper handles cleanly.

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
    "English. "
    "STRICT FACT RULE: state ONLY facts that appear verbatim in the "
    "information block you are given. Never guess or embellish a song's "
    "history, chart position, recording story, meaning, or origin. If the "
    "information block is thin, say something short and atmospheric about the "
    "title, artist and genre instead of inventing history. A short accurate "
    "intro is always better than a longer invented one."
)


def piper_binary() -> str | None:
    """Locate the piper executable (venv bin first, then PATH)."""
    candidate = Path(sys.executable).parent / "piper"
    if candidate.exists():
        return str(candidate)
    return shutil.which("piper")


def voice_model_path(voice: str | None = None) -> Path | None:
    name = voice or config.get("dj.voice", "en_GB-alan-medium")
    # Allow tests/installs to point the voice library elsewhere (e.g. system
    # share dir) without duplicating multi-hundred-MB models.
    voices_dir = config.voices_dir()
    path = voices_dir / f"{name}.onnx"
    return path if path.exists() else None


def available_voices() -> list[str]:
    voices_dir = config.voices_dir()
    if not voices_dir.is_dir():
        return []
    return sorted(p.stem for p in voices_dir.glob("*.onnx"))


def voice_language(voice_id: str | None) -> str:
    """Language bucket of a voice id: ``"russian"`` or ``"english"``.

    Per-song language detection was removed; the DJ's language is now a
    property of the chosen voice. Curated ids resolve via ``VOICE_PROFILES``;
    any other installed ``ru_RU-*``/``ru-*`` voice counts as Russian, all else
    English.
    """
    vid = (voice_id or "").strip()
    if not vid:
        return "english"
    for p in VOICE_PROFILES:
        if p["id"] == vid:
            return p.get("lang") or "english"
    return "russian" if vid.lower().startswith(("ru_", "ru-")) else "english"


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
    # Years must be spelled out so an English TTS reads them as dates, not a
    # string of numerals. Russian-track text is left as-is (digits) because the
    # Russian voice reads year numerals natively.
    if language == "russian":
        # Whatever accent character the model wrote becomes the combining acute
        # espeak obeys, attached to the right vowel (see _normalize_stress_marks).
        text = _normalize_stress_marks(text)
    else:
        text = _strip_stress_marks(text)
        text = _spell_dates(text)
    return text


def prepare_script(text: str, voice: str | None = None,
                   max_sentences: int = 99) -> str:
    """The display-level prep every spoken line goes through.

    Called for LLM output (via ``generate_script``) and for text handed straight
    to ``/api/dj/preview``, so the text a user sees is the text the voice is
    given: years spelled out, and Russian stress marks rewritten into the
    character espeak-ng actually obeys. Deliberately does NOT transliterate —
    Cyrillic stays Cyrillic on screen.
    """
    return _clean_script(text, max_sentences,
                         language=voice_language(voice))


def fallback_script(track: dict[str, Any],
                    program: dict[str, Any] | None = None,
                    language: str | None = None) -> str:
    artist = (track.get("artist") or "").strip()
    title = (track.get("title") or "").strip() or "this next one"
    year = (track.get("year") or "").strip()
    base = ""
    if language == "russian":
        if artist and year:
            base = f"Далее в эфире: {title}, в исполнении {artist}, {year} года."
        elif artist:
            base = f"А сейчас — {title}, {artist}."
        else:
            base = f"Далее — {title}."
        if program:
            label = program.get("label") or "следующий блок"
            kind = program.get("kind")
            if kind == "genre":
                base = f"Звучит подборка в стиле {label}. {base}"
            elif kind == "artist":
                base = f"Шоу, посвящённое {label}. {base}"
            elif kind == "decade":
                base = f"Отправляемся в {label}. {base}"
        # Leave years/dates as digits (the Russian voice reads them natively).
        return base
    # English (default) fallback path.
    if artist and year:
        base = f"Coming up next: {title}, by {artist}, from {year}."
    elif artist:
        base = f"Here's {title}, by {artist}."
    else:
        base = f"Up next, {title}."
    if program:
        label = program.get("label") or "the next set"
        kind = program.get("kind")
        if kind == "genre":
            return _spell_dates(f"Next up, a run of {label}. {base}")
        if kind == "artist":
            return _spell_dates(f"A spotlight on {label} now. {base}")
        if kind == "decade":
            return _spell_dates(f"Traveling back to the {label}. {base}")
    return _spell_dates(base)


def _facts_block(track: dict[str, Any], facts: dict[str, Any]) -> str:
    lines = []
    unknown = []
    for label, key in (("Title", "title"), ("Artist", "artist"),
                       ("Album", "album"), ("Genre", "genre"), ("Year", "year")):
        value = (track.get(key) or "").strip() if track.get(key) else ""
        if value:
            lines.append(f"{label}: {value}")
        elif label != "Title":
            unknown.append(label.lower())
    # NOTE: neither MusicBrainz "first-release-date" nor "release_title" are
    # surfaced to the model. Both describe one arbitrary *recording/release*
    # row (often a later compilation or live cut), not the song's origin, and
    # small models confidently restate them as "originally released in ...".
    # The ID3 year above is far more trustworthy for a personal collection.
    if facts.get("tags"):
        lines.append("Style tags: " + ", ".join(facts["tags"]))
    if facts.get("credited_artists"):
        lines.append("Credited artists: " + ", ".join(facts["credited_artists"]))
    if facts.get("artist_summary"):
        lines.append("About the artist: " + facts["artist_summary"][:800])
    if unknown:
        lines.append(
            "UNKNOWN (do not mention, do not guess): " + ", ".join(unknown)
        )
    return "\n".join(lines)


def generate_script(track: dict[str, Any], previous: dict[str, Any] | None = None,
                    program: dict[str, Any] | None = None,
                    voice: str | None = None) -> str:
    """Ask the local LLM for an on-air intro; fall back to a template.

    The DJ's language follows the voice that will speak the line: pass
    ``voice`` (or leave None for the configured ``dj.voice``) and the script is
    generated in that voice's language (Russian voices -> Russian script).
    """
    max_sentences = config.randint_range(
        "dj.sent_min", "dj.sent_max",
        config.DEFAULTS["dj"]["sent_min"], config.DEFAULTS["dj"]["sent_max"])
    facts: dict[str, Any] = {}
    try:
        facts = enrich.enrich_track(track)
    except Exception as exc:
        log.debug("enrichment failed: %s", exc)

    language = voice_language(voice or config.get("dj.voice"))
    is_russian = language == "russian"

    if not config.get("llm.enabled", True):
        return fallback_script(track, program=program, language=language)

    style = config.get("dj.style", "warm, witty late-night radio host")
    prev_line = ""
    if previous and (previous.get("title") or previous.get("artist")):
        prev_line = (
            f"\nThe song that just finished was "
            f"{previous.get('title') or 'unknown'} by "
            f"{previous.get('artist') or 'unknown'}. You may briefly reference it."
        )

    # When this track opens a new themed program, tell the DJ to announce the
    # vibe switch — this is how a real DJ frames the transition.
    program_line = ""
    if program:
        label = program.get("label") or "the next set"
        kind = program.get("kind")
        if kind == "genre":
            program_line = (
                f"\nThis song opens a new program of {label} music. Welcome the "
                f"listeners into that vibe before introducing the track."
            )
        elif kind == "artist":
            program_line = (
                f"\nThis song opens a spotlight on {label}. Frame the transition "
                f"into this artist's music before the intro."
            )
        elif kind == "decade":
            program_line = (
                f"\nThis song opens a trip back to the {label}. Set the era before "
                f"the intro."
            )

    # How the text must be written so the voice reads it correctly. This is the
    # difference between a catalogue entry and something a host can say: years
    # and numbers as words, stylised names in their spoken form, and punctuation
    # the engine actually pauses on. (Measured: the voice moves its pitch over
    # ~19 semitones when the line is written for speech; it also cannot read
    # "P!nk" or "1999" without mangling them.)
    speech_line = (
        "\nWrite it the way it will be SPOKEN, not as a catalogue listing:\n"
        "- Write every year and every number as words, never as digits: "
        "\"nineteen ninety-nine\", not \"1999\"; \"two songs\", not \"2 songs\"; "
        "\"Blink one eighty-two\" for the band Blink-182.\n"
        "- Write artist and song names in their spoken form. Normalise stylised "
        "spellings and drop symbols that are not pronounced: P!nk is \"Pink\", "
        "Ke$ha is \"Kesha\", Panic! at the Disco is \"Panic at the Disco\", "
        "deadmau5 is \"dead mouse\". The names in the information block may be "
        "stylised - say them plainly, but keep them recognisable and never "
        "invent a different name.\n"
        "- Keep the sentences short, and punctuate for breath: a comma is a "
        "small pause, an em dash a longer one."
    )

    lang_line = ""
    if is_russian:
        # The Russian voice speaks Russian natively; the intro must be written
        # in Russian so the DJ actually talks in Russian. Two pronunciation
        # aids for the TTS engine, which otherwise guesses stress and silently
        # swallows Ё:
        #   * place an acute stress mark (´) on the stressed vowel of EVERY
        #     word (e.g. приве́т, музыка́льный, золото́й) so it is pronounced right;
        #   * always write Ё (ё) wherever the word actually contains it — never
        #     substitute Е (е).
        lang_line = (
            "\nОБЯЗАТЕЛЬНО пиши свой ответ ПО-РУССКИ (на русском языке). "
            "Названия песен и имена исполнителей — на русском, сохраняй их "
            "смысл и форму, не переводи и не транслитерируй.\n"
            "Чтобы голосовой движок правильно выговаривал слова, выполни три "
            "обязательных правила:\n"
            "1) Ставь знак ударения ´ СРАЗУ ПОСЛЕ ударной гласной каждого "
            "слова, например: приве´т, музыка´льный, золото´й, Ага´та Кри´сти. "
            "Используй ТОЛЬКО ´ — никакие другие знаки (ʼ, ʹ, `, ‘, ’) не "
            "ставь: движок их либо не видит, либо читает вслух как название "
            "знака.\n"
            "2) Всегда пиши букву Ё (ё) там, где она действительно есть в "
            "слове, никогда не заменяй её на Е (е).\n"
            "3) Годы и числа оставляй цифрами (например: \"1984\") — русский "
            "голос читает их правильно сам. Словами их писать НЕ нужно."
        )

    user_prompt = (
        f"Persona: {style}.\n"
        f"Write at most {max_sentences} sentences introducing the next song.\n"
        f"Mention one genuinely interesting fact ONLY if the information block "
        f"below directly supports it. Do NOT state any release date or year "
        f"that is not listed below."
        f"{speech_line}{prev_line}{program_line}{lang_line}\n\n"
        f"Information about the next song:\n"
        f"{_facts_block(track, facts)}\n"
    )

    base_url = str(config.get("llm.base_url", "")).rstrip("/")
    model = config.get("llm.model", "qwen3.5:9b")
    timeout = float(config.get("llm.timeout_s", 120))

    # A transient Ollama blip (restart, first-load of a model) should not
    # immediately drop the DJ line to the templated fallback. Retry a few times
    # with a short backoff; the connection is re-attempted each time it is
    # needed, so a briefly-down Ollama recovers on its own.
    content = ""
    last_exc: Exception | None = None
    for attempt in range(int(config.get("llm.retries", 2)) + 1):
        try:
            resp = httpx.post(
                f"{base_url}/api/chat",
                timeout=timeout,
                json={
                    "model": model,
                    "stream": False,
                    "think": False,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "options": {
                        "temperature": float(config.get("llm.temperature", 0.7)),
                        "num_predict": 220,
                    },
                },
            )
            resp.raise_for_status()
            content = (resp.json().get("message") or {}).get("content", "")
            break
        except Exception as exc:  # connection refused, timeout, 5xx, empty body
            last_exc = exc
            if attempt < int(config.get("llm.retries", 2)):
                log.warning("LLM attempt %d failed (%s); retrying", attempt + 1, exc)
                time.sleep(1.0 * (attempt + 1))
            continue
    if content:
        return prepare_script(raw, voice=voice, max_sentences=max_sentences)
        if script:
            return script
        log.warning("LLM returned an empty script; using fallback")
    else:
        log.warning("LLM script generation failed (%s); using fallback", last_exc)
    return fallback_script(track, program=program, language=language)


def _cache_path(text: str, voice: str, speed: float, noise_scale: float,
                noise_w: float = 0.8, sentence_silence: float = 0.0) -> Path:
    digest = hashlib.sha256(
        f"{voice}|{speed}|{noise_scale}|{noise_w}|{sentence_silence}|{text}"
        .encode("utf-8")
    ).hexdigest()[:32]
    return config.DJ_CACHE_DIR / f"{digest}.mp3"


def synthesize(text: str, voice: str | None = None,
               speed: float | None = None,
               noise_scale: float | None = None,
               noise_w: float | None = None,
               sentence_silence: float | None = None) -> Path | None:
    """Render ``text`` to an mp3 with Piper. Returns None on failure.

    The voice (``voice`` or the configured ``dj.voice``) determines the
    language: a Russian voice speaks Cyrillic natively (no transliteration, no
    year->words), everything else is treated as English text for the English
    voices and gets the Cyrillic -> Latin transliteration so the voice can read
    Russian names.
    """
    text = (text or "").strip()
    if not text:
        return None
    voice = voice or config.get("dj.voice", "en_GB-alan-medium")
    language = voice_language(voice)
    # English voices mis-read bare numerals as digits, so years are spelled out
    # as words. The Russian voice, however, reads year numerals natively, so we
    # leave Russian-text dates exactly as written (digits).
    if language != "russian":
        text = _spell_dates(text)
    # Russian (Cyrillic) names cannot be spoken by the English voices — render
    # them as a Latin spelling the voice reads with roughly Russian pronunciation
    # (on-screen dj_text stays Cyrillic; this only affects the audio). The
    # Russian voice reads Cyrillic directly, so it is skipped for Russian.
    if language == "russian":
        text = _normalize_stress_marks(text)
    else:
        text = _strip_stress_marks(text)
        text = _transliterate_cyrillic(text)

    speed = float(speed if speed is not None else config.get("dj.speed", 1.0))
    speed = max(0.5, min(speed, 2.0))
    noise_scale = float(noise_scale if noise_scale is not None
                        else config.get("dj.noise_scale", 0.667))
    noise_scale = max(0.1, min(noise_scale, 2.0))
    noise_w = float(noise_w if noise_w is not None
                    else config.get("dj.noise_w", 0.8))
    noise_w = max(0.0, min(noise_w, 2.0))
    sentence_silence = float(
        sentence_silence if sentence_silence is not None
        else config.get("dj.sentence_silence", 0.25))
    sentence_silence = max(0.0, min(sentence_silence, 2.0))
    out_path = _cache_path(text, voice, speed, noise_scale, noise_w,
                           sentence_silence)
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    binary = piper_binary()
    model = voice_model_path(voice)
    if not binary or not model:
        log.warning("piper unavailable (binary=%s, model=%s)", binary, model)
        return None

    config.DJ_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "dj.wav"
        # Piper's length_scale is inverse to speed.
        length_scale = round(1.0 / speed, 4)
        cmd = [binary, "-m", str(model), "-f", str(wav_path),
               "--length_scale", str(length_scale),
               "--noise_scale", str(round(noise_scale, 4)),
               # Phoneme-width (rhythm) jitter and the silence after each
               # sentence. The model's own punctuation pauses only reach
               # 0.10-0.26 s; without this the DJ sounds rushed.
               "--noise_w_scale", str(round(noise_w, 4)),
               "--sentence_silence", str(round(sentence_silence, 3))]

        try:
            proc = subprocess.run(
                cmd, input=text.encode("utf-8"),
                capture_output=True, timeout=180, check=False,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            log.error("piper invocation failed: %s", exc)
            return None
        if proc.returncode != 0 or not wav_path.exists():
            log.error("piper failed rc=%s: %s", proc.returncode,
                      proc.stderr.decode("utf-8", "replace")[:400])
            return None

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
                  previous: dict[str, Any] | None = None,
                  program: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Generate script + audio for one DJ break. Returns None if unavailable.

    ``program`` (optional) describes the themed program this track opens, so the
    DJ announces the vibe switch ("Next up, a run of Rock...") like a real host.
    """
    script = generate_script(track, previous, program=program)
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


def tts_health() -> dict[str, Any]:
    return {
        "ok": bool(piper_binary() and voice_model_path()),
        "binary": piper_binary(),
        "voice": config.get("dj.voice"),
        "voices_available": available_voices(),
    }


__all__ = [
    "generate_script", "synthesize", "prepare_break", "fallback_script",
    "available_voices", "voice_profiles", "voice_language", "llm_health",
    "tts_health", "audio_duration", "json", "_spell_dates",
]

# Human-readable descriptions of the bundled voices, with notes from community
# feedback about which read most naturally. The DJ picker in the web UI shows
# these so you can pick a voice by character, not just by a model filename.
# ``lang`` marks the voice's language: the DJ speaks the language of the
# *chosen* voice (English voices -> English breaks, Russian voices -> Russian
# breaks). The web UI shows all voices in one picker.
VOICE_PROFILES: list[dict[str, str]] = [
    {
        "id": "en_GB-alan-medium",
        "name": "UK Alan",
        "gender": "male",
        "lang": "english",
        "note": "British male voice; clean and measured for a UK-leaning station.",
    },
    {
        "id": "en_GB-jenny_dioco-medium",
        "name": "UK Jenny Dioco",
        "gender": "female",
        "lang": "english",
        "note": "British female voice; bright and friendly.",
    },
    {
        "id": "en_GB-southern_english_female-low",
        "name": "UK Southern English",
        "gender": "female",
        "lang": "english",
        "note": "British female voice with a gentle southern-English tone; lighter model.",
    },
    {
        "id": "en_US-amy-medium",
        "name": "US Amy",
        "gender": "female",
        "lang": "english",
        "note": "Default. Clear and neutral; good all-rounder but the most "
                "'robotic' of the set on long intros.",
    },
    {
        "id": "en_US-bryce-medium",
        "name": "US Bryce",
        "gender": "male",
        "lang": "english",
        "note": "Deeper male voice; pairs well with a late-night radio persona.",
    },
    {
        "id": "en_US-hfc_female-medium",
        "name": "US HFC Female",
        "gender": "female",
        "lang": "english",
        "note": "High-fidelity female voice; crisp, broadcast-style delivery.",
    },
    {
        "id": "en_US-hfc_male-medium",
        "name": "US HFC Male",
        "gender": "male",
        "lang": "english",
        "note": "High-fidelity male voice; crisp, broadcast-style delivery.",
    },
    {
        "id": "en_US-lessac-high",
        "name": "US Lessac",
        "gender": "female",
        "lang": "english",
        "note": "Warmest, most natural intonation per community feedback; the "
                "usual upgrade pick. Occasionally over-pauses between phrases.",
    },
    {
        "id": "en_US-libritts_r-medium",
        "name": "US LibriTTS-R",
        "gender": "female",
        "lang": "english",
        "note": "Broadest tonal range and most expressive prosody of the "
                "English set; great for lively, varied delivery.",
    },
    {
        "id": "en_US-norman-medium",
        "name": "US Norman",
        "gender": "male",
        "lang": "english",
        "note": "Natural male voice with a steady, announcer-like cadence.",
    },
    {
        "id": "en_US-ryan-high",
        "name": "US Ryan",
        "gender": "male",
        "lang": "english",
        "note": "High-quality Ryan; brighter, more detailed male timbre; the "
                "go-to male option for a different timbre from the women.",
    },
    {
        "id": "en_US-sam-medium",
        "name": "US Sam",
        "gender": "male",
        "lang": "english",
        "note": "Warm male voice; good neutral alternative to the female leads.",
    },
    {
        "id": "ru_RU-irina-medium",
        "name": "Irina",
        "gender": "female",
        "lang": "russian",
        "note": "Russian voice. Selecting it makes the DJ speak Russian "
                "(native Cyrillic, no transliteration).",
    },
    {
        "id": "ru_RU-denis-medium",
        "name": "Denis",
        "gender": "male",
        "lang": "russian",
        "note": "Russian voice (male). Selecting it makes the DJ speak Russian.",
    },
    {
        "id": "ru_RU-dmitri-medium",
        "name": "Dmitri",
        "gender": "male",
        "lang": "russian",
        "note": "Russian voice (male). Selecting it makes the DJ speak Russian.",
    },
    {
        "id": "ru_RU-ruslan-medium",
        "name": "Ruslan",
        "gender": "male",
        "lang": "russian",
        "note": "Russian voice (male). Selecting it makes the DJ speak Russian.",
    },
]


def voice_profiles(lang: str | None = None) -> list[dict[str, str]]:
    """Voice catalogue (id, name, gender, lang, intonation note) for the UI.

    ``lang`` optionally filters to one language; without it the full catalogue
    (English + Russian voices together) is returned for the single picker.
    """
    installed = set(available_voices())
    known = {p["id"] for p in VOICE_PROFILES}
    profiles = []
    for p in VOICE_PROFILES:
        if lang and p.get("lang") != lang:
            continue
        entry = dict(p)
        entry["installed"] = p["id"] in installed
        profiles.append(entry)
    # Include any installed voice not in the curated list (e.g. user-added or a
    # leftover from an earlier catalogue). Humanize the label the same way the
    # curated entries are named: a US/UK prefix and no quality suffix, so the
    # picker never shows a raw model filename like en_US-lessac-medium.
    for v in available_voices():
        if v not in known and (lang is None or voice_language(v) == lang):
            profiles.append({
                "id": v, "name": _humanize_voice_name(v), "gender": "?",
                "lang": voice_language(v), "note": "", "installed": True,
            })
    return profiles


def _humanize_voice_name(voice_id: str) -> str:
    """Turn a Piper voice id into a display label (US/UK prefix, no quality).

    ``en_US-amy-medium`` -> ``US Amy``; ``en_GB-jenny_dioco-medium`` ->
    ``UK Jenny Dioco``; an unknown shape is returned unchanged.
    """
    parts = voice_id.split("-")
    if len(parts) != 3:
        return voice_id
    country, name, _quality = parts
    region = "UK" if country == "en_GB" else "US" if country == "en_US" else country
    pretty = name.replace("_", " ").title()
    return f"{region} {pretty}"
