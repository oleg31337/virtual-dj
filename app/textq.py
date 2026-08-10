"""Text quality checks and filename/folder metadata guessing.

Two jobs:

1. Decide whether a piece of metadata is *usable for an English-speaking DJ* —
   readable Latin script, not mojibake, not a placeholder like "01" or "track
   03". Anything else is excluded from playlists rather than mumbled over.
2. Guess artist / title / album from the file path when tags are missing,
   using the conventions real collections actually follow
   ("Artist - Title.mp3", "Artist/Album/01 Title.mp3", ...).

Pure functions only: no I/O, no network, no config. That keeps the rules
cheap enough to run over a 13k-file library and trivial to unit-test.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

# --- script / readability --------------------------------------------------

# Unicode blocks we cannot pronounce in an English radio show. Detected by
# character name prefix, which covers every block variant without hardcoding
# codepoint ranges.
_NON_LATIN_SCRIPTS = (
    "CYRILLIC", "GREEK", "COPTIC", "ARABIC", "HEBREW", "SYRIAC", "THAANA",
    "DEVANAGARI", "BENGALI", "GURMUKHI", "GUJARATI", "ORIYA", "TAMIL",
    "TELUGU", "KANNADA", "MALAYALAM", "SINHALA", "THAI", "LAO", "TIBETAN",
    "MYANMAR", "GEORGIAN", "ETHIOPIC", "CHEROKEE", "KHMER", "MONGOLIAN",
    "HIRAGANA", "KATAKANA", "HANGUL", "CJK", "YI", "ARMENIAN",
)

# Sequences that only occur when UTF-8/CP1251 bytes were decoded as Latin-1.
# e.g. Cyrillic "Синагогальная" mis-decoded shows up as "Ñèíàãîãàëüíàÿ".
_MOJIBAKE_MARKERS = (
    "Ã", "Ð", "Ñ", "Â", "â€", "Ã¢", "Ã©", "Ã¨", "Ãª", "Ã¯", "Ã´", "Ã»",
    "å", "æ", "ð", "ñ", "ø", "þ", "ý",
)

_LATIN1_ACCENTED = re.compile(r"[\u00C0-\u00FF]")
_LETTERS = re.compile(r"[^\W\d_]", re.UNICODE)

# Decorative junk collections love: "- - - - Artist - - - -", "___Title___".
_DECORATION = re.compile(r"(?:\s*[-–—_~*=]\s*){2,}")
_BRACKET_JUNK = re.compile(
    r"\[(?:www\.[^\]]+|[^\]]*\b(?:mp3|kbps|rip|www|\.com|\.ru|\.net)\b[^\]]*)\]",
    re.IGNORECASE,
)
_URLISH = re.compile(r"\b(?:www\.\S+|\S+\.(?:com|ru|net|org|info)\b)", re.IGNORECASE)

# Placeholder titles that carry no information.
_PLACEHOLDER_WORDS = {
    "unknown", "unknown artist", "unknown album", "unknown title",
    "untitled", "track", "tracks", "audiotrack", "audio track", "no artist",
    "various", "various artists", "va", "none", "n/a", "na", "misc",
    "unnamed", "new recording", "sound", "audio", "music", "song",
}
_TRACK_LABEL = re.compile(
    r"^(?:track|trk|audiotrack|pista|titel|song)\s*[-_. ]?\s*\d{1,3}$",
    re.IGNORECASE,
)
_LEADING_TRACKNUM = re.compile(r"^\s*\(?\d{1,3}\)?\s*(?:[-._)\]]|\s)\s*")
_TRAILING_NOISE = re.compile(
    r"\s*[\(\[]\s*(?:official|hq|hd|audio|lyrics?|video|remaster(?:ed)?"
    r"|explicit|album version|www\.[^\)\]]*)\s*[^\)\]]*[\)\]]\s*$",
    re.IGNORECASE,
)


def strip_decorations(text: str) -> str:
    """Remove separator art, site tags and URLs from a name."""
    if not text:
        return ""
    out = _BRACKET_JUNK.sub(" ", text)
    out = _URLISH.sub(" ", out)
    # Normalise underscore-wrapped separators ("Artist_-_Title") *before*
    # collapsing decoration, otherwise "_-_" looks like separator art and the
    # artist/title boundary is destroyed.
    out = re.sub(r"_+\s*-\s*_+", " - ", out)
    out = out.replace("_", " ")
    out = _DECORATION.sub(" ", out)
    out = out.strip(" -–—_~*=.\t")
    return re.sub(r"\s+", " ", out).strip()


def has_non_latin_script(text: str) -> bool:
    """True if any letter belongs to a script we cannot read on air."""
    for char in text or "":
        if not char.isalpha():
            continue
        try:
            name = unicodedata.name(char)
        except ValueError:
            return True  # unnamed codepoint — treat as unreadable
        if name.startswith(_NON_LATIN_SCRIPTS):
            return True
    return False


def looks_like_mojibake(text: str) -> bool:
    """Detect text whose bytes were decoded with the wrong codec.

    Two signals: known mis-decode digraphs, and an implausible density of
    Latin-1 accented characters (real French/Spanish stays well under a third).
    """
    if not text:
        return False
    letters = _LETTERS.findall(text)
    if len(letters) < 3:
        return False
    if any(marker in text for marker in _MOJIBAKE_MARKERS[:6]):
        return True
    accented = len(_LATIN1_ACCENTED.findall(text))
    return accented / len(letters) > 0.34


def is_placeholder(text: str) -> bool:
    """True for names that carry no identifying information."""
    cleaned = strip_decorations(text or "").strip(" .-_")
    if not cleaned:
        return True
    lowered = cleaned.casefold()
    if lowered in _PLACEHOLDER_WORDS or _TRACK_LABEL.match(lowered):
        return True
    # Pure numbering: "01", "1-04", "05.", "(12)"
    if not _LETTERS.search(cleaned):
        return True
    # A single letter is not a usable title.
    return len(_LETTERS.findall(cleaned)) < 2


def is_usable(text: str | None) -> bool:
    """A name is usable when it is present, readable Latin, and meaningful."""
    if not text or not text.strip():
        return False
    if has_non_latin_script(text) or looks_like_mojibake(text):
        return False
    return not is_placeholder(text)


def rejection_reason(artist: str | None, title: str | None) -> str | None:
    """Why this track cannot be announced, or None when it is fine.

    Reasons are stable identifiers so the UI can group them:
    ``no_title``/``no_artist`` (nothing usable found), ``non_latin``
    (unreadable script), ``mojibake`` (wrong-codec text).
    """
    for value in (title, artist):
        if value and has_non_latin_script(value):
            return "non_latin"
        if value and looks_like_mojibake(value):
            return "mojibake"
    if not is_usable(title):
        return "no_title"
    if not is_usable(artist):
        return "no_artist"
    return None


# --- path-based guessing ---------------------------------------------------

# Folder names that organise a collection but are never an artist.
_CONTAINER_FOLDERS = {
    "mp3", "mp3s", "music", "audio", "songs", "tracks", "media", "misc",
    "various", "various artists", "va", "compilations", "compilation",
    "soundtrack", "soundtracks", "ost", "games", "game", "movies", "film",
    "films", "albums", "album", "singles", "collection", "collections",
    "discography", "best", "best of", "hits", "greatest hits", "new folder",
    "downloads", "shared", "library", "cd", "cd1", "cd2", "disc", "disk",
    "unsorted", "other", "temp", "incoming", "radio", "mixes", "live",
}


def is_container_folder(name: str | None) -> bool:
    """True for organisational folders that must not become an artist name."""
    if not name:
        return True
    return clean_name(name).casefold() in _CONTAINER_FOLDERS


_SEPARATORS = (" - ", " – ", " — ", " -- ", "_-_", " _ ")

# Album folders are often prefixed with a year: "1987 One Second",
# "1987-The Uplift Mofo Party Plan", "(2012) Panic".
_YEAR_PREFIX = re.compile(r"^\s*[\(\[]?(19|20)\d{2}[\)\]]?\s*[-–—._]?\s+?")
_YEAR_ONLY = re.compile(r"^\s*[\(\[]?(19|20)\d{2}[\)\]]?\s*$")
# Disc/medium markers that are not part of any name.
_DISC_PREFIX = re.compile(
    r"^\s*(?:cd|disc|disk|part|pt)\s*\.?\s*\d{1,2}\s*[-–—._)]?\s*", re.IGNORECASE
)


def clean_name(text: str | None) -> str:
    """Normalise one metadata value pulled from a path."""
    if not text:
        return ""
    out = strip_decorations(text)
    out = _TRAILING_NOISE.sub("", out)
    out = out.replace("_", " ")
    out = re.sub(r"\s+", " ", out).strip(" .-")
    return out


def strip_leading_junk(text: str) -> str:
    """Drop disc markers, track numbers and year prefixes from the front."""
    out = text or ""
    for _ in range(3):  # e.g. "CD2 - 08 - Title"
        before = out
        out = _DISC_PREFIX.sub("", out)
        out = _YEAR_PREFIX.sub("", out)
        out = _LEADING_TRACKNUM.sub("", out)
        if out == before:
            break
    return out.strip()


def _split_pair(stem: str) -> tuple[str | None, str | None]:
    """Split "Artist - Title" into its two halves, if that shape is present."""
    for sep in _SEPARATORS:
        if sep in stem:
            left, right = stem.split(sep, 1)
            left, right = clean_name(left), clean_name(right)
            if left and right:
                return left, right
    # Bare hyphen without spaces, e.g. "Artist-Title" — only when both sides
    # look like words rather than a hyphenated single title.
    if stem.count("-") == 1:
        left, right = (part.strip() for part in stem.split("-", 1))
        if len(left) > 2 and len(right) > 2 and " " in f"{left}{right}":
            return clean_name(left), clean_name(right)
    return None, None


def guess_from_path(path: Path, root: Path | None = None) -> dict[str, str | None]:
    """Best-effort ``{artist, title, album, source}`` from a file path.

    Handles the common layouts:
      ``Artist - Title.mp3``
      ``Artist/Album/01 - Title.mp3``
      ``Artist - Album/01 Title.mp3``
      ``Artist/1987 Album/04 Title.mp3``
      ``Artist/CD2_-_08_-_Artist_-_Title.mp3``
    ``source`` records which parts came from the path so callers can decide
    how much to trust them.
    """
    # Strip disc/track/year prefixes, then look for an "Artist - Title" split.
    raw_stem = strip_leading_junk(strip_decorations(path.stem))
    artist, title = _split_pair(raw_stem)

    if artist and title:
        # "CD2 08 Snap Do You See The Light" style leftovers: a split that
        # produced a numeric-only left side is not an artist.
        if not _LETTERS.search(artist):
            artist, title = None, clean_name(raw_stem)
    if not title:
        title = clean_name(raw_stem) or None
    # A three-part "Artist - Album - Title" leaves the album in the middle;
    # the last segment is the most reliable title.
    if title and title.count(" - ") >= 1:
        tail = title.rsplit(" - ", 1)[-1].strip()
        if tail and _LETTERS.search(tail):
            title = tail

    parents = [p.name for p in path.parents]
    if root is not None:
        try:
            depth = len(path.relative_to(root).parts) - 1
        except ValueError:
            depth = len(parents)
        parents = parents[:max(0, depth)]

    album_raw = parents[0] if parents else None
    album = None
    folder_artist = None

    if album_raw:
        # "Artist - Album" folder gives us both.
        folder_left, folder_right = _split_pair(strip_decorations(album_raw))
        if folder_left and folder_right:
            # "1987 - Album" is a year, not an artist.
            if _YEAR_ONLY.match(folder_left):
                album = clean_name(folder_right)
            else:
                folder_artist, album = folder_left, clean_name(folder_right)
        else:
            album = clean_name(strip_leading_junk(album_raw))

    # Otherwise the nearest non-container ancestor folder is usually the
    # artist. "Red Hot Chili Peppers/Mp3/1987-Album/04 Title.mp3" must skip
    # both "Mp3" and the year folder to reach the band.
    if not folder_artist:
        for parent_name in parents[1:4]:
            candidate = clean_name(parent_name)
            if not candidate or _YEAR_ONLY.match(candidate):
                continue
            if is_container_folder(candidate):
                continue
            folder_artist = candidate
            break

    if not artist:
        artist = folder_artist

    # A "1987" or "Mp3" style folder is not an artist name.
    if artist and (
        _YEAR_ONLY.match(artist)
        or not _LETTERS.search(artist)
        or is_container_folder(artist)
    ):
        artist = None

    return {
        "artist": artist or None,
        "title": title or None,
        "album": album or None,
        "source": "path",
    }


__all__ = [
    "clean_name", "strip_decorations", "guess_from_path", "is_usable",
    "is_placeholder", "has_non_latin_script", "looks_like_mojibake",
    "rejection_reason",
]
