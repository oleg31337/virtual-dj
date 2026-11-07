"""Russian numbers written out as words.

Why this exists: the station's Russian DJ lines are spoken by a neural voice
(Supertonic) that garbles bare numerals — "в 1989 году" came out as "в сечи 979
году", "1979", "1772" in different takes, while the same sentence with the year
spelled out was correct every time. The previous engine (Piper/espeak) read
digits natively, so the app deliberately left them alone; that assumption no
longer holds for every engine, and spelling them out is correct for both.

Two shapes are needed:

* **cardinal** for counts — "темп 120" -> "темп сто двадцать",
  "3 минуты 45 секунд" -> "три минуты сорок пять секунд";
* **ordinal in the case the following noun demands** for years —
  "в 1989 году" -> "в тысяча девятьсот восемьдесят девятом году",
  "1989 год" -> "тысяча девятьсот восемьдесят девятый год",
  "с 1989 года" -> "с тысяча восемьсот ... девятого года".

Only integers 0..999_999 are handled; anything else (decimals, ranges, longer
numbers) is left untouched rather than guessed at.
"""

from __future__ import annotations

import re

_UNITS = (
    "ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь",
    "девять", "десять", "одиннадцать", "двенадцать", "тринадцать",
    "четырнадцать", "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать",
    "девятнадцать",
)
_TENS = {
    2: "двадцать", 3: "тридцать", 4: "сорок", 5: "пятьдесят", 6: "шестьдесят",
    7: "семьдесят", 8: "восемьдесят", 9: "девяносто",
}
_HUNDREDS = {
    1: "сто", 2: "двести", 3: "триста", 4: "четыреста", 5: "пятьсот",
    6: "шестьсот", 7: "семьсот", 8: "восемьсот", 9: "девятьсот",
}

# Ordinals, masculine singular, in the three cases a year phrase needs.
_ORD = {
    0: ("нулевой", "нулевом", "нулевого"),
    1: ("первый", "первом", "первого"),
    2: ("второй", "втором", "второго"),
    3: ("третий", "третьем", "третьего"),
    4: ("четвёртый", "четвёртом", "четвёртого"),
    5: ("пятый", "пятом", "пятого"),
    6: ("шестой", "шестом", "шестого"),
    7: ("седьмой", "седьмом", "седьмого"),
    8: ("восьмой", "восьмом", "восьмого"),
    9: ("девятый", "девятом", "девятого"),
    10: ("десятый", "десятом", "десятого"),
    11: ("одиннадцатый", "одиннадцатом", "одиннадцатого"),
    12: ("двенадцатый", "двенадцатом", "двенадцатого"),
    13: ("тринадцатый", "тринадцатом", "тринадцатого"),
    14: ("четырнадцатый", "четырнадцатом", "четырнадцатого"),
    15: ("пятнадцатый", "пятнадцатом", "пятнадцатого"),
    16: ("шестнадцатый", "шестнадцатом", "шестнадцатого"),
    17: ("семнадцатый", "семнадцатом", "семнадцатого"),
    18: ("восемнадцатый", "восемнадцатом", "восемнадцатого"),
    19: ("девятнадцатый", "девятнадцатом", "девятнадцатого"),
    20: ("двадцатый", "двадцатом", "двадцатого"),
    30: ("тридцатый", "тридцатом", "тридцатого"),
    40: ("сороковой", "сороковом", "сорокового"),
    50: ("пятидесятый", "пятидесятом", "пятидесятого"),
    60: ("шестидесятый", "шестидесятом", "шестидесятого"),
    70: ("семидесятый", "семидесятом", "семидесятого"),
    80: ("восьмидесятый", "восьмидесятом", "восьмидесятого"),
    90: ("девяностый", "девяностом", "девяностого"),
    100: ("сотый", "сотом", "сотого"),
    200: ("двухсотый", "двухсотом", "двухсотого"),
    300: ("трёхсотый", "трёхсотом", "трёхсотого"),
    400: ("четырёхсотый", "четырёхсотом", "четырёхсотого"),
    500: ("пятисотый", "пятисотом", "пятисотого"),
    600: ("шестисотый", "шестисотом", "шестисотого"),
    700: ("семисотый", "семисотом", "семисотого"),
    800: ("восьмисотый", "восьмисотом", "восьмисотого"),
    900: ("девятисотый", "девятисотом", "девятисотого"),
    1000: ("тысячный", "тысячном", "тысячного"),
    2000: ("двухтысячный", "двухтысячном", "двухтысячного"),
    3000: ("трёхтысячный", "трёхтысячном", "трёхтысячного"),
}

CASES = ("nominative", "prepositional", "genitive")
_CASE_INDEX = {name: i for i, name in enumerate(CASES)}


def _below_thousand(n: int) -> str:
    """0..999 as words (masculine nominative)."""
    if n < 20:
        return _UNITS[n]
    if n < 100:
        tens, unit = divmod(n, 10)
        return _TENS[tens] + (f" {_UNITS[unit]}" if unit else "")
    hundreds, rest = divmod(n, 100)
    head = _HUNDREDS[hundreds]
    return head + (f" {_below_thousand(rest)}" if rest else "")


def _thousands_head(count: int) -> str:
    """``count`` thousand, with the feminine agreement the noun demands.

    1000 -> "тысяча", 2000 -> "две тысячи", 21000 -> "двадцать одна тысяча",
    5000 -> "пять тысяч".
    """
    if count == 1:
        return "тысяча"
    words = _below_thousand(count)
    last_two = count % 100
    if 11 <= last_two <= 14:
        form = "тысяч"
    elif last_two % 10 == 1:
        form = "тысяча"
        words = re.sub(r"\bодин$", "одна", words)
    elif last_two % 10 in (2, 3, 4):
        form = "тысячи"
        words = re.sub(r"\bдва$", "две", words)
    else:
        form = "тысяч"
    return f"{words} {form}"


def cardinal(n: int) -> str:
    """Cardinal number in words, masculine nominative (0..999_999)."""
    if n < 0 or n > 999_999:
        return str(n)
    if n < 1000:
        return _below_thousand(n)
    thousands, rest = divmod(n, 1000)
    head = _thousands_head(thousands)
    return head + (f" {_below_thousand(rest)}" if rest else "")


def _ordinal_tail(n: int, case: str) -> str:
    """Ordinal for ``n`` in ``case``; only 1..999 with real ordinal forms."""
    index = _CASE_INDEX[case]
    if n in _ORD:
        return _ORD[n][index]
    if n < 20:
        return _UNITS[n]  # no ordinal form: fall back to the cardinal
    if n < 100:
        tens, unit = divmod(n, 10)
        if unit == 0:
            return _ORD[tens * 10][index]
        return f"{_TENS[tens]} {_ordinal_tail(unit, case)}"
    hundreds, rest = divmod(n, 100)
    if rest == 0:
        return _ORD[hundreds * 100][index]
    return f"{_HUNDREDS[hundreds]} {_ordinal_tail(rest, case)}"


def year_words(year: int, case: str = "prepositional") -> str:
    """A year as an ordinal phrase: 1989 -> 'тысяча девятьсот восемьдесят девятом'."""
    if year < 1000 or year > 2999:
        return cardinal(year)
    if year in _ORD:
        return _ORD[year][_CASE_INDEX[case]]
    century, rest = divmod(year, 100)
    head = cardinal(century * 100)
    if rest == 0:
        # 1900 -> 'тысяча девятисотом' (only the last word inflects)
        words = head.rsplit(" ", 1)
        last = words[-1]
        ordinal = _ordinal_tail(int({v: k for k, v in _HUNDREDS.items()}[last])
                                * 100, case)
        return " ".join(words[:-1] + [ordinal])
    return f"{head} {_ordinal_tail(rest, case)}"


_YEAR_NOUN_CASE = (
    ("году", "prepositional"),
    ("года", "genitive"),
    ("годов", "genitive"),
    ("годам", "genitive"),
    ("годах", "prepositional"),
    ("год", "nominative"),
)

_YEAR_RE = re.compile(
    r"(?<!\d)(\d{3,4})\s*(год[ауеомх]*|годов|г\.)(?![\w])", re.IGNORECASE)
_INT_RE = re.compile(r"(?<![\d.,])(\d{1,6})(?![\d.,])")
_BPM_RE = re.compile(r"\b(?:BPM|bpm|бпм|БПМ)\b")


def _year_case(noun: str) -> str:
    low = noun.lower()
    for ending, case in _YEAR_NOUN_CASE:
        if low.startswith(ending):
            return case
    return "nominative"


def spell_years(text: str) -> str:
    """Rewrite ``<year> <год-noun>`` pairs into words, keeping the noun."""
    def repl(match: re.Match[str]) -> str:
        year = int(match.group(1))
        noun = match.group(2)
        if year < 1000 or year > 2999:
            return match.group(0)
        if noun.lower().startswith("г."):
            # "1989 г." is read as "в 1989 году" — spell it that way.
            return f"{year_words(year, 'prepositional')} году"
        return f"{year_words(year, _year_case(noun))} {noun}"

    return _YEAR_RE.sub(repl, text)


def _spell_int(match: re.Match[str]) -> str:
    """Spell one integer, keeping a word boundary where a letter touches it.

    "М2", "U2", "2Pac" glue a number to letters: spelling it in place produced
    "Мдва", which the engine pronounces as one nonsense word. A space on that
    side keeps the number its own word without touching the rest of the line.
    """
    words = cardinal(int(match.group(1)))
    text, start, end = match.string, match.start(), match.end()
    before = text[start - 1] if start else ""
    after = text[end] if end < len(text) else ""
    lead = " " if before.isalpha() else ""
    trail = " " if after.isalpha() else ""
    return f"{lead}{words}{trail}"


def spell_numbers_ru(text: str) -> str:
    """Spell every integer in ``text``; years take the case of their noun.

    Decimals and numbers glued to punctuation are left alone: a wrong guess is
    worse than a digit the engine may still read, and the DJ's lines carry
    years, counts and durations, not measurements.
    """
    if not text:
        return text
    out = _BPM_RE.sub("ударов в минуту", text)
    out = spell_years(out)
    # Anything left is a count/duration: cardinal, nominative.
    return _INT_RE.sub(_spell_int, out)


__all__ = ["cardinal", "year_words", "spell_years", "spell_numbers_ru"]
