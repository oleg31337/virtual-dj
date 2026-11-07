"""Russian number spelling: the fix for digits the neural voice garbles."""

from __future__ import annotations

import pytest

from app.ru_numbers import cardinal, spell_numbers_ru, spell_years, year_words


@pytest.mark.parametrize("number,words", [
    (0, "ноль"),
    (7, "семь"),
    (19, "девятнадцать"),
    (21, "двадцать один"),
    (45, "сорок пять"),
    (100, "сто"),
    (120, "сто двадцать"),
    (1989, "тысяча девятьсот восемьдесят девять"),
    (2000, "две тысячи"),
    (2003, "две тысячи три"),
    (21000, "двадцать одна тысяча"),
    (22000, "двадцать две тысячи"),
    (5000, "пять тысяч"),
    (11000, "одиннадцать тысяч"),
])
def test_cardinal(number, words):
    assert cardinal(number) == words


@pytest.mark.parametrize("year,case,words", [
    (1989, "prepositional", "тысяча девятьсот восемьдесят девятом"),
    (1989, "nominative", "тысяча девятьсот восемьдесят девятый"),
    (1989, "genitive", "тысяча девятьсот восемьдесят девятого"),
    (1984, "prepositional", "тысяча девятьсот восемьдесят четвёртом"),
    (2003, "prepositional", "две тысячи третьем"),
    (2000, "prepositional", "двухтысячном"),
    (1900, "prepositional", "тысяча девятисотом"),
    (1971, "nominative", "тысяча девятьсот семьдесят первый"),
])
def test_year_words(year, case, words):
    assert year_words(year, case) == words


def test_years_take_the_case_of_their_noun():
    assert spell_years("в 1989 году") == "в тысяча девятьсот восемьдесят девятом году"
    assert spell_years("1984 год") == "тысяча девятьсот восемьдесят четвёртый год"
    assert spell_years("с 1989 года") == "с тысяча девятьсот восемьдесят девятого года"
    assert spell_years("о 2003 г.") == "о две тысячи третьем году"


def test_counts_are_cardinal():
    assert spell_numbers_ru("темп 120") == "темп сто двадцать"
    assert spell_numbers_ru("длина 3 минуты 45 секунд") == (
        "длина три минуты сорок пять секунд")


def test_bpm_becomes_words_not_letters():
    # The neural voice reads "BPM" as a letter salad; the Russian phrase is what
    # a host actually says.
    assert spell_numbers_ru("темп 120 BPM") == "темп сто двадцать ударов в минуту"
    assert spell_numbers_ru("120 бпм") == "сто двадцать ударов в минуту"


def test_decimals_and_glued_digits_are_left_alone():
    assert spell_numbers_ru("версия 1.5") == "версия 1.5"
    assert spell_numbers_ru("трек №5") == "трек №пять"  # plain counts still go


def test_a_real_dj_line_round_trips_to_words():
    line = ("Этот трек вышел в 1989 году, темп 120 BPM, "
            "длина 3 минуты 45 секунд, а следующий — в 2003 году.")
    out = spell_numbers_ru(line)
    assert "1989" not in out and "120" not in out and "2003" not in out
    assert "тысяча девятьсот восемьдесят девятом году" in out
    assert "сто двадцать ударов в минуту" in out
    assert "две тысячи третьем году" in out
    # Nothing else in the line was damaged.
    assert out.startswith("Этот трек вышел в тысяча")
    assert out.endswith("году.")


def test_empty_and_numberless_text_is_untouched():
    assert spell_numbers_ru("") == ""
    assert spell_numbers_ru("Далее — Кино.") == "Далее — Кино."


def test_a_number_glued_to_letters_keeps_its_word_boundary():
    """`М2` was spelled "Мдва" — one nonsense word the engine could not say.

    Names carry digits: voice styles (М2), bands (U2, 2Pac, Blink-182). The
    number still becomes a word, but on its own.
    """
    assert spell_numbers_ru("слушаем М2 в пять шагов") == \
        "слушаем М два в пять шагов"
    assert spell_numbers_ru("группа U2") == "группа U два"
    assert spell_numbers_ru("включаем 2Pac") == "включаем два Pac"
    # A plain quantity keeps its normal spacing — no doubled blanks.
    assert spell_numbers_ru("темп 120 ударов") == "темп сто двадцать ударов"
