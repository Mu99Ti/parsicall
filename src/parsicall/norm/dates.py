import re

from persiantools.jdatetime import JalaliDate

from parsicall.norm.digits import digits_to_ascii

_MONTHS: tuple[str, ...] = (
    "فروردین",
    "اردیبهشت",
    "خرداد",
    "تیر",
    "مرداد",
    "شهریور",
    "مهر",
    "آبان",
    "آذر",
    "دی",
    "بهمن",
    "اسفند",
)
_ORDINALS: dict[str, int] = {
    "یکم": 1,
    "دوم": 2,
    "سوم": 3,
    "چهارم": 4,
    "پنجم": 5,
    "ششم": 6,
    "هفتم": 7,
    "هشتم": 8,
    "نهم": 9,
    "دهم": 10,
    "یازدهم": 11,
    "دوازدهم": 12,
    "سیزدهم": 13,
    "چهاردهم": 14,
    "پانزدهم": 15,
    "شانزدهم": 16,
    "هفدهم": 17,
    "هجدهم": 18,
    "نوزدهم": 19,
    "بیستم": 20,
    "بیستویکم": 21,
    "بیستودوم": 22,
    "بیستوسوم": 23,
    "بیستوچهارم": 24,
    "بیستوپنجم": 25,
    "بیستوششم": 26,
    "بیستوهفتم": 27,
    "بیستوهشتم": 28,
    "بیستونهم": 29,
    "سیام": 30,
}
_NUMERIC_RE = re.compile(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})")
_NAMED_RE = re.compile(
    r"(\d{1,2}|.+?)\s+(" + "|".join(map(re.escape, _MONTHS)) + r")\s+(\d{4})"
)


def _day_from_word(word: str) -> int | None:
    return _ORDINALS.get(word.replace("\u200c", "").replace(" ", ""))


def jalali_to_iso(s: str) -> str | None:
    text = digits_to_ascii(s.strip())
    numeric = _NUMERIC_RE.fullmatch(text)
    if numeric:
        year = int(numeric.group(1))
        month = int(numeric.group(2))
        day = int(numeric.group(3))
    else:
        named = _NAMED_RE.fullmatch(text)
        if not named:
            return None
        day_text, month_text, year_text = named.groups()
        if day_text.isdigit():
            day = int(day_text)
        else:
            ordinal = _day_from_word(day_text)
            if ordinal is None:
                return None
            day = ordinal
        month = _MONTHS.index(month_text) + 1
        year = int(year_text)
    try:
        iso: str = JalaliDate(year, month, day).to_gregorian().isoformat()
    except ValueError:
        return None
    return iso
