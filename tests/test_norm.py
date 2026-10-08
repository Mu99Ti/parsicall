import pytest

from parsicall.norm import (
    digits_to_ascii,
    fold_persian_variants,
    jalali_to_iso,
    normalize_zwnj,
    parse_persian_amount,
)


@pytest.mark.parametrize("raw,expected", [
    ("۱۴۰۴", "1404"),
    ("١٤٠٤", "1404"),
    ("۲٬۵۰۰", "2500"),
    ("2500", "2500"),
])
def test_digits_to_ascii(raw, expected):
    assert digits_to_ascii(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("1404/07/17", "2025-10-09"),
    ("۱۴۰۴/۰۷/۱۷", "2025-10-09"),
    ("۱۷ مهر ۱۴۰۴", "2025-10-09"),
    ("", None),
    ("1404/07", None),
])
def test_jalali_to_iso(raw, expected):
    assert jalali_to_iso(raw) == expected


def test_fold_persian_variants():
    assert fold_persian_variants("علي رضايي") == "علی رضایی"


def test_normalize_zwnj():
    assert normalize_zwnj("می\u200cخواهم") == "می خواهم"


@pytest.mark.parametrize("raw,expected", [
    ("۲٬۵۰۰٬۰۰۰ ریال", (2500000, "rial")),
    ("2,500,000 ریال", (2500000, "rial")),
    ("۲۵۰ هزار تومان", (250000, "toman")),
    ("۲۵۰۰۰۰", None),
])
def test_parse_persian_amount(raw, expected):
    assert parse_persian_amount(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("دهم مهر ۱۴۰۴", "2025-10-02"),
    ("بیست و یکم مهر ۱۴۰۴", "2025-10-13"),
    ("سی‌ام اسفند ۱۴۰۳", "2025-03-20"),
    ("سی‌امم مهر ۱۴۰۴", None),
    ("1404/13/01", None),
    ("1404/07/31", None),
    ("مهر ۱۴۰۴", None),
])
def test_jalali_to_iso_contract(raw, expected):
    assert jalali_to_iso(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("۱۰۰ ریال و ۲۰۰ تومان", None),
    ("قیمت ۲۵۰", None),
    ("۲ میلیون ریال", (2000000, "rial")),
    ("10 toman", (10, "toman")),
    ("2.5 هزار ریال", None),
])
def test_parse_persian_amount_contract(raw, expected):
    assert parse_persian_amount(raw) == expected


def test_text_contract():
    assert fold_persian_variants("مُحَمَّد علي") == "محمد علی"
    assert normalize_zwnj("  می\u200cخواهم  ") == "می خواهم"
    assert digits_to_ascii("۲ ۵۰۰") == "2500"
