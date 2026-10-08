import re

_FOLD_MAP: dict[int, str] = {
    ord("ي"): "ی",
    ord("ى"): "ی",
    ord("ئ"): "ی",
    ord("ك"): "ک",
    ord("ۀ"): "ه",
    ord("ة"): "ه",
    ord("أ"): "ا",
    ord("إ"): "ا",
    ord("ٱ"): "ا",
    ord("ؤ"): "و",
}
_TASHKEEL: frozenset[str] = frozenset(chr(c) for c in (*range(0x064B, 0x0656), 0x0670))


def fold_persian_variants(s: str) -> str:
    return "".join(ch for ch in s.translate(_FOLD_MAP) if ch not in _TASHKEEL)


def normalize_zwnj(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\u200c", " ")).strip()
