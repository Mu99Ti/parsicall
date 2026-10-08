import re

_DIGIT_MAP: dict[int, str] = {
    **{0x06F0 + i: str(i) for i in range(10)},
    **{0x0660 + i: str(i) for i in range(10)},
}
_THOUSANDS_SEP = re.compile(r"(?<=\d)[٬،,   ](?=\d)")


def digits_to_ascii(s: str) -> str:
    return _THOUSANDS_SEP.sub("", s.translate(_DIGIT_MAP))
