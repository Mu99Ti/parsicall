import re

from parsicall.norm.digits import digits_to_ascii

_UNITS: dict[str, int] = {"هزار": 1_000, "میلیون": 1_000_000}
_CURRENCIES: dict[str, str] = {"ریال": "rial", "تومان": "toman", "rial": "rial", "toman": "toman"}
_CURRENCY_RE = re.compile(r"\b(?:rial|toman)\b|ریال|تومان", re.IGNORECASE)
_AMOUNT_RE = re.compile(r"(\d+)(?:[.٫](\d+))?")


def parse_persian_amount(s: str) -> tuple[int, str] | None:
    text = digits_to_ascii(s)
    found = {_CURRENCIES[m.group().lower()] for m in _CURRENCY_RE.finditer(text)}
    if len(found) != 1:
        return None
    multipliers = {mult for word, mult in _UNITS.items() if word in text}
    if len(multipliers) > 1:
        return None
    amount = _AMOUNT_RE.search(text)
    if amount is None or amount.group(2) is not None:
        # ponytail: decimals rejected fail-closed — add when repair stage needs them
        return None
    value = int(amount.group(1)) * (multipliers.pop() if multipliers else 1)
    return value, found.pop()
