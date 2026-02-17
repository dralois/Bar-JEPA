import re
from typing import Optional


_NUMERIC_PATTERN = re.compile(
    r'(?<!\d)[+-]?(?:(?:\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?|[.,]\d+))(?:[eE][+-]?\d+)?(?!\d)'
)
_MANTISSA_TEN_POWER_PATTERN = re.compile(
    r'([+-]?(?:(?:\d{1,3}(?:[.,]\d{3})+|\d+)(?:[.,]\d+)?|[.,]\d+))\s*(?:[xX×*]|\\times|\\cdot)\s*10\s*\^\s*\{?\s*([+-]?\d+)\s*\}?'
)
_TEN_POWER_PATTERN = re.compile(r'10\s*\^\s*\{?\s*([+-]?\d+)\s*\}?')
_ALLOWED_PREFIX_PATTERN = re.compile(r'^[\s\'"`\(\[\{<>=~≈≃≅≤≥+\-±$€£¥₹#]*$')
_ALLOWED_SUFFIX_PATTERN = re.compile(r'^[\s\'"`.,;:%\u2030°\)\]\}><>=~≈≃≅≤≥+\-±$€£¥₹:/\\*]*$')
_US_THOUSANDS_ONLY_PATTERN = re.compile(r'^[+-]?\d{1,3}(?:,\d{3})+$')


def _normalize_numeric_token(token: str) -> str:
    """
    Normalizes locale-dependent numeric separators to Python float format.

    :param token: numeric token potentially containing '.' and/or ','
    :return: normalized token using '.' as decimal separator
    """
    if ',' in token and '.' in token:
        # Use the right-most separator as decimal; remove the other as thousands.
        if token.rfind(',') > token.rfind('.'):
            return token.replace('.', '').replace(',', '.')
        return token.replace(',', '')

    if ',' in token:
        # Preserve classic US thousands notation, otherwise treat comma as decimal.
        if _US_THOUSANDS_ONLY_PATTERN.fullmatch(token) is not None:
            return token.replace(',', '')
        return token.replace(',', '.')

    return token


def extract_numeric_value(text: str) -> Optional[float]:
    """
    Extracts numeric value when text is a numeric tick with optional symbols.

    :param text: input text
    :return: parsed float value or None if no valid numeric tick is found
    """
    cleaned = (
        text
        .replace('−', '-')
        .replace('，', ',')
        .replace('‚', ',')
        .strip()
    )

    # Support notation like 6x10^6 / 6×10^6.
    mantissa_power_matches = list(_MANTISSA_TEN_POWER_PATTERN.finditer(cleaned))
    if len(mantissa_power_matches) == 1:
        match = mantissa_power_matches[0]
        prefix = cleaned[:match.start()]
        suffix = cleaned[match.end():]
        if (_ALLOWED_PREFIX_PATTERN.fullmatch(prefix) is not None and
                _ALLOWED_SUFFIX_PATTERN.fullmatch(suffix) is not None):
            try:
                mantissa = float(_normalize_numeric_token(match.group(1)))
                exponent = int(match.group(2))
                return float(mantissa * (10.0 ** exponent))
            except ValueError:
                return None

    # Support scientific notation rendered as 10^{-k} / 10^k.
    power_matches = list(_TEN_POWER_PATTERN.finditer(cleaned))
    if len(power_matches) == 1:
        match = power_matches[0]
        prefix = cleaned[:match.start()]
        suffix = cleaned[match.end():]
        if (_ALLOWED_PREFIX_PATTERN.fullmatch(prefix) is not None and
                _ALLOWED_SUFFIX_PATTERN.fullmatch(suffix) is not None):
            try:
                exponent = int(match.group(1))
                return float(10.0 ** exponent)
            except ValueError:
                return None

    matches = list(_NUMERIC_PATTERN.finditer(cleaned))
    if len(matches) != 1:
        return None

    match = matches[0]
    prefix = cleaned[:match.start()]
    suffix = cleaned[match.end():]

    if _ALLOWED_PREFIX_PATTERN.fullmatch(prefix) is None:
        return None
    if _ALLOWED_SUFFIX_PATTERN.fullmatch(suffix) is None:
        return None

    token = _normalize_numeric_token(match.group(0))
    try:
        return float(token)
    except ValueError:
        return None
