import re
from typing import Optional


_NUMERIC_PATTERN = re.compile(
    r'[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?'
)
_ALLOWED_PREFIX_PATTERN = re.compile(r'^[\s\(\[\{<>=~≈≃≅≤≥+\-±$€£¥₹#]*$')
_ALLOWED_SUFFIX_PATTERN = re.compile(r'^[\s%\u2030°\)\]\}><>=~≈≃≅≤≥+\-±$€£¥₹:/\\*]*$')


def extract_numeric_value(text: str) -> Optional[float]:
    """
    Extracts numeric value when text is a numeric tick with optional symbols.

    :param text: input text
    :return: parsed float value or None if no valid numeric tick is found
    """
    cleaned = text.replace('−', '-').strip()
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

    token = match.group(0).replace(',', '')
    try:
        return float(token)
    except ValueError:
        return None
