import re
from pathlib import Path


_SECTION_LINE = re.compile(r'^\s*\[([^\]]+)\]\s*$')
_VERSION_LINE = re.compile(r'^\s*version\s*=\s*["\']([^"\']+)["\']\s*(?:#.*)?$')


def resolve_project_version() -> str:
    """
    Resolves the project version for run metadata.

    Reads `version` from the `[project]` section in `pyproject.toml`.

    :return: resolved version string or `"unknown"` if unavailable
    """
    seen = set()
    for base in (Path(__file__).resolve().parents[2], Path.cwd()):
        try:
            key = str(base.resolve())
        except Exception:
            key = str(base)
        if key in seen:
            continue
        seen.add(key)

        pyproject = base / 'pyproject.toml'
        if not pyproject.exists():
            continue

        # Line-based parser that avoids importing a TOML dependency.
        try:
            in_project = False
            for line in pyproject.read_text(encoding='utf-8').splitlines():
                stripped = line.strip()
                section = _SECTION_LINE.match(stripped)
                if section is not None:
                    in_project = section.group(1).strip() == 'project'
                    continue
                if not in_project:
                    continue

                match = _VERSION_LINE.match(stripped)
                if match is not None:
                    return match.group(1)
        except Exception:
            continue

    return 'unknown'
