import re
from pathlib import Path
import tomllib


_VERSION_LINE = re.compile(r'^\s*version\s*=\s*"([^"]+)"\s*$')


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

        # Primary path: parse TOML and read [project].version.
        try:
            with pyproject.open('rb') as f:
                data = tomllib.load(f)
            value = str(data.get('project', {}).get('version', '')).strip()
            if value:
                return value
        except Exception:
            pass

        # Fallback path: line-based parser for partially-invalid TOML files.
        try:
            in_project = False
            for line in pyproject.read_text(encoding='utf-8').splitlines():
                stripped = line.strip()
                if stripped == '[project]':
                    in_project = True
                    continue
                if in_project and stripped.startswith('['):
                    in_project = False
                if not in_project:
                    continue

                match = _VERSION_LINE.match(line)
                if match is not None:
                    return match.group(1)
        except Exception:
            continue

    return 'unknown'
