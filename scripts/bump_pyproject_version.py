#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path

PROJECT_SECTION = '[project]'
VERSION_RE = re.compile(r'^(\s*version\s*=\s*")(\d+)\.(\d+)\.(\d+)(".*)$')


def read_current_version(pyproject: Path) -> str:
    in_project = False
    for line in pyproject.read_text(encoding='utf-8').splitlines():
        stripped = line.strip()
        if stripped == PROJECT_SECTION:
            in_project = True
            continue
        if in_project and stripped.startswith('['):
            in_project = False
        if not in_project:
            continue

        match = VERSION_RE.match(line)
        if match is not None:
            return f'{match.group(2)}.{match.group(3)}.{match.group(4)}'

    raise RuntimeError('Could not find [project].version in pyproject.toml')


def bump_patch_version(pyproject: Path) -> str:
    lines = pyproject.read_text(encoding='utf-8').splitlines()
    in_project = False
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped == PROJECT_SECTION:
            in_project = True
            continue
        if in_project and stripped.startswith('['):
            in_project = False
        if not in_project:
            continue

        match = VERSION_RE.match(line)
        if match is None:
            continue

        major = int(match.group(2))
        minor = int(match.group(3))
        patch = int(match.group(4)) + 1
        new_version = f'{major}.{minor}.{patch}'
        lines[idx] = f'{match.group(1)}{new_version}{match.group(5)}'
        pyproject.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        return new_version

    raise RuntimeError('Could not bump [project].version in pyproject.toml')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pyproject',
        default='pyproject.toml',
        help='Path to pyproject.toml'
    )
    parser.add_argument(
        '--print-current',
        action='store_true',
        help='Print current version without changing the file'
    )
    args = parser.parse_args()

    pyproject = Path(args.pyproject)
    if not pyproject.exists():
        print(f'File not found: {pyproject}', file=sys.stderr)
        return 1

    try:
        if args.print_current:
            print(read_current_version(pyproject))
        else:
            print(bump_patch_version(pyproject))
    except RuntimeError as err:
        print(str(err), file=sys.stderr)
        return 1

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
