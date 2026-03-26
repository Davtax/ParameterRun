#!/usr/bin/env python
"""
Script to automate version bumping across all project files.

Usage:
    python scripts/bump_version.py <new_version>
    python scripts/bump_version.py --patch
    python scripts/bump_version.py --minor
    python scripts/bump_version.py --major

Example:
    python scripts/bump_version.py 0.2.0
    python scripts/bump_version.py --minor  # bumps X.Y to X.Y+1
"""

import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple


def parse_version(version_str: str) -> Tuple[int, int, int]:
    """Parse semantic version string into (major, minor, patch)."""
    parts = version_str.split(".")
    if len(parts) != 3:
        raise ValueError(f"Invalid version format: {version_str}. Expected: X.Y.Z")
    try:
        major, minor, patch = (int(p) for p in parts)
        return major, minor, patch
    except ValueError as err:
        raise ValueError(f"Version parts must be integers: {version_str}") from err


def version_to_string(major: int, minor: int, patch: int) -> str:
    """Convert (major, minor, patch) to string."""
    return f"{major}.{minor}.{patch}"


def bump_version(version_str: str, bump_type: str) -> str:
    """Bump version based on type: 'major', 'minor', or 'patch'."""
    major, minor, patch = parse_version(version_str)

    if bump_type == "major":
        major += 1
        minor = 0
        patch = 0
    elif bump_type == "minor":
        minor += 1
        patch = 0
    elif bump_type == "patch":
        patch += 1
    else:
        raise ValueError(f"Unknown bump type: {bump_type}")

    return version_to_string(major, minor, patch)


def get_current_version(init_file: Path) -> str:
    """Extract current version from __init__.py."""
    content = init_file.read_text()
    match = re.search(r'__version__\s*=\s*"([^"]+)"', content)
    if not match:
        raise ValueError(f"Could not find __version__ in {init_file}")
    return match.group(1)


def update_init_file(init_file: Path, new_version: str) -> None:
    """Update version in __init__.py."""
    content = init_file.read_text()
    updated = re.sub(
        r'__version__\s*=\s*"[^"]+"',
        f'__version__ = "{new_version}"',
        content
    )
    init_file.write_text(updated)
    print(f"✓ Updated {init_file.relative_to(init_file.parent.parent.parent)}")


def update_changelog(changelog_file: Path, new_version: str) -> None:
    """Update CHANGELOG.md with new version section."""
    content = changelog_file.read_text()

    # Create new version entry
    date = datetime.now().strftime("%Y-%m-%d")
    new_entry = f"""## [{new_version}] - {date}

### Added
- (Your changes here)

"""

    # Replace [Unreleased] section with new version and restore [Unreleased]
    updated = re.sub(
        r"## \[Unreleased\]",
        f"{new_entry}## [Unreleased]",
        content
    )

    # If no changes were made, insert new version after the header
    if updated == content:
        lines = content.split("\n")
        insert_idx = next(i for i, line in enumerate(lines) if line.startswith("## "))
        lines.insert(insert_idx, new_entry.rstrip("\n"))
        updated = "\n".join(lines)

    changelog_file.write_text(updated)
    print(f"✓ Updated {changelog_file.relative_to(changelog_file.parent.parent.parent)}")


def main() -> None:
    project_root = Path(__file__).parent.parent
    init_file = project_root / "src" / "parameterrun" / "__init__.py"
    changelog_file = project_root / "CHANGELOG.md"

    # Parse arguments
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    arg = sys.argv[1]

    # Get current version
    current_version = get_current_version(init_file)
    print(f"Current version: {current_version}\n")

    # Determine new version
    if arg.startswith("--"):
        bump_type = arg[2:]
        if bump_type not in ["major", "minor", "patch"]:
            print(f"Invalid bump type: {bump_type}")
            sys.exit(1)
        new_version = bump_version(current_version, bump_type)
    else:
        # Validate provided version format
        new_version = arg
        try:
            parse_version(new_version)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

    print(f"New version: {new_version}\n")

    # Update files
    try:
        update_init_file(init_file, new_version)
        update_changelog(changelog_file, new_version)
        print(f"\n✓ Version successfully bumped to {new_version}")
        print("  Remember to: commit, tag, and push the changes")
        print(f"  Example: git tag v{new_version}")
    except Exception as e:
        print(f"Error during update: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

