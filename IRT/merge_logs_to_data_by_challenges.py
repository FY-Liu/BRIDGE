#!/usr/bin/env python3
"""Merge run log files into data_by_challenges grouped by challenge.

This script copies `*_run_log.json` files from one or more source directories
into `data/data_by_challenges`, placing each file in a challenge-specific
folder derived from the task metadata inside the log.

Usage examples:
    python merge_logs_to_data_by_challenges.py
    python merge_logs_to_data_by_challenges.py -s /path/to/new/logs -s other_dir --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, Optional

# Default locations relative to the repository root.
DEFAULT_SOURCE = Path("data/logs_input_output_tokens_constraints")
DEFAULT_DEST = Path("data/data_by_challenges")

# Known task_author/competition prefixes mapped to the folder prefix used in data_by_challenges.
AUTHOR_PREFIX_MAP: Dict[str, str] = {
    "hackthebox": "htb",
    "hkcert-ctf": "hkcert",
    "losfuzzys": "glacier",
    "project-sekai-ctf": "sekai",
}


def slugify(value: str) -> str:
    """Convert a value into a filesystem-friendly slug.

    - Strips a leading difficulty tag like "[Easy] ".
    - Inserts underscores in CamelCase strings.
    - Replaces non-alphanumeric chars with underscores.
    - Removes leading digits/underscores and lowercases the result.
    """
    if not value:
        return ""

    value = re.sub(r"^\[[^\]]+\]\s*", "", value)
    value = re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", value)
    value = re.sub(r"[\\/\s]+", "_", value)
    value = re.sub(r"[^A-Za-z0-9]+", "_", value)
    value = value.strip("_")
    value = re.sub(r"^[0-9_]+", "", value)
    return value.lower()


def slugify_no_camel(value: str) -> str:
    """Slugify without camel-case splitting (helps match legacy folder names)."""
    if not value:
        return ""

    value = re.sub(r"^\[[^\]]+\]\s*", "", value)
    value = re.sub(r"[\\/\s]+", "_", value)
    value = re.sub(r"[^A-Za-z0-9]+", "_", value)
    value = value.strip("_")
    value = re.sub(r"^[0-9_]+", "", value)
    return value.lower()


def prefix_from_task(task: Dict) -> Optional[str]:
    """Resolve the provider prefix (e.g., htb, hkcert, glacier, sekai)."""
    competition = task.get("competition") or {}
    author = (competition.get("task_author") or task.get("task_author") or "").lower()
    comp_name = (competition.get("competition_name") or "").lower()
    path = task.get("path") or ""
    path_author = ""
    parts = path.split("/")
    if len(parts) > 1:
        path_author = parts[1].lower()

    for candidate in (author, comp_name, path_author):
        if candidate in AUTHOR_PREFIX_MAP:
            return AUTHOR_PREFIX_MAP[candidate]

    for candidate in (author, path_author, comp_name):
        slug = slugify(candidate)
        if slug:
            return slug

    return None


def target_dir_name(task: Dict, dest_root: Path) -> Optional[str]:
    """Build the destination folder name for a log file."""
    prefix = prefix_from_task(task)
    name = task.get("name") or Path(task.get("path", "")).name
    slug_candidates = []
    for slug in (slugify(name), slugify_no_camel(name)):
        if slug and slug not in slug_candidates:
            slug_candidates.append(slug)
    if not slug_candidates:
        return None
    # Prefer an existing folder if present to avoid creating near-duplicate names.
    for slug in slug_candidates:
        candidate = f"{prefix}_{slug}" if prefix else slug
        if (dest_root / candidate).exists():
            return candidate
    challenge_slug = slug_candidates[0]
    return f"{prefix}_{challenge_slug}" if prefix else challenge_slug


def iter_run_logs(sources: Iterable[Path]) -> Iterable[Path]:
    """Yield all run log files from the provided sources."""
    for source in sources:
        if not source.exists():
            continue
        yield from source.rglob("*_run_log.json")


def copy_log(
    log_path: Path,
    dest_root: Path,
    overwrite: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
) -> str:
    """Copy a single log file into the appropriate challenge folder."""
    try:
        with log_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return "failed_to_load_json"

    target = target_dir_name(payload.get("task") or {}, dest_root)
    if not target:
        return "missing_metadata"

    dest_dir = dest_root / target
    dest_file = dest_dir / log_path.name
    if dest_file.exists() and not overwrite:
        return "skipped_existing"

    if verbose or dry_run:
        print(f"{log_path} -> {dest_file}")

    if not dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(log_path, dest_file)

    return "copied"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge run logs into data_by_challenges.")
    parser.add_argument(
        "-s",
        "--source",
        action="append",
        type=Path,
        help=f"Source directory containing run logs (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "-d",
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Destination root directory (default: {DEFAULT_DEST})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the destination.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without writing files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each copy action.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = args.source or [DEFAULT_SOURCE]
    dest_root = args.dest

    stats: Dict[str, int] = {
        "copied": 0,
        "skipped_existing": 0,
        "failed_to_load_json": 0,
        "missing_metadata": 0,
    }

    for log_path in iter_run_logs(sources):
        result = copy_log(
            log_path,
            dest_root,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        if result not in stats:
            stats[result] = 0
        stats[result] += 1

    total = sum(stats.values())
    print(
        f"Processed {total} log(s). "
        f"Copied: {stats['copied']}, "
        f"Skipped (existing): {stats['skipped_existing']}, "
        f"Missing metadata: {stats['missing_metadata']}, "
        f"Failed to load JSON: {stats['failed_to_load_json']}."
    )


if __name__ == "__main__":
    main()
