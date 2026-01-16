#!/usr/bin/env python3
"""Sync release dates from MODEL_RELEASE_DATES in irt.ipynb to model_run_mapping.json."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebook",
        type=Path,
        default=Path("IRT/irt.ipynb"),
        help="Path to irt.ipynb containing MODEL_RELEASE_DATES (default: %(default)s)",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("IRT/data/model_run_mapping.json"),
        help="JSON file to update (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report proposed changes without modifying the mapping file.",
    )
    return parser.parse_args()


def _normalize(text: str) -> str:
    """Lowercase and strip to alphanumeric characters for fuzzy matching."""
    return "".join(ch for ch in text.lower() if ch.isalnum())


def load_model_release_dates(notebook_path: Path) -> dict[str, str]:
    notebook = json.loads(notebook_path.read_text())
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source") or [])
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "MODEL_RELEASE_DATES":
                    data = ast.literal_eval(node.value)
                    return {
                        str(key).strip(): str(value).strip()
                        for key, value in data.items()
                        if value
                    }
    raise ValueError(
        f"MODEL_RELEASE_DATES assignment was not found in {notebook_path}"
    )


def build_lookup(release_dates: dict[str, str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for name, date in release_dates.items():
        normalized = _normalize(name)
        if not normalized:
            continue
        # Later entries override earlier ones to prefer the most specific labels.
        lookup[normalized] = date
    return lookup


def update_mapping(
    mapping_path: Path,
    lookup: dict[str, str],
    dry_run: bool,
) -> Tuple[int, int]:
    data: dict[str, Any] = json.loads(mapping_path.read_text())
    updated = 0
    matched = 0
    for key, spec in data.items():
        alias = spec.get("alias") or key
        candidates = [key, alias]
        found_date = None
        for candidate in candidates:
            normalized = _normalize(candidate)
            if normalized in lookup:
                found_date = lookup[normalized]
                break
        if found_date is None:
            continue
        matched += 1
        if spec.get("release_date") != found_date:
            spec["release_date"] = found_date
            updated += 1
    if not dry_run:
        mapping_path.write_text(json.dumps(data, indent=2) + "\n")
    return matched, updated


def main() -> None:
    args = parse_args()
    release_dates = load_model_release_dates(args.notebook)
    lookup = build_lookup(release_dates)
    matched, updated = update_mapping(args.mapping, lookup, args.dry_run)
    action = "Would update" if args.dry_run else "Updated"
    print(f"{action} release dates for {updated} / {matched} matched entries.")


if __name__ == "__main__":
    main()
