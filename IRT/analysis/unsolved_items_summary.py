#!/usr/bin/env python3
"""Summarize items that no model solved in all_a_pyirt.jsonl."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESPONSES = REPO_ROOT / "data" / "all_a_pyirt.jsonl"
DEFAULT_PARAMS = REPO_ROOT / "params" / "all_a_pyirt.csv"
ALL_RUNS_PATH = REPO_ROOT / "data" / "all_runs.jsonl"
MLEBENCH_RESULTS_PATH = REPO_ROOT / "data" / "mlebench_normalized_results.jsonl"
GDPVAL_RESULTS_PATH = REPO_ROOT / "data" / "gdpval_normalized_results.jsonl"
SWEBENCH_RESULTS_PATH = REPO_ROOT / "data" / "swebench_normalized_results.jsonl"
CYBENCH_RESULTS_PATH = REPO_ROOT / "data" / "cybench_normalized_results.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count tasks no model solved, compute their average difficulty, "
            "and list which benchmarks they came from."
        )
    )
    parser.add_argument(
        "--responses",
        type=Path,
        default=DEFAULT_RESPONSES,
        help="Path to all_a_pyirt.jsonl (responses per model).",
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=DEFAULT_PARAMS,
        help="Path to all_a_pyirt.csv (IRT parameters; uses 'b' as difficulty).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of benchmark rows to print.",
    )
    return parser.parse_args()


def load_attempts(responses_path: Path) -> Tuple[Counter, Counter]:
    """Return (attempts, successes) counters keyed by item_id."""
    attempts: Counter = Counter()
    successes: Counter = Counter()

    with responses_path.open() as fh:
        for line in fh:
            record = json.loads(line)
            responses: Dict[str, int] = record.get("responses", {})
            for item_id, value in responses.items():
                attempts[item_id] += 1
                if value:
                    successes[item_id] += 1

    return attempts, successes


def load_jsonl_records(path: Path) -> Iterable[dict]:
    if not path.exists():
        return []
    with path.open() as fh:
        for line in fh:
            yield json.loads(line)


def load_difficulties(params_path: Path) -> Dict[str, float]:
    """Load difficulty ('b') scores keyed by item id."""
    difficulties: Dict[str, float] = {}
    with params_path.open() as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            return difficulties
        item_field = reader.fieldnames[0]
        for row in reader:
            item_id = row.get(item_field)
            if not item_id:
                continue
            try:
                difficulties[item_id] = float(row["b"])
            except (KeyError, ValueError, TypeError):
                continue
    return difficulties


def build_task_source_lookup() -> Dict[str, Dict]:
    """Reproduce task source mapping logic used in analysis.ipynb."""
    metr_task_sources: Dict[str, str] = {}
    for record in load_jsonl_records(ALL_RUNS_PATH):
        task_id = record.get("task_id")
        task_source = record.get("task_source")
        if task_id and task_source and task_id not in metr_task_sources:
            metr_task_sources[task_id] = task_source.lower().replace("-", "")

    mlebench_task_ids = {record["task_id"] for record in load_jsonl_records(MLEBENCH_RESULTS_PATH) if "task_id" in record}
    gdpval_task_ids = {record["task_id"] for record in load_jsonl_records(GDPVAL_RESULTS_PATH) if "task_id" in record}
    swebench_task_ids = {
        record["task_id"] if "task_id" in record else record.get("id")
        for record in load_jsonl_records(SWEBENCH_RESULTS_PATH)
        if ("task_id" in record) or ("id" in record)
    }
    cybench_task_ids = {record["task_id"] for record in load_jsonl_records(CYBENCH_RESULTS_PATH) if "task_id" in record}

    return {
        "metr_task_sources": metr_task_sources,
        "mlebench_task_ids": mlebench_task_ids,
        "gdpval_task_ids": gdpval_task_ids,
        "swebench_task_ids": swebench_task_ids,
        "cybench_task_ids": cybench_task_ids,
    }


def resolve_task_source(task_id: str, lookup: Dict[str, Dict[str, str]]) -> str:
    """Assign task_source using the same precedence as analysis.ipynb."""
    base_task = task_id.split("::", 1)[0]

    if task_id in lookup["metr_task_sources"]:
        return lookup["metr_task_sources"][task_id]
    if task_id in lookup["swebench_task_ids"]:
        return "swebench"
    if task_id in lookup["gdpval_task_ids"]:
        return "gdpval"
    if base_task in lookup["mlebench_task_ids"]:
        return "mlebench"
    if task_id in lookup["cybench_task_ids"]:
        return "cybench"
    return "unknown"


def main() -> None:
    args = parse_args()

    attempts, successes = load_attempts(args.responses)
    unsolved_items = [
        item_id for item_id, n_attempts in attempts.items() if n_attempts > 0 and successes[item_id] == 0
    ]

    difficulties = load_difficulties(args.params)
    unsolved_with_difficulty = [difficulties[item_id] for item_id in unsolved_items if item_id in difficulties]
    avg_difficulty = (
        sum(unsolved_with_difficulty) / len(unsolved_with_difficulty) if unsolved_with_difficulty else float("nan")
    )

    missing_difficulty = sorted(set(unsolved_items) - set(difficulties))
    task_source_lookup = build_task_source_lookup()
    source_counts = Counter(resolve_task_source(item_id, task_source_lookup) for item_id in unsolved_items)

    # Per-source difficulty aggregates
    source_difficulty_sum: Dict[str, float] = Counter()
    source_difficulty_sum_sq: Dict[str, float] = Counter()
    source_difficulty_count: Dict[str, int] = Counter()
    for item_id in unsolved_items:
        source = resolve_task_source(item_id, task_source_lookup)
        if item_id in difficulties:
            source_difficulty_sum[source] += difficulties[item_id]
            source_difficulty_sum_sq[source] += difficulties[item_id] ** 2
            source_difficulty_count[source] += 1

    print(f"Total items with responses: {len(attempts):,}")
    print(f"Items with at least one success: {len([k for k, v in successes.items() if v > 0]):,}")
    print(f"Items no model solved: {len(unsolved_items):,}")
    print(f"Unsolved items with a difficulty score: {len(unsolved_with_difficulty)}")
    print(f"Average difficulty (b) for unsolved items: {avg_difficulty:.3f}")
    if missing_difficulty:
        print(f"Unsolved items missing in params csv: {len(missing_difficulty)}")

    print("\nTask sources for unsolved items (top counts):")
    for source, count in source_counts.most_common(args.top):
        print(f"  {source:20s} {count}")

    print("\nAverage difficulty (b) per task source (with variance):")
    for source, count in source_counts.most_common():
        if source_difficulty_count[source]:
            avg_b = source_difficulty_sum[source] / source_difficulty_count[source]
            mean_sq = source_difficulty_sum_sq[source] / source_difficulty_count[source]
            variance = mean_sq - avg_b**2
            print(
                f"  {source:20s} n={source_difficulty_count[source]:3d} "
                f"avg_b={avg_b:6.3f} var_b={variance:6.3f}"
            )
        else:
            print(f"  {source:20s} n=0   avg_b=   nan  var_b=   nan")


if __name__ == "__main__":
    main()
