#!/usr/bin/env python3
"""Build a sparse py-IRT dataset from multiple run files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class ModelRecord:
    key: str
    display_name: str
    plot: bool


class ModelMapper:
    def __init__(self, mapping_path: Path):
        self._id_lookup: dict[str, ModelRecord] = {}
        self._records: dict[str, ModelRecord] = {}
        self._load(mapping_path)

    def _normalize(self, value: str | None) -> str | None:
        if value is None:
            return None
        value = str(value).strip()
        if not value:
            return None
        return value.lower()

    def _add_identifier(self, identifier: str, record: ModelRecord) -> None:
        normalized = self._normalize(identifier)
        if normalized is None:
            return
        existing = self._id_lookup.get(normalized)
        if existing is None:
            self._id_lookup[normalized] = record
            return
        if existing.key != record.key:
            raise ValueError(
                f"Identifier '{identifier}' maps to multiple model keys: "
                f"{existing.key} vs {record.key}"
            )

    def _load(self, mapping_path: Path) -> None:
        with mapping_path.open(newline="") as f:
            reader = csv.DictReader(f)
            required = {"run_id", "model_key", "display_name", "plot"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"Model mapping file {mapping_path} missing columns: {missing}"
                )
            alias_field = "alias" if "alias" in reader.fieldnames else None
            for row in reader:
                model_key = (row.get("model_key") or "").strip()
                run_id = (row.get("run_id") or "").strip()
                display_name = (row.get("display_name") or "").strip()
                plot_flag = str(row.get("plot", "")).strip().upper() == "T"
                if not model_key:
                    model_key = run_id
                if not model_key:
                    continue
                display = display_name or model_key
                record = self._records.get(model_key)
                if record is None:
                    record = ModelRecord(model_key, display, plot_flag)
                    self._records[model_key] = record
                else:
                    # prefer human-friendly display name if available
                    if display_name and record.display_name != display_name:
                        record.display_name = display_name
                    record.plot = record.plot or plot_flag
                for identifier in (
                    run_id,
                    row.get(alias_field, "") if alias_field else None,
                    record.key,
                    record.display_name,
                ):
                    self._add_identifier(identifier, record)

    def canonicalize(self, *identifiers: str | None) -> ModelRecord | None:
        for identifier in identifiers:
            normalized = self._normalize(identifier)
            if normalized is None:
                continue
            record = self._id_lookup.get(normalized)
            if record:
                return record
        return None

    @property
    def records(self) -> dict[str, ModelRecord]:
        return self._records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert run logs into a sparse py-IRT JSONL file."
    )
    parser.add_argument(
        "--model-mapping",
        type=Path,
        default=Path(__file__).with_name("data") / "model_run_mapping.csv",
        help="CSV describing how run_ids/aliases map to canonical model keys.",
    )
    parser.add_argument(
        "--pyirt-input",
        type=Path,
        action="append",
        default=[],
        help="Existing py-IRT JSONL inputs to include (can be repeated).",
    )
    parser.add_argument(
        "--runs-input",
        type=Path,
        action="append",
        default=[],
        help="Raw JSONL run files (per-task rows) to include (can be repeated).",
    )
    parser.add_argument(
        "--score-column",
        default="score_binarized",
        help="Column to read from runs-input for binary correctness.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the combined py-IRT JSONL file.",
    )
    return parser.parse_args()


def load_pyirt_file(path: Path, mapper: ModelMapper, sink: dict[str, dict[str, float]]):
    with path.open() as f:
        for line in f:
            entry = json.loads(line)
            subject_id = entry["subject_id"]
            record = mapper.canonicalize(subject_id)
            key = record.key if record else subject_id
            responses = sink.setdefault(key, {})
            for item_id, value in entry["responses"].items():
                responses[item_id] = float(value)


def iter_runs(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line in f:
            yield json.loads(line)


def add_run_records(
    paths: list[Path],
    mapper: ModelMapper,
    sink: dict[str, dict[str, float]],
    score_column: str,
) -> tuple[int, int]:
    total_rows = 0
    missing_ids = 0
    conflicts = 0
    for path in paths:
        for record in iter_runs(path):
            total_rows += 1
            run_id = record.get("run_id")
            alias = record.get("alias")
            model_field = record.get("model")
            canonical = mapper.canonicalize(run_id, alias, model_field)
            if canonical is None:
                missing_ids += 1
                canonical_key = str(alias or run_id or model_field or "").strip()
                if not canonical_key:
                    canonical_key = "unknown"
                subject_key = canonical_key
            else:
                subject_key = canonical.key
            score_val = record.get(score_column)
            if score_val is None:
                continue
            try:
                score = float(score_val)
            except (TypeError, ValueError):
                continue
            score = 1.0 if score >= 0.5 else 0.0
            task_id = record.get("task_id")
            if not task_id:
                continue
            responses = sink.setdefault(subject_key, {})
            existing = responses.get(task_id)
            if existing is None or score > existing:
                responses[task_id] = score
            elif existing != score:
                conflicts += 1
    if missing_ids:
        print(
            f"[prepare_sparse_pyirt] {missing_ids} rows used fallback identifiers",
            file=sys.stderr,
        )
    if conflicts:
        print(
            f"[prepare_sparse_pyirt] {conflicts} conflicting responses resolved by max()",
            file=sys.stderr,
        )
    return total_rows, missing_ids


def write_output(path: Path, responses: dict[str, dict[str, float]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for subject_id in sorted(responses):
            payload = {
                "subject_id": subject_id,
                "responses": responses[subject_id],
            }
            f.write(json.dumps(payload) + "\n")


def main() -> None:
    args = parse_args()
    if not args.pyirt_input and not args.runs_input:
        raise SystemExit("Provide at least one --pyirt-input or --runs-input.")

    mapper = ModelMapper(args.model_mapping)
    combined: dict[str, dict[str, float]] = defaultdict(dict)

    for path in args.pyirt_input:
        if not path.exists():
            raise FileNotFoundError(f"py-IRT input {path} does not exist")
        load_pyirt_file(path, mapper, combined)

    rows_loaded = 0
    if args.runs_input:
        for path in args.runs_input:
            if not path.exists():
                raise FileNotFoundError(f"runs input {path} does not exist")
        rows_loaded, missing = add_run_records(
            args.runs_input, mapper, combined, args.score_column
        )
        print(
            f"[prepare_sparse_pyirt] processed {rows_loaded} run rows "
            f"(missing ids: {missing})"
        )

    write_output(args.output, combined)
    print(
        f"[prepare_sparse_pyirt] wrote {len(combined)} subjects and "
        f"{sum(len(v) for v in combined.values())} responses to {args.output}"
    )


if __name__ == "__main__":
    main()
