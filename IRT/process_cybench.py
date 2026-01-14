#!/usr/bin/env python3
"""
Process Cybench data for integration into the IRT pipeline.

This script:
1. Parses First Solve Time (FST) data from Cybench
2. Converts FST to human_minutes format
3. Generates human_minutes_by_task.jsonl for Cybench tasks
4. Creates cybench_normalized_results.jsonl for model results (when available)
"""

import json
import re
from pathlib import Path
from typing import Dict, List


def parse_time_to_minutes(time_str: str) -> float:
    """
    Convert time string in format 'H:MM:SS' or 'HH:MM:SS' to minutes.

    Examples:
        '0:09:00' -> 9.0
        '1:05:00' -> 65.0
        '24:43:12' -> 1483.2
    """
    parts = time_str.split(':')
    if len(parts) == 3:
        hours, minutes, seconds = map(int, parts)
        return hours * 60 + minutes + seconds / 60.0
    elif len(parts) == 2:
        minutes, seconds = map(int, parts)
        return minutes + seconds / 60.0
    else:
        raise ValueError(f"Unexpected time format: {time_str}")


def extract_task_id(challenge_dir: str) -> str:
    """
    Extract a clean task ID from the challenge directory path.

    Examples:
        'benchmark/hackthebox/cyber-apocalypse-2024/crypto/[Very Easy] Dynastic'
        -> 'hackthebox_cyber-apocalypse-2024_crypto_dynastic'

        'benchmark/project-sekai-ctf/sekaictf-2023/crypto/noisier-crc'
        -> 'project-sekai-ctf_sekaictf-2023_crypto_noisier-crc'
    """
    # Remove 'benchmark/' prefix
    path = challenge_dir.replace('benchmark/', '')

    # Split by '/'
    parts = path.split('/')

    # Clean up the last part (task name) - remove difficulty markers in brackets
    if len(parts) > 0:
        task_name = parts[-1]
        # Remove brackets and their contents (e.g., '[Very Easy]', '[Hard]')
        task_name = re.sub(r'\[.*?\]\s*', '', task_name)
        parts[-1] = task_name.strip()

    # Join with underscore and clean up
    task_id = '_'.join(parts)
    task_id = task_id.lower().replace(' ', '-')

    return task_id


def load_fst_data(fst_file: Path) -> Dict[str, float]:
    """
    Load FST data from JSON file and convert to task_id -> minutes mapping.
    """
    with open(fst_file, 'r') as f:
        fst_data = json.load(f)

    task_fst = {}
    for entry in fst_data:
        challenge_dir = entry['challenge_dir']
        fst_str = entry['challenge_fastest_solve_time']

        task_id = extract_task_id(challenge_dir)
        minutes = parse_time_to_minutes(fst_str)

        task_fst[task_id] = minutes

    return task_fst


def generate_human_minutes_jsonl(task_fst: Dict[str, float], output_path: Path):
    """
    Generate human_minutes_by_task.jsonl file for Cybench tasks.
    """
    with open(output_path, 'w') as f:
        for task_id, minutes in sorted(task_fst.items()):
            record = {
                "task_id": task_id,
                "human_minutes": minutes
            }
            f.write(json.dumps(record) + '\n')

    print(f"Generated {output_path} with {len(task_fst)} tasks")


def generate_normalized_results_template(task_fst: Dict[str, float], output_path: Path):
    """
    Generate a template for cybench_normalized_results.jsonl.

    This creates placeholder entries that can be populated with actual model results.
    The format matches other normalized results files (swebench, gdpval, mlebench).
    """
    # Placeholder models - these should be replaced with actual model results
    placeholder_models = [
        {"run_id": "cybench_claude-3.5-sonnet", "model": "claude-3.5-sonnet", "alias": "Claude 3.5 Sonnet"},
        {"run_id": "cybench_gpt-4o", "model": "gpt-4o", "alias": "GPT-4o"},
        {"run_id": "cybench_o1-preview", "model": "o1-preview", "alias": "OpenAI o1-preview"},
        {"run_id": "cybench_claude-3-opus", "model": "claude-3-opus", "alias": "Claude 3 Opus"},
    ]

    records = []
    for task_id, minutes in sorted(task_fst.items()):
        for model_info in placeholder_models:
            record = {
                "task_id": task_id,
                "run_id": model_info["run_id"],
                "alias": model_info["alias"],
                "model": model_info["model"],
                "score_cont": 0.0,  # Placeholder - should be 0-1 based on actual results
                "score_binarized": 0,  # Placeholder - should be 0 or 1 based on actual results
                "fatal_error_from": None,
                "human_minutes": minutes,
                "human_score": 1.0,  # FST represents successful human solve
                "human_source": "fst",  # First Solve Time from CTF competition
                "task_source": "Cybench",
                "generation_cost": 0.0,
                "human_cost": 0.0,
                "time_limit": None,
                "started_at": None,
                "completed_at": None,
                "task_version": None,
                "equal_task_weight": 1.0 / len(task_fst),
                "invsqrt_task_weight": 1.0 / (len(task_fst) ** 0.5),
            }
            records.append(record)

    with open(output_path, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')

    print(f"Generated template {output_path} with {len(records)} placeholder records")
    print(f"NOTE: This is a TEMPLATE. Update score_cont and score_binarized with actual model results.")


def load_model_results_from_paper() -> Dict[str, Dict[str, float]]:
    """
    Extract known model results from the Cybench paper.

    Based on published results:
    - Claude 3.5 Sonnet: 17.5% unguided success rate (7 out of 40 tasks)
    - Claude Sonnet 4.5: 76.5% success rate with 10 attempts
    - GPT-4o: ~17.5% unguided success rate
    - Models can only solve tasks with FST <= 11 minutes in unguided mode

    Returns:
        Dict mapping model_name -> {task_id: success (0 or 1)}
    """
    # For now, return empty dict as we don't have task-level results
    # This function can be extended when detailed results become available
    return {}


def main():
    # Paths
    cybench_repo = Path("/tmp/cybench")
    output_dir = Path("/home/lfy/BRIDGE/IRT/data")

    fst_file = cybench_repo / "analytics" / "CTF fastest solve times.json"
    human_minutes_output = output_dir / "cybench_human_minutes_by_task.jsonl"
    normalized_results_output = output_dir / "cybench_normalized_results.jsonl"

    print("=" * 80)
    print("Processing Cybench Data for IRT Pipeline")
    print("=" * 80)

    # Step 1: Load FST data
    print(f"\n1. Loading FST data from {fst_file}")
    task_fst = load_fst_data(fst_file)
    print(f"   Loaded {len(task_fst)} tasks with FST data")
    print(f"   FST range: {min(task_fst.values()):.2f} - {max(task_fst.values()):.2f} minutes")

    # Step 2: Generate human_minutes JSONL
    print(f"\n2. Generating human minutes file")
    generate_human_minutes_jsonl(task_fst, human_minutes_output)

    # Step 3: Generate normalized results template
    print(f"\n3. Generating normalized results template")
    generate_normalized_results_template(task_fst, normalized_results_output)

    print("\n" + "=" * 80)
    print("Processing Complete!")
    print("=" * 80)
    print(f"\nGenerated files:")
    print(f"  - {human_minutes_output}")
    print(f"  - {normalized_results_output}")
    print(f"\nNext steps:")
    print(f"  1. Update {normalized_results_output} with actual model results")
    print(f"  2. Add Cybench to prepare_sparse_pyirt.py")
    print(f"  3. Regenerate all_a_pyirt.jsonl with Cybench data")
    print()


if __name__ == "__main__":
    main()
