#!/usr/bin/env python3
"""
Parse Cybench evaluation logs and generate cybench_normalized_results.jsonl with ONLY
tasks that have actual evaluation data (40 tasks).

This script does NOT use the 171-task FST template. It generates data directly from logs.
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, Tuple, Optional


def extract_task_id_from_path(challenge_path: str) -> str:
    """
    Extract task ID from challenge path.

    Example:
        'benchmark/hackthebox/cyber-apocalypse-2024/crypto/[Very Easy] Dynastic'
        -> 'hackthebox_cyber-apocalypse-2024_crypto_dynastic'
    """
    path = challenge_path.replace('benchmark/', '')
    parts = path.split('/')

    # Clean up the last part (task name) - remove difficulty markers
    if len(parts) > 0:
        task_name = parts[-1]
        task_name = re.sub(r'\[.*?\]\s*', '', task_name)
        parts[-1] = task_name.strip()

    task_id = '_'.join(parts)
    task_id = task_id.lower().replace(' ', '-')

    return task_id


def normalize_model_name(model_name: str) -> str:
    """
    Normalize model name to standard format.
    """
    if 'claude-3-5-sonnet' in model_name or 'claude-3.5-sonnet' in model_name:
        return 'claude-3.5-sonnet'
    elif 'claude-3-opus' in model_name:
        return 'claude-3-opus'
    elif 'gpt-4o' in model_name:
        return 'gpt-4o'
    elif 'o1-preview' in model_name:
        return 'o1-preview'
    elif 'llama-3-70b' in model_name:
        return 'llama-3-70b-chat'
    elif 'llama-3.1-405b' in model_name:
        return 'llama-3.1-405b-instruct'
    elif 'mixtral' in model_name:
        return 'mixtral-8x22b-instruct'
    elif 'gemini-1.5-pro' in model_name:
        return 'gemini-1.5-pro'
    else:
        return model_name


def parse_filename(filename: str) -> Tuple[str, bool, bool]:
    """
    Parse log filename to extract model name, success status, and subtask mode.

    Returns:
        (model_name, is_success, is_subtask_mode)
    """
    parts = filename.replace('_run_log.json', '').split('_')

    # Find where success/fail is
    success_idx = -1
    for i, part in enumerate(parts):
        if part in ['success', 'fail']:
            success_idx = i
            break

    if success_idx == -1:
        raise ValueError(f"Could not find success/fail in filename: {filename}")

    model_name = '_'.join(parts[:success_idx])
    model_name = normalize_model_name(model_name)

    is_success = parts[success_idx] == 'success'
    is_subtask = success_idx + 1 < len(parts) and parts[success_idx + 1] == 'subtask'

    return model_name, is_success, is_subtask


def extract_difficulty_label(challenge_name: str) -> str:
    """
    Extract difficulty label from challenge name.

    Examples:
        '[Very Easy] Dynastic' -> 'very_easy'
        '[Easy] Crushing' -> 'easy'
        '[Medium] Data Siege' -> 'medium'
        '[Hard] Permuted' -> 'hard'
        'chunky' -> 'unlabelled'
    """
    match = re.search(r'\[(.*?)\]', challenge_name)
    if match:
        label = match.group(1).lower().replace(' ', '_')
        return label
    return 'unlabelled'


def parse_log_file(log_path: Path) -> Dict:
    """Parse a single log file and extract relevant information."""
    with open(log_path, 'r') as f:
        data = json.load(f)

    # Handle both 'challenge' and 'task' keys
    challenge_data = data.get('challenge') or data.get('task')
    if challenge_data is None:
        raise ValueError("No 'challenge' or 'task' key found")

    challenge_name = challenge_data.get('name', '')
    difficulty_label = extract_difficulty_label(challenge_name)

    return {
        'challenge_path': challenge_data['path'],
        'challenge_name': challenge_name,
        'difficulty_label': difficulty_label,
        'status': data.get('status'),
        'num_correct_subtasks': data.get('num_correct_subtasks', 0),
        'num_subtasks': data.get('num_subtasks', 0),
    }


def load_fst_data(fst_path: Path) -> Dict[str, float]:
    """Load FST data and return task_id -> human_minutes mapping."""
    fst_mapping = {}
    if fst_path.exists():
        with open(fst_path, 'r') as f:
            for line in f:
                record = json.loads(line)
                fst_mapping[record['task_id']] = record['human_minutes']
    return fst_mapping


def parse_all_logs(data_dir: Path) -> Dict[Tuple[str, str], Dict]:
    """
    Parse all log files in data_by_challenges directory.

    Returns:
        Dict mapping (task_id, model) -> {score, is_subtask, challenge_path, ...}
    """
    results = {}

    for challenge_dir in sorted(data_dir.iterdir()):
        if not challenge_dir.is_dir():
            continue

        for log_file in challenge_dir.glob('*.json'):
            try:
                # Parse filename
                model_name, is_success, is_subtask = parse_filename(log_file.name)

                # Parse log content
                log_data = parse_log_file(log_file)

                # Extract task ID from challenge path
                task_id = extract_task_id_from_path(log_data['challenge_path'])

                # Determine score
                if is_success:
                    if log_data['num_subtasks'] > 0:
                        score = 1.0 if log_data['num_correct_subtasks'] == log_data['num_subtasks'] else 0.0
                    else:
                        score = 1.0
                else:
                    score = 0.0

                # Store result - use unguided mode score if available
                key = (task_id, model_name)
                mode = 'subtask' if is_subtask else 'unguided'

                if key not in results:
                    results[key] = {
                        'task_id': task_id,
                        'model': model_name,
                        'challenge_path': log_data['challenge_path'],
                        'difficulty_label': log_data['difficulty_label'],
                        'unguided_score': None,
                        'subtask_score': None,
                    }

                # Store score for this mode (keep best score)
                score_key = f'{mode}_score'
                if results[key][score_key] is None or score > results[key][score_key]:
                    results[key][score_key] = score

            except Exception as e:
                print(f"Warning: Error parsing {log_file}: {e}")
                continue

    return results


def main():
    parser = argparse.ArgumentParser(description='Parse Cybench evaluation logs')
    parser.add_argument('--exclude-no-success', action='store_true',
                        help='Exclude tasks where no model succeeded')
    parser.add_argument('--use-subtask', action='store_true',
                        help='Use subtask-guided results instead of unguided results')
    args = parser.parse_args()

    data_dir = Path('/home/lfy/BRIDGE/IRT/data/data_by_challenges')
    # Read FST from master file (never overwritten)
    fst_path = Path('/home/lfy/BRIDGE/IRT/data/cybench_fst_master.jsonl')
    output_path = Path('/home/lfy/BRIDGE/IRT/data/cybench_normalized_results.jsonl')
    human_minutes_output = Path('/home/lfy/BRIDGE/IRT/data/cybench_human_minutes_by_task.jsonl')

    print("=" * 80)
    print("Parsing Cybench Evaluation Logs")
    print("=" * 80)

    # Parse all logs
    print(f"\n1. Parsing logs from {data_dir}")
    parsed_results = parse_all_logs(data_dir)
    print(f"   Found {len(parsed_results)} (task_id, model) evaluation pairs")

    # Get unique tasks
    task_ids = sorted(set(task_id for task_id, _ in parsed_results.keys()))
    print(f"   Unique tasks with evaluations: {len(task_ids)}")

    # Get unique models
    models = sorted(set(model for _, model in parsed_results.keys()))
    print(f"   Unique models: {len(models)}")
    for m in models:
        print(f"      - {m}")

    # If --exclude-no-success, filter out tasks with no successes
    if args.exclude_no_success:
        print(f"\n   Filtering: --exclude-no-success enabled")
        # Find tasks with at least one success
        task_success_count = defaultdict(int)
        for (task_id, model), data in parsed_results.items():
            # Use same score selection logic as main loop
            if args.use_subtask:
                score = data['subtask_score']
                if score is None:
                    score = data['unguided_score'] if data['unguided_score'] is not None else 0.0
            else:
                score = data['unguided_score']
                if score is None:
                    score = data['subtask_score'] if data['subtask_score'] is not None else 0.0
            if score >= 1.0:
                task_success_count[task_id] += 1

        tasks_with_success = set(task_id for task_id, count in task_success_count.items() if count > 0)
        tasks_without_success = set(task_ids) - tasks_with_success

        print(f"   Tasks with at least one success: {len(tasks_with_success)}")
        print(f"   Tasks with no successes (excluded): {len(tasks_without_success)}")
        if tasks_without_success:
            for t in sorted(tasks_without_success)[:10]:
                print(f"      - {t}")
            if len(tasks_without_success) > 10:
                print(f"      ... and {len(tasks_without_success) - 10} more")

        # Filter task_ids and parsed_results
        task_ids = sorted(tasks_with_success)
        parsed_results = {k: v for k, v in parsed_results.items() if k[0] in tasks_with_success}

    # Load existing FST data to get human_minutes
    print(f"\n2. Loading FST data from {fst_path}")
    fst_mapping = load_fst_data(fst_path)
    print(f"   Loaded FST for {len(fst_mapping)} tasks")

    # Match FST to our tasks
    matched_fst = {}
    unmatched_tasks = []
    for task_id in task_ids:
        if task_id in fst_mapping:
            matched_fst[task_id] = fst_mapping[task_id]
        else:
            # Try fuzzy match
            found = False
            for fst_task_id, minutes in fst_mapping.items():
                # Check if task names are similar (last part of the ID)
                task_name = task_id.split('_')[-1].replace('-', '')
                fst_name = fst_task_id.split('_')[-1].replace('-', '')
                if task_name == fst_name:
                    matched_fst[task_id] = minutes
                    found = True
                    break
            if not found:
                unmatched_tasks.append(task_id)

    print(f"   Matched FST for {len(matched_fst)}/{len(task_ids)} tasks")
    if unmatched_tasks:
        print(f"   Tasks without FST match: {len(unmatched_tasks)}")
        for t in unmatched_tasks[:5]:
            print(f"      - {t}")

    # Generate normalized results - ONLY for tasks with actual evaluations
    print(f"\n3. Generating normalized results")
    if args.use_subtask:
        print(f"   Using subtask-guided results (--use-subtask)")
    else:
        print(f"   Using unguided results (default)")

    records = []
    for (task_id, model), data in sorted(parsed_results.items()):
        # Select score based on --use-subtask flag
        if args.use_subtask:
            # Prefer subtask score, fallback to unguided
            score = data['subtask_score']
            if score is None:
                score = data['unguided_score'] if data['unguided_score'] is not None else 0.0
        else:
            # Prefer unguided score, fallback to subtask
            score = data['unguided_score']
            if score is None:
                score = data['subtask_score'] if data['subtask_score'] is not None else 0.0

        human_minutes = matched_fst.get(task_id)

        record = {
            "task_id": task_id,
            "run_id": f"cybench_{model}",
            "alias": model,
            "model": model,
            "score_cont": score,
            "score_binarized": int(score),
            "fatal_error_from": None,
            "human_minutes": human_minutes,
            "human_score": 1.0 if human_minutes else None,
            "human_source": "fst" if human_minutes else None,
            "task_source": "Cybench",
            "difficulty_label": data['difficulty_label'],
            "generation_cost": 0.0,
            "human_cost": 0.0,
            "time_limit": None,
            "started_at": None,
            "completed_at": None,
            "task_version": None,
            "equal_task_weight": 1.0 / len(task_ids) if task_ids else 1.0,
            "invsqrt_task_weight": 1.0 / (len(task_ids) ** 0.5) if task_ids else 1.0,
        }
        records.append(record)

    # Write normalized results
    print(f"\n4. Writing {len(records)} records to {output_path}")
    with open(output_path, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')

    # Update human_minutes file to only include the tasks with evaluations
    print(f"\n5. Updating human_minutes file for {len(task_ids)} tasks")
    with open(human_minutes_output, 'w') as f:
        for task_id in sorted(task_ids):
            if task_id in matched_fst:
                record = {"task_id": task_id, "human_minutes": matched_fst[task_id]}
                f.write(json.dumps(record) + '\n')

    tasks_with_fst = sum(1 for t in task_ids if t in matched_fst)
    print(f"   Written {tasks_with_fst} tasks with FST data")

    # Print statistics
    print(f"\n" + "=" * 80)
    print("Statistics")
    print("=" * 80)

    print(f"\nTotal evaluation records: {len(records)}")
    print(f"Unique tasks: {len(task_ids)}")
    print(f"Tasks with FST: {tasks_with_fst}")

    if records:
        success_count = sum(1 for r in records if r['score_binarized'] == 1)
        print(f"Total successes: {success_count}/{len(records)} ({100*success_count/len(records):.1f}%)")

        # Count by model
        print(f"\nSuccess rate by model:")
        model_stats = defaultdict(lambda: {'total': 0, 'success': 0})
        for record in records:
            model = record['model']
            model_stats[model]['total'] += 1
            if record['score_binarized'] == 1:
                model_stats[model]['success'] += 1

        for model in sorted(model_stats.keys()):
            stats = model_stats[model]
            rate = 100.0 * stats['success'] / stats['total'] if stats['total'] > 0 else 0
            print(f"  {model}: {stats['success']}/{stats['total']} ({rate:.1f}%)")

    print(f"\n✓ Results saved to {output_path}")
    print(f"✓ Human minutes saved to {human_minutes_output}")


if __name__ == "__main__":
    main()
