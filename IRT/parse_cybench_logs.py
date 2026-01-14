#!/usr/bin/env python3
"""
Parse Cybench evaluation logs and update cybench_normalized_results.jsonl with real scores.
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, Tuple


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


def map_directory_to_task_id(dir_name: str) -> str:
    """
    Map directory name to task ID format.

    Examples:
        'htb_dynastic' -> 'hackthebox_cyber-apocalypse-2024_crypto_dynastic'
        'sekai_chunky' -> 'project-sekai-ctf_sekaictf-2023_web_chunky'
        'glacier_avatar' -> 'losfuzzys_glacierctf2023_writeups_misc_avatar'
    """
    # These mappings are based on the directory structure
    if dir_name.startswith('htb_'):
        prefix = 'hackthebox_cyber-apocalypse-2024'
        task_part = dir_name[4:]  # Remove 'htb_'
    elif dir_name.startswith('sekai_'):
        # Need to determine which year (2022 or 2023) - will determine from actual data
        prefix = 'project-sekai-ctf_sekaictf'
        task_part = dir_name[6:]  # Remove 'sekai_'
    elif dir_name.startswith('glacier_'):
        prefix = 'losfuzzys_glacierctf2023_writeups'
        task_part = dir_name[8:]  # Remove 'glacier_'
    elif dir_name.startswith('hkcert_'):
        prefix = 'hkcert-ctf_ctf-challenges_ctf-2022'
        task_part = dir_name[7:]  # Remove 'hkcert_'
    else:
        return dir_name.replace('_', '-')

    return f"{prefix}_{task_part}".replace('_', '-')


def normalize_model_name(model_name: str) -> str:
    """
    Normalize model name to match template format.

    Examples:
        'claude-3-5-sonnet-20240620' -> 'claude-3.5-sonnet'
        'claude-3-opus-20240229' -> 'claude-3-opus'
        'gpt-4o-2024-05-13' -> 'gpt-4o'
        'o1-preview-2024-09-12' -> 'o1-preview'
        'llama-3.1-405b-instruct-turbo' -> 'llama-3.1-405b-instruct-turbo'
    """
    # Map specific model patterns to normalized names
    if 'claude-3-5-sonnet' in model_name or 'claude-3.5-sonnet' in model_name:
        return 'claude-3.5-sonnet'
    elif 'claude-3-opus' in model_name:
        return 'claude-3-opus'
    elif 'gpt-4o' in model_name:
        return 'gpt-4o'
    elif 'o1-preview' in model_name:
        return 'o1-preview'
    else:
        # Keep other models as-is (e.g., llama, mixtral, gemini)
        return model_name


def parse_filename(filename: str) -> Tuple[str, bool, bool]:
    """
    Parse log filename to extract model name, success status, and subtask mode.

    Returns:
        (model_name, is_success, is_subtask_mode)
    """
    # Pattern: {model}_{success/fail}[_subtask]_{challenge_name}_run_log.json
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
    # Normalize model name to match template format
    model_name = normalize_model_name(model_name)

    is_success = parts[success_idx] == 'success'
    is_subtask = success_idx + 1 < len(parts) and parts[success_idx + 1] == 'subtask'

    return model_name, is_success, is_subtask


def load_existing_results(results_path: Path) -> Dict[Tuple[str, str], dict]:
    """
    Load existing cybench_normalized_results.jsonl and return as dict.

    Returns:
        Dict mapping (task_id, model) -> record
    """
    results = {}
    with open(results_path, 'r') as f:
        for line in f:
            record = json.loads(line)
            key = (record['task_id'], record['model'])
            results[key] = record
    return results


def parse_log_file(log_path: Path) -> Dict:
    """Parse a single log file and extract relevant information."""
    with open(log_path, 'r') as f:
        data = json.load(f)

    # Handle both 'challenge' and 'task' keys
    challenge_data = data.get('challenge') or data.get('task')
    if challenge_data is None:
        raise ValueError("No 'challenge' or 'task' key found")

    return {
        'challenge_path': challenge_data['path'],
        'status': data.get('status'),
        'num_correct_subtasks': data.get('num_correct_subtasks', 0),
        'num_subtasks': data.get('num_subtasks', 0),
    }


def parse_all_logs(data_dir: Path) -> Dict[Tuple[str, str], Dict]:
    """
    Parse all log files in data_by_challenges directory.

    Returns:
        Dict mapping (task_id, model) -> {score, is_subtask, ...}
    """
    results = defaultdict(lambda: {'unguided': None, 'subtask': None})

    for challenge_dir in data_dir.iterdir():
        if not challenge_dir.is_dir():
            continue

        dir_name = challenge_dir.name

        for log_file in challenge_dir.glob('*.json'):
            try:
                # Parse filename
                model_name, is_success, is_subtask = parse_filename(log_file.name)

                # Parse log content
                log_data = parse_log_file(log_file)

                # Extract task ID from challenge path
                task_id = extract_task_id_from_path(log_data['challenge_path'])

                # Determine score
                # For success files, score is 1 if all subtasks correct or filename says success
                # For fail files, score is 0
                if is_success:
                    # Check if truly successful (all subtasks correct)
                    if log_data['num_subtasks'] > 0:
                        score = 1.0 if log_data['num_correct_subtasks'] == log_data['num_subtasks'] else 0.0
                    else:
                        score = 1.0  # Filename says success and no subtasks to verify
                else:
                    score = 0.0

                # Store result
                mode = 'subtask' if is_subtask else 'unguided'
                key = (task_id, model_name)

                # Keep the best result (1 > 0)
                if results[key][mode] is None or score > results[key][mode]['score']:
                    results[key][mode] = {
                        'score': score,
                        'num_correct': log_data['num_correct_subtasks'],
                        'num_total': log_data['num_subtasks'],
                    }

            except Exception as e:
                print(f"Warning: Error parsing {log_file}: {e}")
                continue

    return dict(results)


def main():
    data_dir = Path('/home/lfy/BRIDGE/IRT/data/data_by_challenges')
    results_path = Path('/home/lfy/BRIDGE/IRT/data/cybench_normalized_results.jsonl')
    output_path = Path('/home/lfy/BRIDGE/IRT/data/cybench_normalized_results.jsonl')

    print("=" * 80)
    print("Parsing Cybench Evaluation Logs")
    print("=" * 80)

    # Parse all logs
    print(f"\n1. Parsing logs from {data_dir}")
    parsed_results = parse_all_logs(data_dir)
    print(f"   Found results for {len(parsed_results)} (task_id, model) pairs")

    # Load existing template
    print(f"\n2. Loading existing template from {results_path}")
    existing_results = load_existing_results(results_path)
    print(f"   Loaded {len(existing_results)} template records")

    # Update scores
    print(f"\n3. Updating scores with parsed results")
    updated_count = 0
    for (task_id, model), modes in parsed_results.items():
        key = (task_id, model)
        if key in existing_results:
            # Use unguided score if available, otherwise subtask score
            if modes['unguided'] is not None:
                score = modes['unguided']['score']
            elif modes['subtask'] is not None:
                score = modes['subtask']['score']
            else:
                continue

            existing_results[key]['score_cont'] = score
            existing_results[key]['score_binarized'] = int(score)
            updated_count += 1
        else:
            print(f"   Warning: No template entry for ({task_id}, {model})")

    print(f"   Updated {updated_count} records")

    # Write updated results
    print(f"\n4. Writing updated results to {output_path}")
    with open(output_path, 'w') as f:
        for record in sorted(existing_results.values(), key=lambda r: (r['task_id'], r['model'])):
            f.write(json.dumps(record) + '\n')

    # Print statistics
    print(f"\n" + "=" * 80)
    print("Statistics")
    print("=" * 80)

    total_records = len(existing_results)
    updated_records = sum(1 for r in existing_results.values() if r['score_binarized'] > 0)

    print(f"Total records: {total_records}")
    print(f"Records with success (score=1): {updated_records}")
    print(f"Records with failure (score=0): {total_records - updated_records}")

    # Count by model
    model_stats = defaultdict(lambda: {'total': 0, 'success': 0})
    for record in existing_results.values():
        model = record['model']
        model_stats[model]['total'] += 1
        if record['score_binarized'] == 1:
            model_stats[model]['success'] += 1

    print(f"\nSuccess rate by model:")
    for model in sorted(model_stats.keys()):
        stats = model_stats[model]
        rate = 100.0 * stats['success'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {model}: {stats['success']}/{stats['total']} ({rate:.1f}%)")

    print(f"\n✓ Updated results saved to {output_path}")


if __name__ == "__main__":
    main()
