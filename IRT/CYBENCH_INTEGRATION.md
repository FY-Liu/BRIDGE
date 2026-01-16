# Cybench Integration into IRT Pipeline

## Overview

Cybench data has been successfully integrated into the IRT (Item Response Theory) pipeline. Cybench is a cybersecurity CTF (Capture The Flag) benchmark containing 40 professional-level challenges with human First Solve Time (FST) data ranging from 2 minutes to 46+ hours.

## What Was Done

### 1. Data Extraction and Processing

- **Cloned Cybench Repository**: Downloaded the full Cybench repository from https://github.com/andyzorigin/cybench
- **Extracted FST Data**: Processed 171 tasks from `analytics/CTF fastest solve times.json`
- **Created Processing Script**: `process_cybench.py` handles conversion of Cybench data to IRT format

### 2. Generated Files

Three new data files were created in the `data/` directory:

#### `cybench_human_minutes_by_task.jsonl`
- Contains 171 tasks with their human solve times (FST converted to minutes)
- Format: `{"task_id": "...", "human_minutes": ...}`
- Task IDs follow pattern: `source_competition_category_taskname`
- Examples:
  - `hackthebox_cyber-apocalypse-2024_crypto_dynastic` → 9.0 minutes
  - `project-sekai-ctf_sekaictf-2023_web_chunky` → 105.65 minutes
  - `losfuzzys_glacierctf2023_writeups_pwn_35shadesofwasm` → 1321.02 minutes

#### `cybench_normalized_results.jsonl`
- Template file with 684 placeholder records (171 tasks × 4 models)
- Format matches other benchmarks (swebench, gdpval, mlebench)
- **NOTE**: Contains placeholder model results - needs to be updated with actual Cybench evaluation data
- Placeholder models included:
  - Claude 3.5 Sonnet
  - GPT-4o
  - OpenAI o1-preview
  - Claude 3 Opus

### 3. Updated IRT Pipeline Scripts

#### `prepare_sparse_pyirt.py`
Added Cybench support with:
- New `--cybench-input` argument
- `add_cybench_results()` function to process Cybench data
- Automatic loading from `data/cybench_normalized_results.jsonl`
- Integration with existing model mapping system

#### `process_cybench.py`
New script that:
- Parses FST data from Cybench repository
- Converts time formats (H:MM:SS) to minutes
- Generates clean task IDs from directory paths
- Creates both human_minutes and normalized_results files

## Data Statistics

### Cybench Dataset
- **Total Tasks**: 171 CTF challenges
- **FST Range**: 2.0 - 2769.90 minutes (2 min to 46+ hours)
- **Sources**:
  - HackTheBox Cyber Apocalypse 2024
  - Project Sekai CTF 2022/2023
  - LosFuzzys Glacier CTF 2023
  - HKCERT CTF 2022
- **Categories**: Crypto, Web, Forensics, Reversing, Pwn, Misc, etc.

### Integration Test Results
```
[prepare_sparse_pyirt] processed 684 Cybench rows (missing ids: 0)
[prepare_sparse_pyirt] wrote 179 subjects and 87013 responses
```

## Known Cybench Performance (from Paper)

Based on published research (August 2024):
- **Claude 3.5 Sonnet**: 17.5% unguided success rate (7/40 tasks)
- **GPT-4o**: ~17.5% unguided success rate
- **o1-preview**: Limited success on easier tasks
- **Key Finding**: Models can only solve tasks with FST ≤ 11 minutes in unguided mode
- **Updated (2025)**: Claude Sonnet 4.5 achieves 76.5% success with 10 attempts

## Next Steps

### To Use Cybench in IRT Analysis

1. **Update with Real Model Results**:
   ```bash
   # Edit data/cybench_normalized_results.jsonl
   # Update score_cont and score_binarized fields with actual evaluation results
   ```

2. **Generate IRT Input File**:
   ```bash
   python3 prepare_sparse_pyirt.py \
     --model-mapping data/model_run_mapping.json \
     --pyirt-input data/swe_a_pyirt.jsonl \
     --runs-input data/all_runs.jsonl \
     --gdpval-input data/gdpval_normalized_results.jsonl \
     --mlebench-input data/mlebench_normalized_results.jsonl \
     --cybench-input data/cybench_normalized_results.jsonl \
     --output data/all_with_cybench_pyirt.jsonl \
     --print-subject-counts \
     --keep-unmapped-pyirt-subjects
   ```

3. **Fit IRT Model**:
   ```bash
   python3 fit_irt.py --input_path data/all_with_cybench_pyirt.jsonl
   ```

4. **Merge Human Minutes**:
   ```bash
   # Combine Cybench human minutes with existing ones
   cat data/human_minutes_by_task.jsonl \
       data/cybench_human_minutes_by_task.jsonl \
       > data/combined_human_minutes.jsonl

   # Merge into IRT output
   python3 merge_human_minutes.py \
     --csv params/all_with_cybench_pyirt.csv \
     --jsonl data/combined_human_minutes.jsonl
   ```

5. **Run Analysis**:
   - Update `analysis.ipynb` to include Cybench tasks
   - Tasks will be identified by source: `task_source == 'Cybench'`

### To Add Real Cybench Results

The placeholder file currently has all scores set to 0. To add real evaluation data:

1. Run models on Cybench benchmark using the official framework
2. Extract success/failure for each task
3. Update `data/cybench_normalized_results.jsonl`:
   - Set `score_binarized` to 1 (success) or 0 (failure)
   - Set `score_cont` to the same value (or a continuous score if available)

Example record structure:
```json
{
  "task_id": "hackthebox_cyber-apocalypse-2024_crypto_dynastic",
  "run_id": "cybench_claude-3.5-sonnet",
  "alias": "Claude 3.5 Sonnet",
  "model": "claude-3.5-sonnet",
  "score_cont": 1.0,  // ← UPDATE THIS
  "score_binarized": 1,  // ← UPDATE THIS
  "human_minutes": 9.0,
  "human_source": "fst",
  "task_source": "Cybench",
  ...
}
```

## Files Modified/Created

### Created:
- `process_cybench.py` - Cybench data processing script
- `data/cybench_human_minutes_by_task.jsonl` - Human solve times
- `data/cybench_normalized_results.jsonl` - Model results template
- `CYBENCH_INTEGRATION.md` - This documentation

### Modified:
- `prepare_sparse_pyirt.py` - Added Cybench support

### Temporary (can be deleted):
- `/tmp/cybench/` - Cloned repository (171 MB)

## Resources

- **Cybench Paper**: https://arxiv.org/abs/2408.08926
- **GitHub**: https://github.com/andyzorigin/cybench
- **Website**: https://cybench.github.io
- **Published Results**: Tasks in Appendix C/I of the paper

## Task ID Format

Cybench task IDs are derived from directory paths:
```
benchmark/hackthebox/cyber-apocalypse-2024/crypto/[Very Easy] Dynastic
  ↓
hackthebox_cyber-apocalypse-2024_crypto_dynastic
```

Difficulty markers in brackets (e.g., `[Very Easy]`, `[Hard]`) are removed from task IDs.

## Integration with Existing Benchmarks

Cybench data integrates seamlessly with existing benchmarks:
- **SWE-Bench**: 500 tasks (software engineering)
- **MLE-Bench**: 114 task-metrics (ML engineering)
- **GDPval**: 220 tasks (economic value)
- **METR**: 170 tasks (SWAA, HCAST, RE-Bench)
- **Cybench**: 171 tasks (cybersecurity CTF)

Total with Cybench: **1,175 tasks** across 5 diverse domains

## Summary

Cybench has been successfully integrated into the IRT pipeline and is ready for analysis once actual model evaluation results are available. The current template provides the structure and can be populated with real scores as models are evaluated on the Cybench benchmark.
